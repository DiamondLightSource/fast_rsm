"""
This file contains a suite of utility functions for processing data acquired
specifically at Diamond.
"""
from types import SimpleNamespace
from typing import Tuple
from datetime import datetime
from time import time
import time as toptime
import logging
import os
import sys
import multiprocessing
from pathlib import Path
import numpy as np
import h5py
from diffraction_utils import Frame, Region
from fast_rsm.binning import finite_diff_grid
from fast_rsm.experiment import Experiment
from fast_rsm.config_loader import check_config_schema,experiment_config, parse_setup_file
from fast_rsm.pyfai_interface import pyfai_static_qmap ,pyfai_static_exitangles,\
pyfai_static_ivsq,pyfai_moving_qmap_smm_new,pyfai_moving_exitangles_smm,\
pyfai_moving_ivsq_smm_new,save_config_variables,createponi


def setup_processing(exp_setup_file: Path,job_file_path:str, scan_numbers: list ):
    """
    uses process configuration to create an experiment object 
    and setup a logger if requested
    returns :  Experiment, config, logger
    """
    cfg = create_process_config(exp_setup_file,job_file_path,scan_numbers)
    debug_logger=logging.getLogger('fastrsm_debug')
    experiment,cfg=create_experiment(cfg)


    return experiment, cfg, debug_logger



def create_process_config(exp_setup_file: Path,job_file_path:str ,scan_numbers: list):
    """
    creates a simplenamespace configuration settings object
    """
    process_settings=parse_setup_file(exp_setup_file)
    default_config=experiment_config(scan_numbers)
    default_config['full_path']=job_file_path
    filtered_settings={k:v for k,v in process_settings.items() if v is not None}
    default_config.update(filtered_settings)
    check_config_schema(default_config)

    cfg = SimpleNamespace(**default_config)
    with open(cfg.full_path,encoding='utf-8') as f:
        cfg.joblines = f.readlines()

    cfg.dps_centres = [cfg.dpsx_central_pixel,
                   cfg.dpsy_central_pixel, cfg.dpsz_central_pixel]
    cfg.oop = initial_value_checks(cfg.dps_centres,cfg.cylinder_axis,\
                                    cfg.setup, cfg.output_file_size)

    # Max number of cores available for processing.
    cfg.num_threads = multiprocessing.cpu_count()
    cfg.pythonlocation = sys.executable

    cfg.mask_regions_list, cfg.specific_pixels,cfg.mask_regions = make_mask_lists(
        cfg.specific_pixels, cfg.mask_regions)
    cfg.data_dir = Path(cfg.local_data_path)

    return cfg




def check_folder(folder):
    """
    check individual scan folders for corrupt .h5 files
    """
    corrupt_files=[]
    for file in folder:
        try:
            f=h5py.File(file)
            f.close()
        except OSError:
            corrupt_files.append(file)
    return len(corrupt_files)


def check_folder_corruption(folderpath: Path):
    """
    check any folders associated with scan
    
    """
    corrupt_count=0

    corrupt_files=[]
    if not folderpath.exists():
        return corrupt_count,corrupt_files
    subdir_list=[folderpath/f"{foldfile}" for foldfile in os.listdir(folderpath)]
    subdir_filespaths=[path for path in subdir_list if Path(path).exists()]
    for file in subdir_filespaths:
        try:
            f=h5py.File(file)
            f.close()
        except OSError:
            corrupt_files.append(file)
    corrupt_count+= len(corrupt_files)
    return corrupt_count,corrupt_files

def get_good_paths(process_config: SimpleNamespace):
    """
    checks for corrupted .h5 files in scan folder
    """
    cfg=process_config
    good_scan_list=[]
    corrupt_scan_list=[]
    cfg.corrupted_file_list={}
    for scan in cfg.scan_numbers:
        corruption_details=check_folder_corruption(cfg.data_dir / f"i07-{scan}")
        if corruption_details[0]==0:
            good_scan_list.append(scan)
        else:
            corrupt_scan_list.append(scan)
        if len(corruption_details[1])>0:
            cfg.corrupted_file_list[f"i07-{scan}.nxs"]=corruption_details[1]

    cfg.good_nxs_paths = [cfg.data_dir / f"i07-{x}.nxs" for x in good_scan_list]
    cfg.bad_nxs_paths=[cfg.data_dir / f"i07-{x}.nxs" for x in corrupt_scan_list]
    return cfg

def colour_text(colour,string):
    """
    return a colour string to print using chosen colour and string text
    """
    colourkeys = {
        'blue': [
            '\033[34m',
            '\033[0m.'],
        'green': [
            '\033[32m',
            '\033[0m'],
        'red':[
            '\033[31m',
            '\033[0m']}
    start, end = colourkeys[colour]
    return f"{start}{string}{end}"


def data_corruption_warning(cfg: SimpleNamespace):
    """
    give warning in terminal if corrupt scans found
    """
    if len(cfg.bad_nxs_paths)==0:
        return
    print(colour_text('red','='*35))
    print(colour_text('red',"WARNING - the following .h5 files were found to be corrupt\
                       and will be ignored during processing"))
    print(f"Directory: {str(cfg.bad_nxs_paths[0].parent)}")
    for path in cfg.bad_nxs_paths:
        print(f"\t {path.name}")
        for file in cfg.corrupted_file_list[path.name]:
            print(f"--    {file.name}")
        print(colour_text('red','-'*35))
    print(colour_text('red','='*35))
    return

def create_experiment(process_config: SimpleNamespace):
    """
    creates an experiment object from process configuration settings
    """
    cfg=process_config
    # Work out the paths to each of the nexus files. Store as pathlib.Path
    # objects.
    cfg = get_good_paths(cfg)
    data_corruption_warning(cfg)
    # Finally, instantiate the Experiment object.
    experiment = Experiment.from_i07_nxs(
        cfg.good_nxs_paths, cfg.beam_centre, cfg.detector_distance, cfg.setup,
        using_dps=cfg.using_dps, experimental_hutch=cfg.experimental_hutch)

    experiment.mask_pixels(cfg.specific_pixels)
    experiment.mask_edf(cfg.edfmaskfile)
    experiment.mask_regions(cfg.mask_regions_list)

    # adjustment_args = [cfg.detector_distance, cfg.dps_centres, cfg.load_from_dat,\
    # cfg.scan_numbers, cfg.skipscans, cfg.skipimages,
    #                    cfg.slithorratio, cfg.slitvertratio, cfg.data_dir]
    experiment, cfg = standard_adjustments(
        experiment, cfg)
    # grab ub information
    cfg.ubinfo = [
        scan.metadata.data_file.nx_instrument.diffcalchdr for scan in experiment.scans]
    cfg.do_time_check=do_time_check
    return experiment, cfg

def do_time_check(outstring, queue=None,logn=None):
    

    pid = os.getpid()
    allowed = None
    if hasattr(os, "sched_getaffinity"):
        try:
            allowed = sorted(os.sched_getaffinity(0))  # e.g., [0] or [0,1,...,31]
        except BaseException:
            allowed = None

    cur = None
    if hasattr(os, "sched_getcpu"):
        try:
            cur = os.sched_getcpu()  # current CPU core ID
        except BaseException:
            cur = None

    t_wall = toptime.time()
    t_cpu  = toptime.process_time()
    return f"{outstring} pid={pid} allowed={allowed if allowed is not None else 'n/a'} current_cpu={cur} wall={t_wall:.6f} cpu={t_cpu:.6f}"



def initial_value_checks(dps_centres, cylinder_axis, setup, output_file_size):
    """
    does initial checks on dps_centres, cylinder_axis, setup, output_file_size
    sets oop value
    returns: oop
    """
    dpsx_central_pixel, dpsy_central_pixel, dpsz_central_pixel = dps_centres
    # Warn if dps offsets are silly.
    if ((dpsx_central_pixel > 10) or (dpsy_central_pixel > 10) or
            (dpsz_central_pixel > 10)):
        raise ValueError("DPS central pixel units should be meters. Detected "
                         "values greater than 10m")

    # Which synchrotron axis should become the out-of-plane (001) direction.
    # Defaults to 'y'; can be 'x', 'y' or 'z'.
    setup_oops = {'vertical': 'x', 'horizontal': 'y', 'DCD': 'y'}
    if setup in setup_oops:
        oop = setup_oops[setup]
    else:
        raise ValueError(
            "Setup not recognised. Must be 'vertical', 'horizontal' or 'DCD.")
    if output_file_size > 2000:
        raise ValueError("output_file_size must not exceed 2000. "
                         f"Value received was {output_file_size}.")

    # Overwrite the above oop value depending on requested cylinder axis for polar
    # coords.
    if cylinder_axis is not False:
        oop = cylinder_axis

    return oop

def setup_dps(scan,process_config):
    """
    adjust dps settings in scan metadata
    """
    cfg=process_config
    if scan.metadata.data_file.setup == 'DCD':
        # If we're using the DCD and the DPS, our offset calculation is
        # somewhat involved.
        # Work out the in-plane and out-of-plane incident light angles.
        # To do this, first grab a unit vector pointing along the beam.
        lab_frame = Frame(Frame.lab, scan.metadata.diffractometer,
                            coordinates=Frame.cartesian)
        beam_direction = scan.metadata.diffractometer.get_incident_beam(
            lab_frame).array

        # Now do some basic handling of spherical polar coordinates.
        out_of_plane_theta = np.sin(beam_direction[1])
        cos_theta_in_plane = beam_direction[2] / \
            np.cos(out_of_plane_theta)
        in_plane_theta = np.arccos(cos_theta_in_plane)

        # Work out the total displacement from the undeflected beam of the
        # central pixel, in the x and y directions (we know z already).
        # Note that dx, dy are being calculated with signs consistent with
        # synchrotron coordinates.
        total_dx = -cfg.detector_distance * np.tan(in_plane_theta)
        total_dy = cfg.detector_distance * np.tan(out_of_plane_theta)

        # From these values we can compute true DPS offsets.
        dps_off_x = total_dx - cfg.dpsx_central_pixel
        dps_off_y = total_dy - cfg.dpsy_central_pixel

        scan.metadata.data_file.dpsx += dps_off_x
        scan.metadata.data_file.dpsy += dps_off_y
        scan.metadata.data_file.dpsz -= cfg.dpsz_central_pixel
    else:
        # If we aren't using the DCD, our life is much simpler.
        scan.metadata.data_file.dpsx -= cfg.dpsx_central_pixel
        scan.metadata.data_file.dpsy -= cfg.dpsy_central_pixel
        scan.metadata.data_file.dpsz -= cfg.dpsz_central_pixel

def standard_adjustments(experiment, process_config):
    """
    carries out standard adjustments to the Experiment object 
    keeping track of total images,  calculating dps offsets, 
    setting skipscans and skipimages, setting up slitratio values
    """
    cfg=process_config
    # detector_distance, dps_centres, load_from_dat, scan_numbers, skipscans, skipimages, \
    #     slithorratio, slitvertratio, data_dir = adjustment_args
    # dpsx_central_pixel, dpsy_central_pixel, dpsz_central_pixel = dps_centres

    cfg.total_images = 0
    for i, scan in enumerate(experiment.scans):
        cfg.total_images += scan.metadata.data_file.scan_length
        # Deal with the dps offsets.
        if scan.metadata.data_file.using_dps:
            setup_dps(scan,cfg)

        # Load from .dat files if we've been asked.
        if cfg.load_from_dat:
            dat_path = cfg.data_dir / f"{cfg.scan_numbers[i]}.dat"
            scan.metadata.data_file.populate_data_from_dat(dat_path)
        # reads in skip information and skips specified images in specified
        # files

        if cfg.skipscans is not None:
            if int(cfg.scan_numbers[i]) in cfg.skipscans:
                scan.skip_images += cfg.skipimages[np.where(
                    np.array(cfg.skipscans) == int(cfg.scan_numbers[i]))[0][0]]

    if experiment.scans[0].metadata.data_file.is_rotated:
        cfg.slitratios = [cfg.slithorratio, cfg.slitvertratio]
    else:
        cfg.slitratios = [cfg.slitvertratio, cfg.slithorratio]

    return experiment, cfg


def make_mask_lists(specific_pixels, mask_regions):
    """
    takes in specific pixels and mask regions
    makes sure specific pixels x,y is correct way round for analysis
    sets up mask Region objects, and corrects region x,y
    """
    # Prepare the pixel mask. First, deal with any specific pixels that we have.
    # Note that these are defined (x, y) and we need (y, x) which are the
    # (slow, fast) axes. So: first we need to deal with that!
    if specific_pixels is not None:
        specific_pixels = specific_pixels[1], specific_pixels[0]

    # Now deal with any regions that may have been defined.
    mask_regions_list = []
    if mask_regions is not None:
        mask_regions_list = [maskval if isinstance(maskval, Region) \
                             else Region(*maskval) for maskval in mask_regions]
        mask_regions=[[val.x_start,val.x_end,val.y_start,val.y_end] \
                      if isinstance(val,Region) else val for val in mask_regions]
    # Now swap (x, y) for each of the regions.
    if mask_regions_list is not None:
        for region in mask_regions_list:
            region.x_start, region.y_start = region.y_start, region.x_start
            region.x_end, region.y_end = region.y_end, region.x_end

    return mask_regions_list, specific_pixels,mask_regions


def make_new_hdf5(cfg: SimpleNamespace, scan_index: int, name_start: str,
                  experiment: Experiment):
    """
    makes a new hdf5 file using a process_config, scan_index, 
    name  and Experiment 
    """
    name_end = cfg.scan_numbers[scan_index]
    datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    cfg.projected_name = f'{name_start}_{name_end}_{datetime_str}'
    cfg.process_start_time = time()
    experiment.load_curve_values(experiment.scans[scan_index])
    cfg.pyfaiponi = createponi(experiment, cfg.local_output_path)
    return h5py.File(f'{cfg.local_output_path}/{cfg.projected_name}.hdf5', "w")


def run_one_scan_process(cfg, i, experiment, inputscan, runoptions):
    """
    creates a hdf5 file and then sends off a process option to run
    """
    runfunction, namestart, infostring = runoptions
    hf = make_new_hdf5(cfg, i, namestart, experiment)
    runfunction(experiment, hf, inputscan, cfg)
    print(f"saved {infostring} data to\
            {cfg.local_output_path}/{cfg.projected_name}.hdf5")
    total_time = (time() - cfg.process_start_time) / 60
    print(f"\n {infostring} calculations took {total_time} minutes")


def run_scanlist_loop(cfg, experiment, runoptions):
    """
    iterates through list of scans and runs requested process
    options
    """
    for i, scan in enumerate(experiment.scans):
        run_one_scan_process(cfg, i, experiment, scan, runoptions)


def run_scanlist_combined(cfg, experiment, runoptions):
    """
    sends whole list of scans to run requested process together
    """
    scanlist = experiment.scans
    run_one_scan_process(cfg, 0, experiment, scanlist, runoptions)


def get_run_functions(process_config):
    """
    returns a dict of functions for mapping either static or moving, depending
    on configuration requested
    """
    cfg = process_config
    static_functions = {'pyfai_qmap': [pyfai_static_qmap, "Qmap", "2d Qmap"],
                        'pyfai_exitangles': \
                        [pyfai_static_exitangles, "exitmap", "2d exit angle map"],
                        'pyfai_ivsq': [pyfai_static_ivsq, "IvsQ", "1d integration "]}

    moving_functions = {'pyfai_qmap': [pyfai_moving_qmap_smm_new, "Qmap", "2d Qmap"],
                        'pyfai_exitangles':\
                         [pyfai_moving_exitangles_smm, "exitmap", "2d exit angle map"],
                        'pyfai_ivsq': [pyfai_moving_ivsq_smm_new, "IvsQ", "1d integration "]}

    if cfg.map_per_image:
        functions_dict = static_functions
        scanlist_function = run_scanlist_loop
    else:
        functions_dict = moving_functions
        scanlist_function = run_scanlist_combined

    return functions_dict, scanlist_function


def run_process_list(experiment, process_config):
    """
    separate function for sending of jobs defined by process output list and input arguments
    """
    cfg = process_config

    functions_dict, scanlist_function = get_run_functions(cfg)
    pyfai_options = ['pyfai_qmap', 'pyfai_exitangles', 'pyfai_ivsq']
    for output in cfg.process_outputs:
        if output in pyfai_options:
            runoptions = functions_dict[output]
            scanlist_function(cfg, experiment, runoptions)

    if 'full_reciprocal_map' in cfg.process_outputs:
        run_full_map_process(experiment, cfg)


def run_full_map_process(experiment, cfg):
    """
    runs full reciprocal map processing using a process configuration
    """
    processing_dir = Path(cfg.local_output_path)

    # Here we calculate a sensible file name that hasn't been taken.
    i = 0
    save_file_name = f"mapped_scan_{cfg.scan_numbers[0]}_{i}"
    save_path = processing_dir / save_file_name
    # Make sure that this name hasn't been used in the past.
    extensions = [".hdf5", ".npy", ".vtk", "_l.txt", "_tth.txt", "_Q.txt", ""]
    while any(os.path.exists(str(save_path) + ext) for ext in extensions):
        i += 1
        save_file_name = f"mapped_scan_{cfg.scan_numbers[0]}_{i}"
        save_path = processing_dir / save_file_name

        if i > 1e7:
            raise ValueError(
                "Either you tried to save this file 10000000 times, or something "
                "went wrong. I'm going with the latter, but exiting out anyway.")
    map_frame = Frame(frame_name=cfg.frame_name, coordinates=cfg.coordinates)
    start_time = time()
    # Calculate and save a binned reciprocal space map, if requested.
    cfg.mapped_data = experiment.binned_reciprocal_space_map_smm(
        cfg.num_threads, map_frame, cfg,
        output_file_size=cfg.output_file_size, oop=cfg.oop,
        min_intensity_mask=cfg.min_intensity,
        output_file_name=save_path,
        volume_start=cfg.volume_start, volume_stop=cfg.volume_stop,
        volume_step=cfg.volume_step,
        map_each_image=cfg.map_per_image)

    save_binoviewer_hdf5(str(save_path)+'.hdf5', cfg)
    print(f"\nSaved binoviewer file to {save_path}.hdf5.\n")

    # Finally, print that it's finished We'll use this to work out when the
    # processing is done.
    total_time = time() - start_time
    print(f"\nProcessing took {total_time}s")
    print(
        f"This corresponds to {total_time*1000/cfg.total_images}ms per image.\n")


def save_binoviewer_hdf5(output_path: str, process_config: SimpleNamespace):
    """
    Saves the .npy file as a binoviewer-readable hdf5 file.
    """

    cfg = process_config
    # Load the volume and the bounds.
    volume, start, stop, step = cfg.mapped_data
    # cfg.mapped_data

    # binoviewer expects float64s with no NaNs.
    volume = volume.astype(np.float64)

    # Allow binoviewer to generate the NaNs naturally.
    contributions = np.empty_like(volume)
    contributions[np.isnan(volume)] = 0
    contributions[~np.isnan(volume)] = 1
    volume = np.nan_to_num(volume)

    # make sure to use consistent conventions for the grid.
    # internally, the used grid is defined by np.arange(start, stop, step)
    # which may be missing the last element!
    true_grid = finite_diff_grid(start, stop, step)
    binoviewer_step = step
    binoviewer_start = [interval[0] for interval in true_grid]
    binoviewer_start_int = [int(np.floor(start[i] / step[i]))
                            for i in range(3)]
    binoviewer_stop_int = [
        int(binoviewer_start_int[i] + volume.shape[i] - 1)
        for i in range(3)
    ]
    binoviewer_stop = [binoviewer_stop_int[i] * binoviewer_step[i]
                       for i in range(3)]

    # Make h, k and l arrays in the expected format.
    x_arr, y_arr, z_arr = (
        tuple(np.array([i, binoviewer_start[i], binoviewer_stop[i],
                        binoviewer_step[i],
                        float(binoviewer_start_int[i]),  # binoviewer
                        float(binoviewer_stop_int[i])])  # uses int()
              for i in range(3))
    )

    labels={'hkl':['h','k','l'],'qxqyqz':['Qx','Qy','Qz']}

    with h5py.File(output_path, "w") as hf:

        # Add metadata to the root group
        hf.attrs['file_time'] = datetime.now().isoformat()
        hf.attrs['h5py_version'] = h5py.version.version
        hf.attrs['HDF5_version'] = h5py.version.hdf5_version

        # needs to still be called binoculars for compatibility
        # make edits in binoviewer so it accepts binoviewer moving forward
        binoculars_group = hf.create_group("binoculars")
        binoculars_group.attrs['type'] = 'Space'
        axes_group = binoculars_group.create_group("axes")
        axes_labels=labels[cfg.frame_name]
        axes_datasets = {axes_labels[0]: x_arr,\
                         axes_labels[1]: y_arr,\
                         axes_labels[2]: z_arr}
        for name, data in axes_datasets.items():
            axes_group.create_dataset(name, data=data)

        binoculars_group.create_dataset("contributions", data=contributions)
        binoculars_group.create_dataset("counts", data=volume)
        save_config_variables(hf, cfg)


def get_volume_and_bounds(path_to_npy: str) -> Tuple[np.ndarray]:
    """
    Takes the path to a .npy file. Loads the volume stored in the .npy file, and
    also grabs the definition of the corresponding finite differences volume
    from the bounds file.

    Args:
        path_to_npy:
            Path to the .npy file containing the volume of interest.

    Returns:
        A tuple taking the form (volume, start, stop, step), where each element
        of the tuple is a numpy array of values. The first value returned is the
        full reciprocal space volume, and could be very large (up to ~GB).
    """
    # In case we were given a pathlib.path.
    path_to_npy = str(path_to_npy)
    # Work out the path to the bounds file.
    path_to_bounds = path_to_npy[:-4] + "_bounds.txt"

    # Load the data.
    volume = np.load(path_to_npy)
    q_bounds = np.loadtxt(path_to_bounds)

    # Scrape the start/stop/step values.
    start, stop, step = q_bounds[:, 0], q_bounds[:, 1], q_bounds[:, 2]

    # And return what we were asked for!
    return volume, start, stop, step


