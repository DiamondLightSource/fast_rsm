"""
This file contains a suite of utility functions for processing data acquired
specifically at Diamond.
"""

import os
import sys
import multiprocessing
from pathlib import Path
import numpy as np
import yaml
from types import SimpleNamespace
from typing import Tuple
import h5py
import fast_histogram
from scipy.constants import physical_constants
from datetime import datetime
import nexusformat.nexus as nx
from diffraction_utils import Frame, Region
from fast_rsm.binning import weighted_bin_1d, finite_diff_grid
from fast_rsm.pyfai_interface import *
from fast_rsm.experiment import Experiment
from fast_rsm.logging_config import configure_logging,get_frsm_logger


# # The frame/coordinate system you want the map to be carried out in.
# # Options for frame_name argument are:
# #     Frame.hkl     (map into hkl space - requires UB matrix in nexus file)
# #     Frame.sample_holder   (standard map into 1/Å)
# #     Frame.lab     (map into frame attached to lab.)
# #
# # Options for coordinates argument are:
# #     Frame.cartesian   (normal cartesian coords: hkl, Qx Qy Qz, etc.)
# #     Frame.polar       (cylindrical polar with cylinder axis set by the
# #                        cylinder_axis variable)
# #
# # Frame.polar will give an output like a more general version of PyFAI.
# # Frame.cartesian is for hkl maps and Qx/Qy/Qz. Any combination of frame_name
# # and coordinates will work, so try them out; get a feel for them.
# # Note that if you want something like a q_parallel, q_perpendicular projection,
# # you should choose Frame.lab with cartesian coordinates. From this data, your
# # projection can be easily computed.
# frame_name = Frame.hkl
# coordinates = Frame.cartesian

# # Ignore this unless you selected Frame.polar.
# # This sets the axis about which your polar coordinates will be generated.
# # Options are 'x', 'y' and 'z'. These are the synchrotron coordinates, rotated
# # according to your requested frame_name. For instance, if you select
# # Frame.lab, then 'x', 'y' and 'z' will correspond exactly to the synchrotron
# # coordinate system (z along beam, y up). If you select frame.sample_holder and
# # rotate your sample by an azimuthal angle µ, then 'y' will still be vertically
# # up, but 'x' and 'z' will have been rotated about 'y' by the angle µ.
# # Leave this as "None" if you aren't using cylindrical coordinates.
##cylinder_axis = None



def experiment_config(scans):
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the YAML file
    config_path = os.path.join(base_dir, "default_config.yaml")
    
    # Load the YAML file
    with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    config_dict['scan_numbers']=scans

    return config_dict

    
def create_standard_experiment(input_config: dict,DEBUG_LOG=0):

    cfg=SimpleNamespace(**input_config)
    
    configure_logging(DEBUG_LOG)
    logger=get_frsm_logger()
    logger.debug("creating standard experiment object")
    f = open(cfg.full_path)
    cfg.joblines = f.readlines()
    f.close()
    dps_centres= [cfg.dpsx_central_pixel,cfg.dpsy_central_pixel,cfg.dpsz_central_pixel]
    cfg.oop= initial_value_checks(dps_centres,cfg.cylinder_axis,cfg.setup,cfg.output_file_size)

    # Max number of cores available for processing.
    cfg.num_threads = multiprocessing.cpu_count()
    cfg.pythonlocation = sys.executable
    data_dir = Path(cfg.local_data_path)
    # Work out the paths to each of the nexus files. Store as pathlib.Path objects.
    nxs_paths = [data_dir / f"i07-{x}.nxs" for x in cfg.scan_numbers]

    mask_regions_list,specific_pixels =  make_mask_lists(cfg.specific_pixels,cfg.mask_regions)

    # Finally, instantiate the Experiment object.
    experiment = Experiment.from_i07_nxs(
        nxs_paths, cfg.beam_centre, cfg.detector_distance, cfg.setup,
        using_dps=cfg.using_dps, experimental_hutch=cfg.experimental_hutch)

    experiment.mask_pixels(specific_pixels)
    experiment.mask_edf(cfg.edfmaskfile)
    experiment.mask_regions(mask_regions_list)

    adjustment_args=[cfg.detector_distance,dps_centres,cfg.load_from_dat,cfg.scan_numbers,cfg.skipscans,cfg.skipimages,\
                    cfg.slithorratio,cfg.slitvertratio,data_dir]
    experiment,cfg.total_images,cfg.slitratios=standard_adjustments(experiment,adjustment_args)
    # grab ub information
    cfg.ubinfo = [scan.metadata.data_file.nx_instrument.diffcalchdr for scan in experiment.scans]

    return experiment, cfg, logger

def initial_value_checks(dps_centres,cylinder_axis,setup,output_file_size):
    dpsx_central_pixel,dpsy_central_pixel,dpsz_central_pixel=dps_centres
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
    if cylinder_axis is not None:
        oop = cylinder_axis
    
    return oop

def standard_adjustments(experiment,adjustment_args):
    detector_distance,dps_centres,load_from_dat,scan_numbers,skipscans,skipimages,\
         slithorratio,slitvertratio,data_dir=adjustment_args
    dpsx_central_pixel,dpsy_central_pixel,dpsz_central_pixel=dps_centres

    total_images = 0
    for i, scan in enumerate(experiment.scans):
        total_images += scan.metadata.data_file.scan_length
        # Deal with the dps offsets.
        if scan.metadata.data_file.using_dps:
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
                cos_theta_in_plane = beam_direction[2] / np.cos(out_of_plane_theta)
                in_plane_theta = np.arccos(cos_theta_in_plane)

                # Work out the total displacement from the undeflected beam of the
                # central pixel, in the x and y directions (we know z already).
                # Note that dx, dy are being calculated with signs consistent with
                # synchrotron coordinates.
                total_dx = -detector_distance * np.tan(in_plane_theta)
                total_dy = detector_distance * np.tan(out_of_plane_theta)

                # From these values we can compute true DPS offsets.
                dps_off_x = total_dx - dpsx_central_pixel
                dps_off_y = total_dy - dpsy_central_pixel

                scan.metadata.data_file.dpsx += dps_off_x
                scan.metadata.data_file.dpsy += dps_off_y
                scan.metadata.data_file.dpsz -= dpsz_central_pixel
            else:
                # If we aren't using the DCD, our life is much simpler.
                scan.metadata.data_file.dpsx -= dpsx_central_pixel
                scan.metadata.data_file.dpsy -= dpsy_central_pixel
                scan.metadata.data_file.dpsz -= dpsz_central_pixel

            # Load from .dat files if we've been asked.
            if load_from_dat:
                dat_path = data_dir / f"{scan_numbers[i]}.dat"
                scan.metadata.data_file.populate_data_from_dat(dat_path)
        # reads in skip information and skips specified images in specified files

        if skipscans is not None:
            if (int(scan_numbers[i]) in skipscans):
                scan.skip_images += skipimages[np.where(
                    np.array(skipscans) == int(scan_numbers[i]))[0][0]]
                
    if experiment.scans[0].metadata.data_file.is_rotated:
        slitratios = [slithorratio, slitvertratio]
    else:
        slitratios = [slitvertratio, slithorratio]

    return experiment,total_images,slitratios

def make_mask_lists(specific_pixels,mask_regions):
    # Prepare the pixel mask. First, deal with any specific pixels that we have.
    # Note that these are defined (x, y) and we need (y, x) which are the
    # (slow, fast) axes. So: first we need to deal with that!
    if specific_pixels is not None:
        specific_pixels = specific_pixels[1], specific_pixels[0]

    # Now deal with any regions that may have been defined.
    mask_regions_list = []
    if mask_regions is not None:
        mask_regions_list = [maskval if isinstance(
            maskval, Region) else Region(*maskval) for maskval in mask_regions]

    # Now swap (x, y) for each of the regions.
    if mask_regions_list is not None:
        for region in mask_regions_list:
            region.x_start, region.y_start = region.y_start, region.x_start
            region.x_end, region.y_end = region.y_end, region.x_end
    
    return mask_regions_list,specific_pixels

def make_new_hdf5(cfg: SimpleNamespace ,scan_index: int, name_start: str,\
                  experiment: Experiment):
    name_end = cfg.scan_numbers[scan_index]
    datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    cfg.projected_name = f'{name_start}_{name_end}_{datetime_str}'
    cfg.process_start_time= time()
    experiment.load_curve_values(experiment.scans[scan_index])
    cfg.pyfaiponi =createponi( experiment,cfg.local_output_path)
    return h5py.File(f'{cfg.local_output_path}/{cfg.projected_name}.hdf5', "w")


def run_process_list(experiment,process_config):
    """
    separate function for sending of jobs defined by process output list and input arguments
    """
    cfg=process_config
    logger.debug("entered run_process_list")
    # check for deprecated GIWAXS functions and print message if needed
    for output in cfg.process_outputs:
        print(deprecation_msg(output))
    
    if ('pyfai_qmap' in cfg.process_outputs) & (cfg.map_per_image == True):
        for i, scan in enumerate(experiment.scans):
            hf = make_new_hdf5(cfg,i,'Qmap',experiment)
            pyfai_static_qmap(experiment,hf, scan, cfg)
            print(f"saved 2d map  data to\
                   {cfg.local_output_path}/{cfg.projected_name}.hdf5")
            total_time = time() - cfg.process_start_time
            print(f"\n 2d Q map calculations took {total_time}s")

    if ('pyfai_qmap' in cfg.process_outputs) & (cfg.map_per_image == False):
        scanlist = experiment.scans
        hf = make_new_hdf5(cfg,0,'Qmap',experiment)
        pyfai_moving_qmap_smm(experiment,hf, scanlist, cfg)
        print(f"saved 2d map data to {cfg.local_output_path}/{cfg.projected_name}.hdf5")

        total_time = time() - cfg.process_start_time
        print(f"\n 2d Q map calculation took {total_time}s")

    if ('pyfai_ivsq' in cfg.process_outputs) & (cfg.map_per_image == True):
        for i, scan in enumerate(experiment.scans):
            name_end = cfg.scan_numbers[i]
            datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            projected_name = f'IvsQ_{name_end}_{datetime_str}'
            hf = h5py.File(f'{cfg.local_output_path}/{projected_name}.hdf5', "w")
            process_start_time = time()
            experiment.load_curve_values(scan)
            PYFAI_PONI =createponi( experiment,cfg.local_output_path)
            pyfai_static_ivsq(experiment,  hf, scan, cfg.num_threads,\
                            cfg.local_output_path, PYFAI_PONI, cfg.ivqbins, cfg.qmapbins)
            save_config_variables(hf, cfg)
            hf.close()
            print(
                f"saved 1d integration data to {cfg.local_output_path}/{projected_name}.hdf5")
            total_time = time() - process_start_time
            print(f"\n Azimuthal integrations took {total_time}s")

    if ('pyfai_ivsq' in cfg.process_outputs) & (cfg.map_per_image == False):
        scanlist = experiment.scans
        name_end = cfg.scan_numbers[0]
        datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        projected_name = f'IvsQ_{name_end}_{datetime_str}'
        hf = h5py.File(f'{cfg.local_output_path}/{projected_name}.hdf5', "w")
        process_start_time = time()
        experiment.load_curve_values(scanlist[0])
        PYFAI_PONI =createponi( experiment,cfg.local_output_path)
        pyfai_moving_ivsq_smm(experiment, hf, scanlist, cfg.num_threads,\
            cfg.local_output_path, PYFAI_PONI, cfg.radialrange, cfg.radialstepval,\
            cfg.qmapbins, slitdistratios=cfg.slitratios)
        save_config_variables(hf, cfg)
        hf.close()
        print(
            f"saved 1d integration data to {cfg.local_output_path}/{projected_name}.hdf5")
        total_time = time() - process_start_time
        print(f"\n Azimuthal integration took {total_time}s")

    if ('pyfai_exitangles' in cfg.process_outputs) & (cfg.map_per_image == True):
        for i, scan in enumerate(experiment.scans):
            name_end = cfg.scan_numbers[i]
            datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
            projected_name = f'exitmap_{name_end}_{datetime_str}'
            hf = h5py.File(f'{cfg.local_output_path}/{projected_name}.hdf5', "w")
            process_start_time = time()
            experiment.load_curve_values(scan)
            PYFAI_PONI =createponi( experiment,cfg.local_output_path)
            pyfai_static_exitangles(experiment,hf, scan, cfg.num_threads,\
                                    PYFAI_PONI, cfg.ivqbins, cfg.qmapbins)
            save_config_variables(hf, cfg)
            hf.close()
            print(
                f"saved 2d exit angle map  data to {cfg.local_output_path}/{projected_name}.hdf5")
            total_time = time() - process_start_time
            print(f"\n 2d exit angle map calculations took {total_time}s")

    if ('pyfai_exitangles' in cfg.process_outputs) & (cfg.map_per_image == False):
        scanlist = experiment.scans
        name_end = cfg.scan_numbers[0]
        datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        projected_name = f'exitmap_{name_end}_{datetime_str}'
        hf = h5py.File(f'{cfg.local_output_path}/{projected_name}.hdf5', "w")
        process_start_time = time()
        experiment.load_curve_values(scanlist[0])
        PYFAI_PONI =createponi( experiment,cfg.local_output_path)   
        pyfai_moving_exitangles_smm(experiment,hf, scanlist, cfg)
        save_config_variables(hf, cfg)
        hf.close()
        print(
            f"saved 2d exit angle map  data to {cfg.local_output_path}/{projected_name}.hdf5")
        total_time = time() - process_start_time
        print(f"\n 2d exit angle map calculations took {total_time}s")

    if 'full_reciprocal_map' in cfg.process_outputs:
        
        processing_dir = Path(cfg.local_output_path)

        # Here we calculate a sensible file name that hasn't been taken.
        i = 0

        save_file_name = f"mapped_scan_{cfg.scan_numbers[0]}_{i}"
        save_path = processing_dir / save_file_name
        # Make sure that this name hasn't been used in the past.
        extensions = [".npy", ".vtk", "_l.txt", "_tth.txt", "_Q.txt", ""]
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
        experiment.binned_reciprocal_space_map_smm(
            cfg.num_threads, map_frame, cfg,
            output_file_size=cfg.output_file_size, oop=cfg.oop,
            min_intensity_mask=cfg.min_intensity,
            output_file_name=save_path,
            volume_start=cfg.volume_start, volume_stop=cfg.volume_stop,
            volume_step=cfg.volume_step,
            map_each_image=cfg.map_per_image)

        if cfg.save_binoculars_h5 == True:
            outvars = globals()

            save_binoculars_hdf5(str(save_path) + ".npy", str(save_path) +
                                '.hdf5', cfg.joblines, cfg.pythonlocation, outvars)
            print(f"\nSaved BINoculars file to {save_path}.hdf5.\n")

        # Finally, print that it's finished We'll use this to work out when the
        # processing is done.
        total_time = time() - start_time
        print(f"\nProcessing took {total_time}s")
        print(f"This corresponds to {total_time*1000/cfg.total_images}ms per image.\n")

def save_binoculars_hdf5(path_to_npy: np.ndarray,
                         output_path: str, joblines, pythonlocation, outvars=None):
    """
    Saves the .npy file as a binoculars-readable hdf5 file.
    """
    # Load the volume and the bounds.
    volume, start, stop, step = get_volume_and_bounds(path_to_npy)

    # Binoculars expects float64s with no NaNs.
    volume = volume.astype(np.float64)

    # Allow binoculars to generate the NaNs naturally.
    contributions = np.empty_like(volume)
    contributions[np.isnan(volume)] = 0
    contributions[~np.isnan(volume)] = 1
    volume = np.nan_to_num(volume)

    # make sure to use consistent conventions for the grid.
    # internally, the used grid is defined by np.arange(start, stop, step)
    # which may be missing the last element!
    true_grid = finite_diff_grid(start, stop, step)
    binoculars_step = step
    binoculars_start = [interval[0] for interval in true_grid]
    binoculars_start_int = [int(np.floor(start[i] / step[i]))
                            for i in range(3)]
    binoculars_stop_int = [
        int(binoculars_start_int[i] + volume.shape[i] - 1)
        for i in range(3)
    ]
    binoculars_stop = [binoculars_stop_int[i] * binoculars_step[i]
                       for i in range(3)]

    # Make h, k and l arrays in the expected format.
    h_arr, k_arr, l_arr = (
        tuple(np.array([i, binoculars_start[i], binoculars_stop[i],
                        binoculars_step[i],
                        float(binoculars_start_int[i]),  # binoculars
                        float(binoculars_stop_int[i])])  # uses int()
              for i in range(3))
    )

    # Turn those into an axes group.
    axes_group = nx.NXgroup(h=h_arr, k=k_arr, l=l_arr)

    config_group = nx.NXgroup()
    configlist = ['setup', 'experimental_hutch', 'using_dps', 'beam_centre', \
                  'detector_distance', 'dpsx_central_pixel', 'dpsy_central_pixel',\
                'dpsz_central_pixel','local_data_path', 'local_output_path',\
                'output_file_size', 'save_binoculars_h5', 'map_per_image',\
                'volume_start', 'volume_step', 'volume_stop',\
                'load_from_dat', 'edfmaskfile', 'specific_pixels', \
                'mask_regions', 'process_outputs', 'scan_numbers']
    # Get a list of all available variables
    if outvars is not None:
        variables = list(outvars.keys())

        # Iterate through the variables
        for var_name in variables:
            # Check if the variable name is in configlist
            if var_name in configlist:
                # Get the variable value
                var_value = outvars[var_name]

                # Add the variable to config_group
                config_group[var_name] = str(var_value)
        if 'ubinfo' in outvars:
            for i, coll in enumerate(outvars['ubinfo']):
                config_group[f'ubinfo_{i+1}'] = nx.NXgroup()
                config_group[f'ubinfo_{i+1}'][f'lattice_{i+1}'] = coll['diffcalc_lattice']
                config_group[f'ubinfo_{i+1}'][f'u_{i+1}'] = coll['diffcalc_u']
                config_group[f'ubinfo_{i+1}'][f'ub_{i+1}'] = coll['diffcalc_ub']
    config_group['python_version'] = pythonlocation
    config_group['joblines'] = joblines
    # Make a corresponding (mandatory) "binoculars" group.
    binoculars_group = nx.NXgroup(
        axes=axes_group, contributions=contributions,\
              counts=(volume), i07configuration=config_group)
    binoculars_group.attrs['type'] = 'Space'

    # Make a root which contains the binoculars group.
    bin_hdf = nx.NXroot(binoculars=binoculars_group)

    # Save it!
    bin_hdf.save(output_path)

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



#========old obsolete functions
def load_exact_map(
    q_vector_path: str, intensities_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns exact q vectors and their corresponding intensities, after loading
    them from the provided paths.

    Args:
        q_vector_path:
            Path to exact q vectors.
        intensities_path:
            Path to either corrected or uncorrected intensities.

    Returns:
        (q_vectors, intensities)
        Where, if intensities has a shape (1000,), q_vectors has a shape
        (1000, 3). The intensity measured at q_vectors[i] is intensities[i].
    """
    raw_q = np.load(q_vector_path)
    raw_intensities = np.load(intensities_path)
    raw_q = raw_q.reshape((raw_intensities.shape[0], 3))
    return raw_q, raw_intensities


def intensity_vs_q_exact(
        q_vectors: np.ndarray, intensities: np.ndarray, num_bins=1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This routine is currently only available to data acquired when the
    "map_per_image" option is checked. Note that the "map_per_image" option can
    generate enormous quantities of data (generally producing 5x your input
    data, each time you run a calculation). Q vectors and intensities provided
    need to be obtained using the "load_exact_map" function in this module.

    Args:
        q_vectors:
            An array of q_vectors obtained using the "load_exact_map" function.
        intensities_path:
            An array of intensities loaded using the "load_exact_map" function.
        num_bins:
            Desired length of your output Q and Intensity arrays.

    Returns:
        A (intensity, Q) tuple, where both Q and intensity are represented by
        numpy arrays of length num_bins.
    """
    # These arrays could be affected by the call to weighted_bin_1d. Make sure
    # that callers don't have their objects mangled by making copies here.
    q_vectors, intensities = np.copy(q_vectors), np.copy(intensities)

    # Completely ignore all nan values.
    q_vectors = q_vectors[~np.isnan(intensities)]
    intensities = intensities[~np.isnan(intensities)]

    q_lengths = np.linalg.norm(q_vectors, axis=1)

    start = float(np.nanmin(q_lengths))
    stop = float(np.nanmax(q_lengths))

    # Work out the binned intensities. Note that this isn't normalised by the
    # number of times each bin is binned to; we have to do that manually.
    final_intensities = fast_histogram.histogram1d(
        x=q_lengths,
        bins=num_bins,
        range=[start, stop],
        weights=intensities
    )

    # Work out how many times each bin was binned to for normalisation.
    final_intensity_counts = fast_histogram.histogram1d(
        x=q_lengths,
        bins=num_bins,
        range=[start, stop],
    )

    # Carry out the normalisation.
    final_intensities /= final_intensity_counts

    # Also return an array of q vectors to make plotting as easy as possible on
    # the other side of this function call.
    binned_qs = np.linspace(start, stop, num_bins)

    return final_intensities, binned_qs


def qxy_qz_exact(
    q_vectors: str,
    intensities: str,
    qxy_bins: int = 1000,
    qz_bins: int = 1000,
    qz_axis: int = 2
) -> np.ndarray:
    """
    This routine is currently only available to data acquired when the
    "map_per_image" option is checked. Note that the "map_per_image" option can
    generate enormous quantities of data (generally producing 5x your input
    data, each time you run a calculation). Q vectors and intensities provided
    need to be obtained using the "load_exact_map" function in this module.

    Args:
        q_vectors:
            An array of q_vectors obtained using the "load_exact_map" function.
        intensities_path:
            An array of intensities loaded using the "load_exact_map" function.
        qxy_bins:
            How many qxy steps do you want. Should probably be slightly less
            than the horizontal resolution of your detector.
        qz_bins:
            How many qz steps do you want. Should probably be slightly less than
            the vertical resolution of your detector.
        qz_axis:
            Which axis of the q_vectors array should be taken to be qz? Defaults
            to 2.

    Returns:
        (qxy_qz_intensities, qxy_qz_q_values)
        Numpy arrays where the slow axis (axis=0) is in the qxy direction, and
        the fast axis (axis=1) is in the qz direction.
    """
    # qz is an entirely reasonable name in this context.
    # pylint: disable=invalid-name

    # These arrays could be affected by the call to weighted_bin_1d. Make sure
    # that callers don't have their objects mangled by making copies here.
    q_vectors, intensities = np.copy(q_vectors), np.copy(intensities)

    # Completely ignore all nan values.
    q_vectors = q_vectors[~np.isnan(intensities)]
    intensities = intensities[~np.isnan(intensities)]

    # Deal with variable qz axis.
    i, j, k = (qz_axis + 1) % 3, (qz_axis + 2) % 3, qz_axis
    # Create an array of qxy, qz q-vectors.
    qxy = np.sqrt(q_vectors[:, i]**2 + q_vectors[:, j]**2)
    qz = q_vectors[:, k]

    # Run a 2D binning (I couldn't be bothered writing a really quick one, so
    # here I'm just defaulting to the reasonable implementation provided by
    # the fast_histogram library).

    # To do this, we need to know min/max of each of qxy and qz.
    min_qxy = np.min(qxy)
    max_qxy = np.max(qxy)
    min_qz = np.min(qz)
    max_qz = np.max(qz)

    # Now run the binning. This is not normalised.
    qxy_qz_intensities = fast_histogram.histogram2d(
        x=qxy,
        y=qz,
        bins=(qxy_bins, qz_bins),
        range=[[min_qxy, max_qxy], [min_qz, max_qz]],
        weights=intensities
    )

    # Work out how many times we binned into each pixel in the above routine.
    qxy_qz_intensity_counts = fast_histogram.histogram2d(
        x=qxy,
        y=qz,
        bins=(qxy_bins, qz_bins),
        range=[[min_qxy, max_qxy], [min_qz, max_qz]]
    )

    # Normalise by the count of how many times we binned into each pixel.
    qxy_qz_intensities /= qxy_qz_intensity_counts

    # For convenience, also return the corresponding Q values at all pixels.
    binned_qxy = np.linspace(min_qxy, max_qxy, qxy_bins)
    binned_qz = np.linspace(min_qz, max_qz, qz_bins)
    qxy_qz_q_vals = np.meshgrid(binned_qxy, binned_qz, indexing='ij')

    return qxy_qz_intensities, qxy_qz_q_vals


def q_to_theta(q_values: np.ndarray, energy: float) -> np.ndarray:
    """
    Takes a set of q_values IN INVERSE ANGSTROMS and and energy IN ELECTRON
    VOLTS and returns the corresponding theta values (TIMES BY 2 TO GET 2THETA).

    Args:
        q_values:
            The q values in units of inverse angstroms (multiplied by 2pi, for
            those that know the different conventions).
        energy:
            The energy of the incident beam in units of electron volts. Not kev.
            Not joules. Electron volts, please.

    Returns:
        A numpy array of the corresponding theta values (NOT TWO THETA). Just
        multiply by 2 to get two theta.
    """
    # First calculate the wavevector of the incident light.
    planck = physical_constants["Planck constant in eV s"][0]
    speed_of_light = physical_constants["speed of light in vacuum"][0] * 1e10
    # My q_values are angular, so my wavevector needs to be angular too.
    ang_wavevector = 2 * np.pi * energy / (planck * speed_of_light)

    # Do some basic geometry.
    theta_values = np.arccos(1 - np.square(q_values) / (2 * ang_wavevector**2))

    # Convert from radians to degrees.
    return theta_values * 180 / np.pi




def _project_to_1d(volume: np.ndarray,
                   start: np.ndarray,
                   stop: np.ndarray,
                   step: np.ndarray,
                   num_bins: int = 1000,
                   bin_size: float = None,
                   tth=False,
                   only_l=False,
                   energy=None):
    """
    Maps this experiment to a simple intensity vs two-theta representation.
    This virtual 'two-theta' axis is the total angle scattered by the light,
    *not* the projection of the total angle about a particular axis.

    Under the hood, this runs a binned reciprocal space map with an
    auto-generated resolution such that the reciprocal space volume is
    100MB. You really shouldn't need more than 100MB of reciprocal space
    for a simple line chart...

    TODO: one day, this shouldn't run the 3D binned reciprocal space map
    and should instead directly bin to a one-dimensional space. The
    advantage of this approach is that it is much easier to maintain. The
    disadvantage is that theoretical optimal performance is worse, memory
    footprint is higher and some data _could_ be incorrectly binned due to
    double binning. In reality, it's fine: inaccuracies are pedantic and
    performance is pretty damn good.

    TODO: Currently, this method assumes that the beam energy is constant
    throughout the scan. Assumptions aren't exactly in the spirit of this
    module - maybe one day someone can find the time to do this "properly".
    """
    q_x = np.arange(start[0], stop[0], step[0], dtype=np.float32)
    q_y = np.arange(start[1], stop[1], step[1], dtype=np.float32)
    q_z = np.arange(start[2], stop[2], step[2], dtype=np.float32)

    if tth:
        q_x = q_to_theta(q_x, energy)
        q_y = q_to_theta(q_y, energy)
        q_z = q_to_theta(q_z, energy)
    if only_l:
        q_x *= 0
        q_y *= 0

    q_x, q_y, q_z = np.meshgrid(q_x, q_y, q_z, indexing='ij')

    # Now we can work out the |Q| for each voxel.
    q_x_squared = np.square(q_x)
    q_y_squared = np.square(q_y)
    q_z_squared = np.square(q_z)

    q_lengths = np.sqrt(q_x_squared + q_y_squared + q_z_squared)

    # It doesn't make sense to keep the 3D shape because we're about to map
    # to a 1D space anyway.
    rsm = volume.ravel()
    q_lengths = q_lengths.ravel()

    # If we haven't been given a bin_size, we need to calculate it.
    min_q = float(np.nanmin(q_lengths))
    max_q = float(np.nanmax(q_lengths))

    if bin_size is None:
        # Work out the bin_size from the range of |Q| values.
        bin_size = (max_q - min_q) / num_bins

    # Now that we know the bin_size, we can make an array of the bin edges.
    bins = np.arange(min_q, max_q, bin_size)

    # Use this to make output & count arrays of the correct size.
    out = np.zeros_like(bins, dtype=np.float32)
    count = np.zeros_like(out, dtype=np.uint32)

    # Do the binning.
    out = weighted_bin_1d(q_lengths, rsm, out, count,
                          min_q, max_q, bin_size)

    # Normalise by the counts.
    out /= count.astype(np.float32)

    out = np.nan_to_num(out, nan=0)

    # Now return (intensity, Q).
    return out, bins


def intensity_vs_q(output_file_name: str,
                   volume: np.ndarray,
                   start: np.ndarray,
                   stop: np.ndarray,
                   step: np.ndarray,
                   num_bins: int = 1000,
                   bin_size: float = None):
    """
    Calculates intensity as a function of the magnitude of the scattering vector
    from the intensities and their finite differences geometry description.
    """
    intensity, q = _project_to_1d(
        volume, start, stop, step, num_bins, bin_size)

    to_save = np.transpose((q, intensity))
    output_file_name = str(output_file_name) + "_Q.txt"
    print(f"Saving intensity vs q to {output_file_name}")
    np.savetxt(output_file_name, to_save, header="|Q| intensity")
    return q, intensity


def intensity_vs_tth(output_file_name: str,
                     volume: np.ndarray,
                     start: np.ndarray,
                     stop: np.ndarray,
                     step: np.ndarray,
                     energy: float,
                     num_bins: int = 1000,
                     bin_size: float = None):
    """
    Calculates intensity as a function of 2θ, where θ is the total diffracted
    angle (not along any particular direction).
    """
    # This just aliases to self._project_to_1d, which handles the minor
    # difference between a projection to |Q| and 2θ.
    intensity, tth = _project_to_1d(
        volume, start, stop, step, num_bins, bin_size, tth=True, energy=energy)

    to_save = np.transpose((tth, intensity))
    output_file_name = str(output_file_name) + "_tth.txt"
    print(f"Saving intensity vs tth to {output_file_name}")
    np.savetxt(output_file_name, to_save, header="tth intensity")

    return tth, intensity


def intensity_vs_l(output_file_name: str,
                   volume: np.ndarray,
                   start: np.ndarray,
                   stop: np.ndarray,
                   step: np.ndarray,
                   num_bins: int = 1000,
                   bin_size: float = None):
    """
    Calculates intensity as a function of l, ignoring h and k. Only relevant if
    you have used the hkl coordinate system.
    """
    intensity, l = _project_to_1d(
        volume, start, stop, step, num_bins, bin_size, only_l=True)

    to_save = np.transpose((l, intensity))
    output_file_name = str(output_file_name) + "_l.txt"
    print(f"Saving intensity vs l to {output_file_name}")
    np.savetxt(output_file_name, to_save, header="l intensity")



def most_recent_cluster_output():
    """
    Returns the filename of the most recent cluster stdout output.
    """
    # Get all the cluster job files that have been created.
    files = [x for x in os.listdir() if x.startswith("cluster_job.sh.o")]
    numbers = [int(x[16:]) for x in files]

    # Work out which cluster job is the most recent.
    most_recent_job_no = np.max(numbers)
    most_recent_file = ""
    for file in files:
        if str(most_recent_job_no) in file:
            most_recent_file = file

    return most_recent_file


def most_recent_cluster_error():
    """
    Returns the filename of the most recent cluster stderr output.
    """
    # Get all the cluster job files that have been created.
    files = [x for x in os.listdir() if x.startswith("cluster_job.sh.e")]
    numbers = [int(x[16:]) for x in files]

    # Work out which cluster job is the most recent.
    most_recent_job_no = np.max(numbers)
    most_recent_file = ""
    for file in files:
        if str(most_recent_job_no) in file:
            most_recent_file = file

    return most_recent_file
