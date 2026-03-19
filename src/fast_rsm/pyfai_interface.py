"""
Module for the functions used to interface with pyFAI package
"""

import copy
import multiprocessing
import os
import sys
from datetime import datetime
from multiprocessing import Lock, Manager, Process, get_context  # Queue
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from time import time
from types import SimpleNamespace

import numpy as np
import psutil
import pyFAI
import pyFAI.calibrant
import pyFAI.detectors
import yaml

from fast_rsm.angle_pixel_q import calcq, gamdel2rots
from fast_rsm.experiment import Experiment, do_savedats, do_savetiffs
from fast_rsm.image import Image
from fast_rsm.logging_config import get_debug_logger, get_logger, listener_process
from fast_rsm.pyfai_workers import pyfai_init_worker, worker_unpack
from fast_rsm.scan import Scan, chunk

# ----------------------------
# Tuning: set BLAS/OpenMP threads to 1 to avoid oversubscription
# (important when using multiprocessing)
for var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(var, "1")

LOGGER_DEBUG = "fastrsm_debug"
# ====general functions


def find_bad_image_paths(scan: Scan):
    badpaths = []
    for num, end in enumerate(scan.metadata.data_file.raw_image_paths):
        if not end.endswith(".tif"):
            badpaths.append(num)
    return badpaths


def createponi(experiment: Experiment, outpath, offset=0):
    """
    creates a poni file from experiment settings to use in pyFAI functions

    Parameters
    ----------
    outpath : string
        directory to save poni file to.
    image2dshape : array
        shape of image data.
    beam_centre : array, optional
        x,y values for beam centre. The default is 0.
    offset : float, optional
        offset in x for PONI1 value. The default is 0.

    Returns
    -------
    ponioutpath : string
        path to saved poni file.

    """
    image2dshape = experiment.imshape
    beam_centre = experiment.beam_centre
    datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    ponioutpath = rf"{outpath}/fast_rsm_{datetime_str}.poni"
    with open(ponioutpath, "w", encoding="utf-8") as f:
        f.write("# PONI file created by fast_rsm\n#\n")
        f.write("poni_version: 2\n")
        f.write("Detector: Detector\n")
        f.write('Detector_config: {"pixel1":')
        pixel_line = (
            f"{experiment.pixel_size}, "
            f'"pixel2": {experiment.pixel_size}, '
            f'"max_shape": [{image2dshape[0]}, {image2dshape[1]}]'
        )
        f.write(pixel_line)
        f.write("}\n")
        f.write(f"Distance: {experiment.detector_distance}\n")
        if beam_centre == 0:
            poni1 = (image2dshape[0] - offset) * experiment.pixel_size
            poni2 = image2dshape[1] * experiment.pixel_size
        elif (offset == 0) & (experiment.setup != "vertical"):
            poni1 = (beam_centre[0]) * experiment.pixel_size
            poni2 = beam_centre[1] * experiment.pixel_size
        else:  # (offset == 0) & (experiment.setup == 'vertical'):
            poni1 = beam_centre[1] * experiment.pixel_size
            poni2 = (image2dshape[0] - beam_centre[0]) * experiment.pixel_size

        f.write(f"Poni1: {poni1}\n")
        f.write(f"Poni2: {poni2}\n")
        f.write("Rot1: 0.0\n")
        f.write("Rot2: 0.0\n")
        f.write("Rot3: 0.0\n")
        f.write(f"Wavelength: {experiment.incident_wavelength}")
    return ponioutpath


def get_full_indices(scan, scanlength, scalegamma):
    fullrange = np.arange(0, scanlength, scalegamma)
    selectedindices = [n for n in fullrange if n not in scan.skip_images]
    return selectedindices


def get_input_args(experiment, scan, process_config: SimpleNamespace):
    """
    create the input arguments for processing depending on process
    configuration
    """
    cfg = process_config
    fullrange = np.arange(0, cfg.scanlength, cfg.scalegamma)
    selectedindices = [n for n in fullrange if n not in scan.skip_images]
    if cfg.multi:
        inputindices = chunk(selectedindices, cfg.num_threads)
    else:
        inputindices = selectedindices

    input_args = [[experiment, indices, scan, cfg] for indices in inputindices]
    return input_args


def reshape_to_signalshape(arr, signal_shape):
    """
    reshape data to match expected signal shape
    """
    testsize = int(np.prod(signal_shape)) - np.shape(arr)[0]

    fullshape = signal_shape + np.shape(arr)[1:]
    if testsize == 0:
        return np.reshape(arr, fullshape)
    extradata = np.zeros((testsize,) + (np.shape(arr)[1:]))
    outarr = np.concatenate((arr, extradata))
    return np.reshape(outarr, fullshape)


def get_qmapbins(qlimits, experiment):
    """
    obtain number of map bins required for a q step of 0.01
    """

    qstep = round(
        calcq(1.00, experiment.incident_wavelength)
        - calcq(1.01, experiment.incident_wavelength),
        4,
    )
    binshor = abs(round(((qlimits[1] - qlimits[0]) / qstep) * 1.05))
    binsver = abs(round(((qlimits[3] - qlimits[2]) / qstep) * 1.05))
    return (binshor, binsver)


def get_corner_thetas(process_config: SimpleNamespace):
    """
    calculate theta angles given inplane and out-of-plane angles to the detector
    corners.
    """
    cfg = process_config
    corner_indexes = [[0, 2], [0, 3], [1, 2], [1, 3]]
    cfg.fullranges90 = [
        (val, 0) if val <= 90 else (val - 90, 90) for val in np.abs(cfg.fullranges)
    ]

    corner_items = [
        [cfg.fullranges90[ind][0] for ind in pair] for pair in corner_indexes
    ]
    corner_values = np.radians(corner_items)
    corner_diagonal_angles = np.degrees(
        np.arctan(
            [np.sqrt(np.tan(cv[0]) ** 2 + np.tan(cv[1]) ** 2) for cv in corner_values]
        )
    )
    absranges = [
        np.abs(dval) + cfg.fullranges90[i][1]
        for i, dval in enumerate(corner_diagonal_angles)
    ]
    radmax = np.max(absranges)
    return absranges, radmax


# def pyfai_init_worker(smmlock, shm_intensities_name, shm_counts_name, shmshape):
#     """
#     intialiser for pyfai mappings
#     """
#     global lock
#     global SHM_INTENSITY
#     global INTENSITY_ARRAY
#     global SHM_COUNT
#     global COUNT_ARRAY
#     SHM_INTENSITY = SharedMemory(name=shm_intensities_name)
#     SHM_COUNT = SharedMemory(name=shm_counts_name)
#     INTENSITY_ARRAY = np.ndarray(
#         shape=shmshape, dtype=np.float32, buffer=SHM_INTENSITY.buf
#     )
#     COUNT_ARRAY = np.ndarray(shape=shmshape, dtype=np.float32, buffer=SHM_COUNT.buf)
#     lock = smmlock


def get_inc_angles_out(incident_angle, setup, index):
    if np.size(incident_angle) > 1:
        inc_angle = -np.radians(incident_angle[index])
    elif isinstance(incident_angle, np.float64):
        inc_angle = -np.radians(incident_angle)
    else:
        inc_angle = -np.radians(float(incident_angle))

    if setup == "DCD":
        inc_angle_out = 0  # debug setting incident angle to 0
    else:
        inc_angle_out = inc_angle

    return inc_angle, inc_angle_out


def get_gam_del_vals(gammadata, two_theta_start, deltadata, index):
    gamval = 0
    delval = 0
    if np.size(gammadata) > 1:
        gamval = -np.array(two_theta_start).ravel()[index]
    elif np.size(gammadata) == 1:
        gamval = -np.array(two_theta_start).ravel()
    if np.size(deltadata) > 1:
        delval = np.array(deltadata).ravel()[index]
    elif np.size(deltadata) == 1:
        delval = np.array(deltadata).ravel()
    return [gamval, delval]


def pyfai_setup_limits(experiment: Experiment, scanlist, limitfunction, process_config):
    """
    calculate setup values needed for pyfai calculations
    """
    # pylint: disable=attribute-defined-outside-init
    cfg = process_config
    slitratios = cfg.slitratios
    if isinstance(scanlist, Scan):
        scanlistnew = [scanlist]
    else:
        scanlistnew = scanlist

    limhor = None
    limver = None
    for scan in scanlistnew:
        experiment.load_curve_values(scan)

        if slitratios is not None:
            slitvertratio, slithorratio = slitratios
        else:
            slitvertratio = slithorratio = None

        scanlimhor = limitfunction("hor", slithorratio=slithorratio)
        scanlimver = limitfunction("vert", slitvertratio=slitvertratio)

        scanlimits = [scanlimhor[0], scanlimhor[1], scanlimver[0], scanlimver[1]]
        if limhor is None:
            limhor = scanlimits[0:2]
            limver = scanlimits[2:]
        else:
            limhor = combine_ranges(limhor, scanlimits[0:2])
            limver = combine_ranges(limver, scanlimits[2:])

    outlimits = [limhor[0], limhor[1], limver[0], limver[1]]
    if experiment.setup == "vertical":
        experiment.beam_centre = [experiment.beam_centre[1], experiment.beam_centre[0]]
        experiment.beam_centre[1] = experiment.imshape[0] - experiment.beam_centre[1]

    datacheck = "data" in list(scan.metadata.data_file.nx_detector)
    localpathcheck = "local_image_paths" in scan.metadata.data_file.__dict__.keys()
    intcheck = isinstance(scan.metadata.data_file.scan_length, int)
    if datacheck & intcheck:
        scanlength = np.shape(scan.metadata.data_file.nx_detector.data[:, 1, :])[0]
        scanlength = min(scanlength, scan.metadata.data_file.scan_length)
    elif datacheck:
        scanlength = np.shape(scan.metadata.data_file.nx_detector.data[:, 1, :])[0]
    elif localpathcheck:
        scanlength = len(scan.metadata.data_file.local_image_paths)
    else:
        scanlength = scan.metadata.data_file.scan_length

    # check for scans finished early
    if not scan.metadata.data_file.has_hdf5_data:
        badimagecheck = find_bad_image_paths(scan)
        if len(badimagecheck) > 0:
            scanlength -= len(badimagecheck)
    return outlimits, scanlength, scanlistnew


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_pyfai_calib_image(ai):
    LaB6 = pyFAI.calibrant.get_calibrant("LaB6")

    det = pyFAI.detectors.Maxipix()  # choose detector with the same pixel size 5.5e-5
    newshape = (515, 2069)
    det.shape = newshape
    det.max_shape = newshape
    ai.detector = det
    return LaB6.fake_calibration_image(ai)


def load_flat_test_image():
    dummy_ai = pyFAI.load(
        "/dls/science/users/rpy65944/output/fast_rsm_2026-01-08_13h56m28s.poni"
    )
    dummy_img = load_pyfai_calib_image(dummy_ai)
    flat_image = np.ones(np.shape(dummy_img))
    flat_image[dummy_ai.mask == 1] = 0
    return flat_image, dummy_ai


def combine_ranges(range1, range2):
    """
    combines two ranges to give the widest possible range
    """
    return (min(range1[0], range2[0]), max(range1[1], range2[1]))


def check_full_1d_radial_range(
    experiment: Experiment, process_config, absranges, radmax
):
    cfg = process_config
    centre_check = {1: True, 0: False, 2: False}
    hor_centre = centre_check[np.sum([(val > 0) for val in cfg.fullranges[0:2]])]
    ver_centre = centre_check[np.sum([(val > 0) for val in cfg.fullranges[2:]])]
    if hor_centre and ver_centre:
        radialrange = (0, np.max(absranges))
    # elif hor_centre:
    #     radialrange = (min(abs(np.array(cfg.fullranges[2:]))), radmax)
    # elif ver_centre:
    #     radialrange = (min(abs(np.array(cfg.fullranges[0:2]))), radmax)
    else:
        radialrange = (min(absranges), radmax)
    # if str(cfg.unit_qip_name).startswith("q"):
    #     return [calcq(val,experiment.incident_wavelength) for val in radialrange]
    return radialrange


def get_d5i_values(scan):
    if hasattr(scan.metadata.data_file.nx_entry, "d5i"):
        d5i_full = np.array(scan.metadata.data_file.nx_entry.d5i.data)
    else:
        d5i_full = np.ones(scan.metadata.data_file.scan_length)
    return d5i_full


def start_listener():
    manager = Manager()
    log_queue = manager.Queue()
    listener = Process(
        target=listener_process, args=(log_queue, get_logger, LOGGER_DEBUG)
    )
    listener.start()
    return listener, log_queue


def add_buffer_to_limits(limits):
    buffers = [np.max([0.05, 0.05 * np.abs(lim)]) for lim in limits]
    limit1 = limits[0] - buffers[0]
    limit2 = limits[1] + buffers[1]
    limit3 = limits[2] - buffers[2]
    limit4 = limits[3] + buffers[3]

    return [limit1, limit2, limit3, limit4]


def setup_debug_logger():
    sys.stdout.reconfigure(line_buffering=True)
    logger = get_logger(LOGGER_DEBUG)
    listener, log_queue = start_listener()
    return logger, listener, log_queue


def setup_job(process_config, experiment, scan, limit_key):
    cfg = copy.copy(process_config)

    limit_functions = {"ang": experiment.calcanglim, "q": experiment.calcqlim}
    cfg.fullranges, cfg.scanlength, cfg.scanlistnew = pyfai_setup_limits(
        experiment, scan, limit_functions[limit_key], cfg
    )

    absranges, radmax = get_corner_thetas(cfg)
    cfg.fullranges = add_buffer_to_limits(cfg.fullranges)
    cfg.scalegamma = 1
    cfg.sample_orientation = 1
    cfg.scan_ind = 0
    if cfg.radialrange is None:
        cfg.radialrange = check_full_1d_radial_range(experiment, cfg, absranges, radmax)

    if cfg.ivqbins is None:
        cfg.ivqbins = int(
            np.ceil((cfg.radialrange[1] - cfg.radialrange[0]) / cfg.radialstepval)
        )
    return cfg


def setup_pool_info(cfg, experiment, scan):

    d5i_full = get_d5i_values(scan)
    imageindices = get_full_indices(scan, cfg.scanlength, cfg.scalegamma)

    incident_angle, setup = experiment.incident_angle, cfg.setup
    all_inc_angles = [
        get_inc_angles_out(incident_angle, setup, ind) for ind in imageindices
    ]

    gammadata, two_theta_start, deltadata = (
        experiment.gammadata,
        experiment.two_theta_start,
        experiment.deltadata,
    )
    gamdelvals = [
        get_gam_del_vals(gammadata, two_theta_start, deltadata, ind)
        for ind in imageindices
    ]

    return d5i_full, all_inc_angles, gamdelvals


def setup_initial_ai(pyfaiponi, fiber_integrator=False):
    if fiber_integrator:
        return pyFAI.load(pyfaiponi, type_="pyFAI.integrator.fiber.FiberIntegrator")

    return pyFAI.load(pyfaiponi)


def get_batch_details(multi, imageindices, batchsize):
    if multi:
        batches = list(chunked(imageindices, batchsize))
    else:
        batches = imageindices

    num_batches = len(batches)
    completed = 0
    return batches, num_batches, completed


def check_data_shape(inlist, scan):
    signal_shape = np.shape(scan.metadata.data_file.default_signal)
    if len(signal_shape) > 1:
        outlist = [reshape_to_signalshape(arr, signal_shape) for arr in inlist]
        return outlist
    return inlist


# ===================================
# ====data saving functions


def save_1d_integration_static(cfg, hf, outlist, scan=None):
    """
    save 1d Intensity Vs Q profile to hdf5 file
    """

    dset = hf.create_group("integrations")
    dset.create_dataset("Intensity", data=[outlist[0]])
    dset.create_dataset("Q_angstrom^-1", data=outlist[1])
    dset.create_dataset("2thetas", data=outlist[2])

    if (scan is not None) & ("scanfields" not in hf.keys()):
        save_scan_field_values(hf, scan)
    if cfg.savedats is True:
        do_savedats(hf, outlist[0], outlist[1], outlist[2])
    save_config_variables(hf, cfg)
    hf.close()


def save_1d_integration(hf, cfg, int_final, counts_final, tth_vals_final, q_final):

    int_array = np.divide(
        int_final[0],
        counts_final[0],
        out=np.copy(int_final),
        where=counts_final[0].astype(float) > 0.0,
    )
    outlist = [int_array, q_final, tth_vals_final]
    save_1d_integration_static(cfg, hf, outlist)


def save_qperp_qpara(experiment, hf, qperp_qpara_map, scan=0):
    """
    save a qpara vs qperp map to hdf5 file

    """
    dset = hf.create_group("qperp_qpara")
    dset.create_dataset("images", data=qperp_qpara_map[0])
    dset.create_dataset("qpararanges", data=qperp_qpara_map[1])
    dset.create_dataset("qperpranges", data=qperp_qpara_map[2])
    if "scanfields" not in hf.keys():
        save_scan_field_values(hf, scan)

    if experiment.savetiffs is True:
        do_savetiffs(hf, qperp_qpara_map[0], qperp_qpara_map[1], qperp_qpara_map[2])


def save_config_variables(hf, process_config):
    """
    save all variables in the configuration file to the output hdf5 file
    """
    cfg = process_config
    config_group = hf.create_group("i07configuration")
    outdict = vars(cfg)
    with open(cfg.default_config_path, "r", encoding="utf-8") as f:
        default_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    # add in extra to defaults that arent set by user, so that parsing
    # defaults finds it
    default_config_dict["ubinfo"] = 0
    default_config_dict["pythonlocation"] = 0
    default_config_dict["joblines"] = 0
    for key in default_config_dict:
        if key == "ubinfo":
            for i, coll in enumerate(outdict["ubinfo"]):
                ubgroup = config_group.create_group(f"ubinfo_{i + 1}")
                ubgroup.create_dataset(
                    f"lattice_{i + 1}", data=coll["diffcalc_lattice"]
                )
                ubgroup.create_dataset(f"u_{i + 1}", data=coll["diffcalc_u"])
                ubgroup.create_dataset(f"ub_{i + 1}", data=coll["diffcalc_ub"])
            continue
        val = outdict[key]
        if val is None:
            val = "None"
        config_group.create_dataset(f"{key}", data=val)


def save_scan_field_values(hf, scan):
    """
    saves scanfields recorded in nexus file to hdf5 output
    """

    try:
        rank = scan.metadata.data_file.diamond_scan.scan_rank.nxdata
        fields = scan.metadata.data_file.diamond_scan.scan_fields
        scanned = [x.decode("utf-8").split(".")[0] for x in fields[:rank].nxdata]
        scannedvalues = [
            np.unique(scan.metadata.data_file.nx_instrument[field].value)
            for field in scanned
        ]
        scannedvaluesout = [
            scannedvals[~np.isnan(scannedvals)] for scannedvals in scannedvalues
        ]
    except BaseException:
        scanned, scannedvaluesout = None, None

    dset = hf.create_group("scanfields")
    if scan != 0:
        if scanned is not None:
            for i, field in enumerate(scanned):
                dset.create_dataset(f"dim{i}_{field}", data=scannedvaluesout[i])


def save_hf_map_static(hf, cfg, start_time, mapname, mapdata, mapaxisinfo, scan=None):
    end_time = time()
    times = [start_time, end_time]
    dset = hf.create_group(f"{mapname}")
    dset.create_dataset(f"{mapname}_map", data=mapdata)
    dset.create_dataset("map_para", data=mapaxisinfo[1])
    dset.create_dataset("map_para_unit", data=mapaxisinfo[3])
    dset.create_dataset("map_perp", data=mapaxisinfo[0])
    dset.create_dataset("map_perp_unit", data=mapaxisinfo[2])
    dset.create_dataset("map_perp_indices", data=[0, 1, 2])
    dset.create_dataset("map_para_indices", data=[0, 1, 3])

    if (scan is not None) & ("scanfields" not in hf.keys()):
        save_scan_field_values(hf, scan)
    if cfg.savetiffs:
        do_savetiffs(hf, mapdata, mapaxisinfo[1], mapaxisinfo[0])
    save_config_variables(hf, cfg)
    hf.close()
    minutes = (times[1] - times[0]) / 60
    print(f"total calculation took {minutes}  minutes")


def save_hf_map(
    experiment: Experiment,
    hf,
    mapname,
    sum_array,
    counts_array,
    mapaxisinfo,
    start_time,
    process_config,
):
    cfg = process_config
    norm_array = np.divide(
        sum_array, counts_array, out=np.copy(sum_array), where=counts_array != 0.0
    )
    save_hf_map_static(hf, cfg, start_time, mapname, norm_array, mapaxisinfo)


def save_masks(hf, mask_list):
    dset = hf.create_group("masks")
    dset.create_dataset("total_mask", data=mask_list[0])
    dset.create_dataset("image_mask", data=mask_list[1])
    dset.create_dataset("sector_mask", data=mask_list[2])


def start_smm(smm, memshape):
    """
    start up the shared memory manager and associated data arrays
    """
    shm_intensities = smm.SharedMemory(size=np.zeros(memshape, dtype=np.float32).nbytes)
    shm_counts = smm.SharedMemory(size=np.zeros(memshape, dtype=np.float32).nbytes)
    arrays_arr = np.ndarray(memshape, dtype=np.float32, buffer=shm_intensities.buf)
    counts_arr = np.ndarray(memshape, dtype=np.float32, buffer=shm_counts.buf)
    arrays_arr.fill(0)
    counts_arr.fill(0)
    lock = Lock()
    return shm_intensities, shm_counts, arrays_arr, counts_arr, lock


def setup_copy_ai(aistart, slitratios):
    out_ai = copy.deepcopy(aistart)
    if slitratios[0] is not None:
        out_ai.pixel1 *= slitratios[0]
        out_ai.poni1 *= slitratios[0]

    if slitratios[1] is not None:
        out_ai.pixel2 *= slitratios[1]
        out_ai.poni2 *= slitratios[1]
    return out_ai


def calc_rots_from_gamdel(
    gamdelval,
    inc_angle,
    alphacritical,
    setup,
):
    gamval, delval = gamdelval
    if (-np.degrees(inc_angle) > alphacritical) & (setup == "DCD"):
        # if above critical angle, account for direct beam adding to delta
        return gamdel2rots(gamval, delval + np.degrees(-inc_angle))

    return gamdel2rots(gamval, delval)


def set_ai_rots(rots, current_ai, setup):
    """
    get components need for mapping with pyFAI
    """
    out_ai = copy.deepcopy(current_ai)
    out_ai.rot1, out_ai.rot2, out_ai.rot3 = rots

    if setup == "vertical":
        out_ai.rot1 = rots[1]
        out_ai.rot2 = -rots[0]
    return out_ai


def get_pyfai_image_data(setup: str, metadata, idx):

    # outimage = scan.load_image(i)
    outimage = Image(metadata, idx, load_image=True)
    mask = np.isnan(outimage.data).astype(bool)
    if setup == "vertical":
        return np.rot90(outimage.data, -1), np.rot90(mask, -1)

    return np.array(outimage.data), mask


# ===================================
# ====moving detector processing


def pyfai_moving_ivsq_smm_new(experiment: Experiment, hf, scanlist, process_config):
    """
    calculate 1d Intensity Vs Q profile for a moving detector scan
    """

    cfg = setup_job(process_config, experiment, scanlist, "ang")
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger()
    else:
        log_queue = None
    # num_threads = int(cfg.num_threads)  # e.g., 40
    intensity_results_per_scan, count_results_per_scan, tth_results_per_scan = [
        [],
        [],
        [],
    ]

    cfg.unit_qip_name = "2th_deg"  # "qip_A^-1"# "qip_A^-1""2th_deg"  #
    cfg.unit_qoop_name = "2th_deg"  # "qoop_A^-1"
    cfg.multi = True
    cfg.shapeqi = (1, np.abs(cfg.ivqbins))

    cfg.batchsize = 15
    ctx = get_context("fork")
    with ctx.Pool(processes=cfg.num_threads) as pool:
        for scanind, scan in enumerate(cfg.scanlistnew):
            experiment.load_curve_values(scan)
            cfg, batches, num_batches, completed = setup_pool_info(
                cfg, experiment, scan
            )
            args_iter = (
                ("move_ivq", experiment, batch, scan, cfg, log_queue, logn)
                for logn, batch in enumerate(batches)
            )
            accumulator_intensity = np.zeros((1, cfg.ivqbins), dtype=np.float32)
            accumulator_count = np.zeros((1, cfg.ivqbins), dtype=np.float32)
            accumulator_tth = np.zeros((1, cfg.ivqbins), dtype=np.float32)
            accumulator_mask = []

            for partial in pool.imap_unordered(worker_unpack, args_iter, chunksize=10):
                if (completed == 0) and (scanind == 0):
                    accumulator_mask = partial[3]
                accumulator_intensity += partial[0]
                accumulator_count += partial[1]
                accumulator_tth += partial[2]

                completed += 1
                if completed % 10 == 0 or completed == num_batches:
                    print(f"  completed {completed}/{num_batches} batches", flush=True)

            intensity_results_per_scan.append(accumulator_intensity)
            count_results_per_scan.append(accumulator_count)
            tth_results_per_scan.append(accumulator_tth / completed)
            print(f"[scan {scanind + 1}] finished.")
    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener

    int_final = np.sum(intensity_results_per_scan, axis=0)
    counts_final = np.sum(count_results_per_scan, axis=0)
    tth_vals_final = tth_results_per_scan[0]
    q_final = [calcq(val, experiment.incident_wavelength) for val in tth_vals_final]
    save_1d_integration(hf, cfg, int_final, counts_final, tth_vals_final, q_final)


def pyfai_moving_qmap_smm_new(experiment: Experiment, hf, scanlist, process_config):
    """
    calculate q_para vs q_perp map for a moving detector scan
    """

    cfg = setup_job(process_config, experiment, scanlist, "q")
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger()
    else:
        log_queue = None

    intensity_results_per_scan = []
    count_results_per_scan = []
    mapaxisinfo = []
    t0 = time()
    cfg.multi = True
    cfg.unit_qip_name = "qip_A^-1"  # "2th_deg"  # "qtot_A^-1"# "qip_A^-1"
    cfg.unit_qoop_name = "qoop_A^-1"
    # "2th_deg"
    cfg.aistart = setup_initial_ai(cfg.pyfaiponi)
    cfg.batchsize = 30
    print(f"starting multiprocessing with {cfg.num_threads} cpus")

    ram_gb = psutil.virtual_memory().total / 1024**3
    cores = multiprocessing.cpu_count()

    print("Memory per CPU (GB):", ram_gb / cores)

    ctx = get_context("fork")
    with SharedMemoryManager() as smm:
        cfg.shapeqpqp = (cfg.qmapbins[1], cfg.qmapbins[0])
        shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
            smm, cfg.shapeqpqp
        )
        start_time = time()
        for scanind, scan in enumerate(
            cfg.scanlistnew
        ):  # chunksize=1 makes sense here: each task is already “large” (25 images)
            experiment.load_curve_values(scan)
            imageindices = get_full_indices(scan, cfg.scanlength, cfg.scalegamma)
            batches, num_batches, completed = get_batch_details(
                False, imageindices, cfg.batchsize
            )
            # Prepare batches
            d5i_full, all_inc_angles, gamdelvals = setup_pool_info(
                cfg, experiment, scan
            )
            current_ai = setup_copy_ai(cfg.aistart, cfg.slitratios)
            # Only lightweight job descriptors sent to workers
            pool_function = worker_unpack("move_qmap")

            # args_iter = [
            #     [batch, scan.metadata, unique_scan_vals, cfg, log_queue, logn]
            #     for logn, batch in enumerate(batches)
            # ]
            alphacritical = cfg.alphacritical
            setup = cfg.setup
            newrots = [
                calc_rots_from_gamdel(
                    gamdelvals[ind_n], all_inc_angles[ind_n][0], alphacritical, setup
                )
                for ind_n in np.arange(len(batches))
            ]
            # img_list = [
            #     get_pyfai_image_data(setup, scan.metadata, ind) for ind in batches
            # ]
            # ai_list = [set_ai_rots(newrot, current_ai, setup) for newrot in newrots]
            args_iter = [
                [
                    ind,
                    d5i_full[ind_n],
                    scan.metadata,
                    current_ai,
                    newrots[ind_n],
                    cfg,
                    log_queue,
                    ind_n,
                ]
                for ind_n, ind in enumerate(batches)
            ]
            chunksize = max(1, num_batches // (cfg.num_threads * 4))
            with ctx.Pool(
                processes=cfg.num_threads,
                initializer=pyfai_init_worker,
                initargs=(
                    lock,
                    shm_intensities.name,
                    shm_counts.name,
                    cfg.shapeqpqp,
                ),
            ) as pool:
                mapaxisinfolist = pool.starmap(pool_function, args_iter)
        qmap_final = np.zeros(cfg.shapeqpqp, dtype=np.float32)
        counts_final = np.zeros(cfg.shapeqpqp, dtype=np.float32)
        shmI = SharedMemory(name=shm_intensities.name)
        shmC = SharedMemory(name=shm_counts.name)

        intensity_view = np.ndarray(cfg.shapeqpqp, dtype=np.float32, buffer=shmI.buf)
        count_view = np.ndarray(cfg.shapeqpqp, dtype=np.float32, buffer=shmC.buf)

        qmap_final += intensity_view
        counts_final += count_view
        # intensity_results_per_scan.append(accumulator_intensity)
        # count_results_per_scan.append(accumulator_count)

    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener

    # qmap_final = np.sum(intensity_results_per_scan, axis=0)
    # counts_final = np.sum(count_results_per_scan, axis=0)
    save_hf_map(
        experiment,
        hf,
        "qpara_qperp",
        qmap_final,
        counts_final,
        mapaxisinfolist[0],
        t0,
        cfg,
    )


def pyfai_moving_exitangles_smm_new(
    experiment: Experiment, hf, scanlist, process_config
):
    """
    calculate q_para vs q_perp map for a moving detector scan
    """
    logger = get_debug_logger()
    cfg = setup_job(process_config, experiment, scanlist, "ang")
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger()
    else:
        log_queue = None
    intensity_results_per_scan = []
    count_results_per_scan = []
    mapaxisinfo = []
    t0 = time()
    cfg.multi = True
    cfg.unit_qip_name = "exit_angle_horz_deg"  # "2th_deg"  # "qtot_A^-1"# "qip_A^-1"
    cfg.unit_qoop_name = "exit_angle_vert_deg"  # "2th_deg"

    cfg.batchsize = 15
    ctx = get_context("fork")
    with ctx.Pool(processes=cfg.num_threads) as pool:
        for scanind, scan in enumerate(
            cfg.scanlistnew
        ):  # chunksize=1 makes sense here: each task is already “large” (25 images)
            experiment.load_curve_values(scan)
            cfg, batches, num_batches, completed = setup_pool_info(
                cfg, experiment, scan
            )
            args_iter = (
                ("move_exit", experiment, batch, scan, cfg, log_queue, logn)
                for logn, batch in enumerate(batches)
            )
            accumulator_intensity = np.zeros(
                (1, cfg.qmapbins[1], cfg.qmapbins[0]), dtype=np.float32
            )
            accumulator_count = np.zeros(
                (1, cfg.qmapbins[1], cfg.qmapbins[0]), dtype=np.float32
            )

            accumulator_mask = []

            for partial in pool.imap_unordered(worker_unpack, args_iter, chunksize=10):
                if (completed == 0) and (scanind == 0):
                    accumulator_mask = partial[3]
                    mapaxisinfo.append(partial[2])
                accumulator_intensity += partial[0]
                accumulator_count += partial[1]
                completed += 1
                if completed % 10 == 0 or completed == num_batches:
                    print(f"  completed {completed}/{num_batches} batches", flush=True)
            intensity_results_per_scan.append(accumulator_intensity)
            count_results_per_scan.append(accumulator_count)

    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener

    qmap_final = np.sum(intensity_results_per_scan, axis=0)
    counts_final = np.sum(count_results_per_scan, axis=0)
    save_hf_map(
        experiment, hf, "exit_angles", qmap_final, counts_final, mapaxisinfo[0], t0, cfg
    )


# ===================================
# ====static detector processing


def pyfai_static_ivsq_new(
    experiment: Experiment, hf, scan, process_config: SimpleNamespace
):
    """
    calculate Intensity Vs Q 1d profile from static detector scan
    """
    cfg = setup_job(process_config, experiment, scan, "ang")
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger()
    cfg.unit_qip_name = "2th_deg"  # "qip_A^-1"# "qip_A^-1""2th_deg"  #
    cfg.unit_qoop_name = "2th_deg"  # "qoop_A^-1"
    print(f"starting process pool with num_threads={cfg.num_threads}")
    cfg.multi = False
    all_ints, all_two_ths, all_qs, scan_masks = [[], [], [], []]
    cfg.shapeqi = (1, np.abs(cfg.ivqbins))
    ctx = get_context("fork")
    process_count = cfg.num_threads

    cfg, batches, num_batches, completed = setup_pool_info(cfg, experiment, scan)
    args_iter = (
        (
            "static_ivq",
            experiment,
            batch,
            scan,
            cfg,
        )
        for batch in batches
    )

    with ctx.Pool(processes=process_count) as pool:
        for partial in pool.imap(worker_unpack, args_iter, chunksize=1):
            if completed == 0:
                scan_masks.append(partial[2])
            all_ints.append(partial[0])
            all_two_ths.append(partial[1])
            all_qs.append(
                [calcq(val, experiment.incident_wavelength) for val in partial[1]]
            )
            completed += 1
            if completed % 10 == 0 or completed == num_batches:
                print(f"  completed {completed}/{num_batches} batches", flush=True)

    inlist = [all_ints, all_qs, all_two_ths]
    outlist = check_data_shape(inlist, scan)
    save_masks(hf, scan_masks[0])

    save_1d_integration_static(cfg, hf, outlist, scan)
    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener


def pyfai_static_exitangles_new(
    experiment: Experiment, hf, scan, process_config: SimpleNamespace
):

    cfg = setup_job(process_config, experiment, scan, "ang")
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger()
    t0 = time()
    cfg.unit_qip_name = "exit_angle_horz_deg"  # "qip_A^-1"# "qip_A^-1""2th_deg"  #
    cfg.unit_qoop_name = "exit_angle_vert_deg"  # "qoop_A^-1"
    print(f"starting process pool with num_threads={cfg.num_threads}")
    cfg.multi = False
    all_maps, all_xlabels, all_ylabels, scan_mask, all_mapaxisinfo = [
        [],
        [],
        [],
        [],
        [],
    ]

    ctx = get_context("fork")
    process_count = cfg.num_threads

    cfg, batches, num_batches, completed = setup_pool_info(cfg, experiment, scan)
    args_iter = (("static_exit", experiment, batch, scan, cfg) for batch in batches)

    with ctx.Pool(processes=process_count) as pool:
        for partial in pool.imap(worker_unpack, args_iter, chunksize=1):
            if completed == 0:
                scan_mask.append(partial[4])
            all_maps.append(partial[0])
            all_xlabels.append(partial[1])
            all_ylabels.append(partial[2])
            all_mapaxisinfo.append(partial[3])
            completed += 1
            if completed % 10 == 0 or completed == num_batches:
                print(f"  completed {completed}/{num_batches} batches", flush=True)
    print("finished process pool")
    inlist = [all_maps, all_xlabels, all_ylabels]
    outlist = check_data_shape(inlist, scan)
    save_hf_map_static(hf, cfg, t0, "exit_angles", outlist[0], all_mapaxisinfo[0], scan)
    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener


def pyfai_static_qmap_new(
    experiment: Experiment, hf, scan, process_config: SimpleNamespace
):

    cfg = setup_job(process_config, experiment, scan, "q")
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger()
    t0 = time()
    cfg.unit_qip_name = "qip_A^-1"  # "qip_A^-1"# "qip_A^-1""2th_deg"  #
    cfg.unit_qoop_name = "qoop_A^-1"  # "qoop_A^-1"
    print(f"starting process pool with num_threads={cfg.num_threads}")
    cfg.multi = False
    all_maps, all_xlabels, all_ylabels, scan_mask, all_mapaxisinfo = [
        [],
        [],
        [],
        [],
        [],
    ]

    ctx = get_context("fork")
    process_count = cfg.num_threads

    cfg, batches, num_batches, completed = setup_pool_info(
        cfg, experiment, scan, fiber_integrator=True
    )
    args_iter = (("static_qmap", experiment, batch, scan, cfg) for batch in batches)
    with ctx.Pool(processes=process_count) as pool:
        for partial in pool.imap(worker_unpack, args_iter, chunksize=1):
            if completed == 0:
                scan_mask.append(partial[4])
            all_maps.append(partial[0])
            all_xlabels.append(partial[1])
            all_ylabels.append(partial[2])
            all_mapaxisinfo.append(partial[3])
            completed += 1
            if completed % 10 == 0 or completed == num_batches:
                print(f"  completed {completed}/{num_batches} batches", flush=True)
    print("finished process pool")
    inlist = [all_maps, all_xlabels, all_ylabels]
    outlist = check_data_shape(inlist, scan)
    save_hf_map_static(hf, cfg, t0, "qpara_qperp", outlist[0], all_mapaxisinfo[0], scan)
    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener


# OLD shared memory


# def init_pyfai_process_pool(
#         locks: List[Lock],
#         num_threads: int,
#         metadata: RSMMetadata,
#         shapeqi: tuple,
#     shapecake: tuple,
#         shapeqpqpmap: tuple,
#         output_file_name: str = None
# ) -> None:
#     """
#     Initializes a processing pool to have a global shared lock.

#     Args:
#         locks:
#             A list of the locks that will be shared between spawned processes.
#         num_threads:
#             The total number of processes that are being spawned in the pool.
#         shape:
#             Passed if you want to make PYFAI_QI and CAKE arrays global.
#     """
#     # pylint: disable=global-variable-undefined.

#     # Make a global lock for the shared memory block used in parallel code.
#     global LOCKS

#     # Some metadata that a worker thread should always have access to.
#     global NUM_THREADS
#     global METADATA

#     # Not always necessary and may be set to None.
#     global OUTPUT_FILE_NAME

#     # These are numpy arrays whose buffer corresponds to the shared memory
#     # buffer. It's more convenient to access these later than to directly work
#     # with the shared memory buffer.
#     global PYFAI_QI
#     global CAKE
#     global QPQPMAP

#     # We want to keep track of what we've called our shared memory arrays.
#     global SHARED_PYFAI_QI_NAME
#     global SHARED_CAKE_NAME
#     global SHARED_QPQPMAP_NAME
#     # Why do we need to make the shared memory blocks global, if we're giving
#     # global access to them via the numpy 'PYFAI_QI' and 'CAKE' arrays? The answer
#     # is that we need the shared memory arrays to remain in scope, or they'll be
#     # freed.
#     global SHARED_PYFAI_QI
#     global SHARED_CAKE
#     global SHARED_QPQPMAP

#     LOCKS = locks
#     NUM_THREADS = num_threads
#     METADATA = metadata

#     OUTPUT_FILE_NAME = output_file_name

#     # Work out how many bytes we're going to need by making a dummy array.
#     arrqi = np.ndarray(shape=shapeqi, dtype=np.float32)
#     arrcake = np.ndarray(shape=shapecake, dtype=np.float32)
#     arrqpqpmap = np.ndarray(shape=shapeqpqpmap, dtype=np.float32)

#     # Construct the shared memory buffers.
#     SHARED_PYFAI_QI_NAME = f'pyfai_qi_{current_process().name}'
#     SHARED_CAKE_NAME = f'cake_{current_process().name}'
#     SHARED_QPQPMAP_NAME = f'qpqpmap_{current_process().name}'

#     check_shared_memory(SHARED_PYFAI_QI_NAME)
#     check_shared_memory(SHARED_CAKE_NAME)
#     check_shared_memory(SHARED_QPQPMAP_NAME)

#     SHARED_PYFAI_QI = SharedMemory(
#         name=SHARED_PYFAI_QI_NAME, create=True, size=arrqi.nbytes)
#     SHARED_CAKE = SharedMemory(
#         name=SHARED_CAKE_NAME, create=True, size=arrcake.nbytes)
#     SHARED_QPQPMAP = SharedMemory(
#         name=SHARED_QPQPMAP_NAME, create=True, size=arrqpqpmap.nbytes)

#     # Construct the global references to the shared memory arrays.
#     PYFAI_QI = np.ndarray(shapeqi, dtype=np.float32,
#                           buffer=SHARED_PYFAI_QI.buf)
#     CAKE = np.ndarray(shapecake, dtype=np.float32, buffer=SHARED_CAKE.buf)
#     QPQPMAP = np.ndarray(shapeqpqpmap, dtype=np.float32,
#                          buffer=SHARED_QPQPMAP.buf)

#     # Initialize the shared memory arrays.
#     PYFAI_QI.fill(0)
#     CAKE.fill(0)
#     QPQPMAP.fill(0)

#     print(f"Finished initializing worker {current_process().name}.")

# def start_smm(smm, memshape):
#     """
#     start up the shared memory manager and associated data arrays
#     """
#     shm_intensities = smm.SharedMemory(
#         size=np.zeros(memshape, dtype=np.float32).nbytes)
#     shm_counts = smm.SharedMemory(
#         size=np.zeros(
#             memshape,
#             dtype=np.float32).nbytes)
#     arrays_arr = np.ndarray(
#         memshape,
#         dtype=np.float32,
#         buffer=shm_intensities.buf)
#     counts_arr = np.ndarray(
#         memshape,
#         dtype=np.float32,
#         buffer=shm_counts.buf)
#     arrays_arr.fill(0)
#     counts_arr.fill(0)
#     l = Lock()
#     return shm_intensities, shm_counts, arrays_arr, counts_arr, l
