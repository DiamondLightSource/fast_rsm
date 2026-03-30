"""
Module for the functions used to interface with pyFAI package
"""

import copy
import multiprocessing
import os
from dataclasses import dataclass
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
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.integrator.fiber import FiberIntegrator

from fast_rsm.angle_pixel_q import calcq, gamdel2rots
from fast_rsm.experiment import Experiment, do_savedats, do_savetiffs
from fast_rsm.image import Image
from fast_rsm.logging_config import get_logger, listener_process
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
    max_shape_vals = f"{image2dshape[0]}, {image2dshape[1]}"
    if experiment.setup == "vertical":
        max_shape_vals = f"{image2dshape[1]}, {image2dshape[0]}"
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
            f'"max_shape": [{max_shape_vals}]'
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
    cornervalues = [
        np.abs(dval) + cfg.fullranges90[i][1]
        for i, dval in enumerate(corner_diagonal_angles)
    ]
    radmax = np.max(cornervalues)
    return cornervalues, radmax


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


def parse_hvslit_ratios(slitratios):
    if slitratios is not None:
        return slitratios
    return None, None


def check_scanlist(scanlist):
    if isinstance(scanlist, Scan):
        return [scanlist]
    return scanlist


def pyfai_setup_limits(experiment: Experiment, scanlist, limitfunction, process_config):
    """
    calculate setup values needed for pyfai calculations
    """
    # pylint: disable=attribute-defined-outside-init
    cfg = process_config
    slithorratio, slitvertratio = parse_hvslit_ratios(cfg.slitratios)
    scanlistnew = check_scanlist(scanlist)

    limhor = None
    limver = None
    for scan in scanlistnew:
        experiment.load_curve_values(scan)

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
    experiment: Experiment, process_config, cornervalues, radmax
):
    cfg = process_config
    full_theta_ranges = np.concatenate([cornervalues, np.abs(cfg.fullranges)])
    centre_check = {1: True, 0: False, 2: False}
    hor_centre = centre_check[np.sum([(val > 0) for val in cfg.fullranges[0:2]])]
    ver_centre = centre_check[np.sum([(val > 0) for val in cfg.fullranges[2:]])]
    if hor_centre and ver_centre:
        radialrange = (0, np.max(cornervalues))
    else:
        radialrange = (min(full_theta_ranges), radmax)
    return radialrange


def get_d5i_values(scan):
    if hasattr(scan.metadata.data_file.nx_entry, "d5i"):
        d5i_full = np.array(scan.metadata.data_file.nx_entry.d5i.data)
    else:
        d5i_full = np.ones(scan.metadata.data_file.scan_length)
    return d5i_full


# ===================================
# ====data saving functions


def save_1d_integration_static(cfg, hf, outlist, scan=None):
    """
    save 1d Intensity Vs Q profile to hdf5 file
    """

    dset = hf.create_group("integrations")
    dset.create_dataset("Intensity", data=outlist[0])
    dset.create_dataset("Q_angstrom^-1", data=outlist[1])
    dset.create_dataset(f"{outlist[3]}", data=outlist[2])

    if (scan is not None) & ("scanfields" not in hf.keys()):
        save_scan_field_values(hf, scan)
    if cfg.savedats is True:
        do_savedats(hf, outlist[0], outlist[1], outlist[2])
    save_config_variables(hf, cfg)
    hf.close()


def save_1d_integration(
    hf, cfg, ints_final, counts_final, tth_vals_final, q_final, tth_string
):

    int_array = np.divide(
        ints_final,
        counts_final,
        out=np.copy(ints_final),
        where=counts_final.astype(float) > 0.0,
    )
    outlist = [int_array, q_final, tth_vals_final, tth_string]
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


# ============


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


def setup_debug_logger(num_cpus: int):
    # sys.stdout.reconfigure(line_buffering=True)
    logger = get_logger(LOGGER_DEBUG)
    listener, log_queue = start_listener()
    print(f"starting multiprocessing with {num_cpus} cpus")
    ram_gb = psutil.virtual_memory().total / 1024**3
    cores = multiprocessing.cpu_count()
    print("Memory per CPU (GB):", ram_gb / cores)
    return logger, listener, log_queue


def setup_job(
    process_config: SimpleNamespace, experiment: Experiment, scan, limit_key
) -> SimpleNamespace:
    cfg = copy.copy(process_config)

    limit_functions = {"ang": experiment.calcanglim, "q": experiment.calcqlim}
    cfg.fullranges, cfg.scanlength, cfg.scanlistnew = pyfai_setup_limits(
        experiment, scan, limit_functions[limit_key], cfg
    )

    cornervalues, radmax = get_corner_thetas(cfg)
    cfg.fullranges = add_buffer_to_limits(cfg.fullranges)
    cfg.scalegamma = 1
    cfg.sample_orientation = 1
    cfg.scan_ind = 0
    if cfg.radialrange is None:
        cfg.radialrange = check_full_1d_radial_range(
            experiment, cfg, cornervalues, radmax
        )

    if cfg.ivqbins is None:
        cfg.ivqbins = int(
            np.ceil((cfg.radialrange[1] - cfg.radialrange[0]) / cfg.radialstepval)
        )
    return cfg


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


def set_ai_slits(aistart: AzimuthalIntegrator | FiberIntegrator, slitratios: tuple):
    if slitratios[0] is not None:
        aistart.pixel1 *= slitratios[0]
        aistart.poni1 *= slitratios[0]

    if slitratios[1] is not None:
        aistart.pixel2 *= slitratios[1]
        aistart.poni2 *= slitratios[1]


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


@dataclass
class pyfai_settings:
    setup: str
    radialrange: np.ndarray
    polarization: int
    shapedataout: np.ndarray
    unit_ip_name: str  # "qip_A^-1"# "qip_A^-1""2th_deg"  #
    unit_oop_name: str
    batchsize: int = 30
    multi: bool = False
    method: tuple = ("no", "csr", "cython")
    azimuthal_sector: np.ndarray | None = None
    sample_orientation: int = 1
    fullranges: np.ndarray | None = None


@dataclass
class angle_info:
    gamma: np.ndarray
    delta: np.ndarray
    two_theta_start: np.ndarray
    incident_angle: float
    alpha_critical: float = 0.0
    scalegamma: int | float = 1


def get_functions_dict(map_per_image: bool) -> dict:
    if map_per_image:
        return {
            "pyfai_qmap": [pyfai_static_qmap_refactor, "Qmap", "2d Qmap"],
            "pyfai_exitangles": [
                pyfai_static_exitangles_refactor,
                "exitmap",
                "2d exit angle map",
            ],
            "pyfai_ivsq": [pyfai_static_ivsq_new_refactor, "IvsQ", "1d integration "],
        }

    return {
        "pyfai_qmap": [pyfai_moving_qmap_smm_refactor, "Qmap", "2d Qmap"],
        "pyfai_exitangles": [
            pyfai_moving_exitangles_refactor,
            "exitmap",
            "2d exit angle map",
        ],
        "pyfai_ivsq": [pyfai_moving_ivsq_smm_refactor, "IvsQ", "1d integration "],
    }


def setup_pool_info_refactor(setup, scan_angles: angle_info, scan, imageindices):

    d5i_full = get_d5i_values(scan)

    incident_angle = scan_angles.incident_angle
    all_inc_angles = [
        get_inc_angles_out(incident_angle, setup, ind) for ind in imageindices
    ]

    gamdelvals = [
        get_gam_del_vals(
            scan_angles.gamma, scan_angles.two_theta_start, scan_angles.delta, ind
        )
        for ind in imageindices
    ]

    return d5i_full, all_inc_angles, gamdelvals


def setup_pool_info(cfg, experiment, scan, imageindices):

    d5i_full = get_d5i_values(scan)

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


def setup_args_iter(
    scan,
    pyfai_info: pyfai_settings,
    scan_angles: angle_info,
    cfg,
    aistart,
    log_queue=None,
    shared=False,
):
    imageindices = get_full_indices(scan, cfg.scanlength, scan_angles.scalegamma)
    batches, num_batches, completed = get_batch_details(
        pyfai_info.multi, imageindices, pyfai_info.batchsize
    )
    # Prepare batches
    d5i_full, all_inc_angles, gamdelvals = setup_pool_info_refactor(
        cfg.setup, scan_angles, scan, imageindices
    )
    # current_ai = setup_copy_ai(cfg.aistart, cfg.slitratios)

    newrots = [
        calc_rots_from_gamdel(
            gamdelvals[ind_n],
            all_inc_angles[ind_n][0],
            scan_angles.alpha_critical,
            cfg.setup,
        )
        for ind_n in np.arange(len(batches))
    ]

    return [
        [
            pyfai_info,
            ind,
            d5i_full[ind_n],
            scan.metadata,
            aistart,
            newrots[ind_n],
            log_queue,
            ind_n,
            shared,
        ]
        for ind_n, ind in enumerate(batches)
    ]


# ===================================
# ====moving detector processing


def run_shared_memory(
    pyfai_info: pyfai_settings, scanangles_list: list[angle_info], pool_setup, cfg
):

    pool_function, aistart, scanlistnew, num_threads = pool_setup
    ctx = get_context("fork")
    with SharedMemoryManager() as smm:
        shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
            smm, pyfai_info.shapedataout
        )
        start_time = time()
        for scan_ind, scan in enumerate(scanlistnew):
            scan_angles = scanangles_list[scan_ind]

            args_iter = setup_args_iter(
                scan, pyfai_info, scan_angles, cfg, aistart, shared=True
            )
            with ctx.Pool(
                processes=num_threads,
                initializer=pyfai_init_worker,
                initargs=(
                    lock,
                    shm_intensities.name,
                    shm_counts.name,
                    pyfai_info.shapedataout,
                ),
            ) as pool:
                mapaxisinfolist = pool.starmap(pool_function, args_iter)

        ints_final = np.zeros(pyfai_info.shapedataout, dtype=np.float32)
        counts_final = np.zeros(pyfai_info.shapedataout, dtype=np.float32)
        shmI = SharedMemory(name=shm_intensities.name)
        shmC = SharedMemory(name=shm_counts.name)

        intensity_view = np.ndarray(
            pyfai_info.shapedataout, dtype=np.float32, buffer=shmI.buf
        )
        count_view = np.ndarray(
            pyfai_info.shapedataout, dtype=np.float32, buffer=shmC.buf
        )

        ints_final += intensity_view
        counts_final += count_view
    return ints_final, counts_final, mapaxisinfolist


def get_scanangles(experiment: Experiment, scan: Scan):
    experiment.load_curve_values(scan)

    return angle_info(
        gamma=experiment.gammadata,
        delta=experiment.deltadata,
        two_theta_start=experiment.two_theta_start,
        incident_angle=experiment.incident_angle,
    )


def pyfai_moving_ivsq_smm_refactor(
    experiment: Experiment, hf, scanlist, process_config
) -> None:
    """
    calculate q_para vs q_perp map for a moving detector scan
    """

    cfg = setup_job(process_config, experiment, scanlist, "ang")
    log_queue = None
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger(cfg.num_threads)
    aistart = setup_initial_ai(cfg.pyfaiponi)
    set_ai_slits(aistart, cfg.slitratios)
    pyfai_info = pyfai_settings(
        setup=cfg.setup,
        radialrange=cfg.radialrange,
        unit_ip_name="2th_deg",
        unit_oop_name="2th_deg",
        shapedataout=np.array([np.abs(cfg.ivqbins)]),
        polarization=cfg.polarization,
        azimuthal_sector=cfg.azimuthal_sector,
    )
    scanangles_list = [get_scanangles(experiment, scan) for scan in cfg.scanlistnew]
    pool_function = worker_unpack("move_ivsq")
    pool_setup = [pool_function, aistart, cfg.scanlistnew, cfg.num_threads]

    ints_final, counts_final, mapaxisinfolist = run_shared_memory(
        pyfai_info, scanangles_list, pool_setup, cfg
    )
    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener
    outaxisinfo = mapaxisinfolist[0]
    tth_vals_final = outaxisinfo[0]
    tth_string = outaxisinfo[1]
    q_final = [calcq(val, experiment.incident_wavelength) for val in tth_vals_final]
    save_1d_integration(
        hf, cfg, ints_final, counts_final, tth_vals_final, q_final, tth_string
    )


def pyfai_moving_qmap_smm_refactor(
    experiment: Experiment, hf, scanlist, process_config
):
    """
    calculate q_para vs q_perp map for a moving detector scan
    """
    cfg = setup_job(process_config, experiment, scanlist, "q")
    log_queue = None
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger(cfg.num_threads)
    aistart = setup_initial_ai(cfg.pyfaiponi)
    set_ai_slits(aistart, cfg.slitratios)
    pyfai_info = pyfai_settings(
        setup=cfg.setup,
        radialrange=cfg.radialrange,
        unit_ip_name="qip_A^-1",
        unit_oop_name="qoop_A^-1",
        shapedataout=np.array(cfg.qmapbins),
        polarization=cfg.polarization,
        fullranges=cfg.fullranges,
    )
    t0 = time()
    scanangles_list = [get_scanangles(experiment, scan) for scan in cfg.scanlistnew]
    pool_function = worker_unpack("move_qmap")
    pool_setup = [pool_function, aistart, cfg.scanlistnew, cfg.num_threads]
    ints_final, counts_final, mapaxisinfolist = run_shared_memory(
        pyfai_info, scanangles_list, pool_setup, cfg
    )

    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener

    save_hf_map(
        hf,
        "qpara_qperp",
        ints_final,
        counts_final,
        mapaxisinfolist[0],
        t0,
        cfg,
    )


def pyfai_moving_exitangles_refactor(
    experiment: Experiment, hf, scanlist, process_config
):
    """
    calculate exit_perp Vs exit para for a moving detector scan
    """

    cfg = setup_job(process_config, experiment, scanlist, "ang")
    log_queue = None
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger(cfg.num_threads)
    aistart = setup_initial_ai(cfg.pyfaiponi)
    set_ai_slits(aistart, cfg.slitratios)
    pyfai_info = pyfai_settings(
        setup=cfg.setup,
        radialrange=cfg.radialrange,
        unit_ip_name="exit_angle_horz_deg",
        unit_oop_name="exit_angle_vert_deg",
        shapedataout=np.array(cfg.qmapbins),
        polarization=cfg.polarization,
        fullranges=cfg.fullranges,
    )
    t0 = time()
    scanangles_list = [get_scanangles(experiment, scan) for scan in cfg.scanlistnew]
    pool_function = worker_unpack("move_exit")
    pool_setup = [pool_function, aistart, cfg.scanlistnew, cfg.num_threads]
    ints_final, counts_final, mapaxisinfolist = run_shared_memory(
        pyfai_info, scanangles_list, pool_setup, cfg
    )

    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener

    save_hf_map(
        hf,
        "exit_angles",
        ints_final,
        counts_final,
        mapaxisinfolist[0],
        t0,
        cfg,
    )


# ===================================
# ====static detector processing


def run_single_scan_pool(pool_function, args_iter, num_threads):
    mapped_data, mapaxisinfo, mask_info = [[], [], []]
    ctx = get_context("fork")
    completed = 0
    with ctx.Pool(processes=num_threads) as pool:
        for partial in pool.starmap(pool_function, args_iter, chunksize=1):
            if completed == 0:
                mask_info.append(partial[2])
                mapaxisinfo.append(partial)
            mapped_data.append(partial[0])
            # labels1.append(partial[1])

            completed += 1
            if completed % 10 == 0 or completed == len(args_iter):
                print(f"  completed {completed}/{len(args_iter)} batches", flush=True)
    return mapped_data, mapaxisinfo, mask_info


def pyfai_static_ivsq_new_refactor(
    experiment: Experiment, hf, scan, process_config: SimpleNamespace
):
    """
    calculate Intensity Vs Q 1d profile from static detector scan
    """
    cfg = setup_job(process_config, experiment, scan, "ang")
    log_queue = None
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger(cfg.num_threads)
    aistart = setup_initial_ai(cfg.pyfaiponi)
    set_ai_slits(aistart, cfg.slitratios)
    pyfai_info = pyfai_settings(
        setup=cfg.setup,
        radialrange=cfg.radialrange,
        unit_ip_name="2th_deg",
        unit_oop_name="2th_deg",
        shapedataout=np.array([np.abs(cfg.ivqbins)]),
        polarization=cfg.polarization,
        azimuthal_sector=cfg.azimuthal_sector,
    )

    pool_function = worker_unpack("static_ivsq")
    scan_angles = get_scanangles(experiment, scan)
    args_iter = setup_args_iter(
        scan, pyfai_info, scan_angles, cfg, aistart, shared=False
    )

    mapped_data, mapaxisinfo, mask_info = run_single_scan_pool(
        pool_function=pool_function, args_iter=args_iter, num_threads=cfg.num_threads
    )
    two_th_vals = mapaxisinfo[0][0]
    q_vals = [calcq(val, experiment.incident_wavelength) for val in two_th_vals]

    inlist = [mapped_data[0], q_vals, two_th_vals]
    outlist = check_data_shape(inlist, scan)
    outlist.append(mapaxisinfo[0][1])
    save_masks(hf, mask_info[0])

    save_1d_integration_static(cfg, hf, outlist, scan)
    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener


def pyfai_static_qmap_refactor(
    experiment: Experiment, hf, scan, process_config: SimpleNamespace
):
    cfg = setup_job(process_config, experiment, scan, "q")
    log_queue = None
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger(cfg.num_threads)
    aistart = setup_initial_ai(cfg.pyfaiponi)
    set_ai_slits(aistart, cfg.slitratios)
    pyfai_info = pyfai_settings(
        setup=cfg.setup,
        radialrange=cfg.radialrange,
        unit_ip_name="qip_A^-1",
        unit_oop_name="qoop_A^-1",
        shapedataout=np.array(cfg.qmapbins),
        polarization=cfg.polarization,
        fullranges=cfg.fullranges,
    )
    t0 = time()
    pool_function = worker_unpack("static_qmap")
    scan_angles = get_scanangles(experiment, scan)
    args_iter = setup_args_iter(
        scan, pyfai_info, scan_angles, cfg, aistart, shared=False
    )

    mapped_data, mapaxisinfo, *_ = run_single_scan_pool(
        pool_function=pool_function, args_iter=args_iter, num_threads=cfg.num_threads
    )
    outdata = check_data_shape(mapped_data, scan)
    save_hf_map_static(hf, cfg, t0, "qpara_qperp", outdata, mapaxisinfo[0], scan)
    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener


def pyfai_static_exitangles_refactor(
    experiment: Experiment, hf, scan, process_config: SimpleNamespace
):
    cfg = setup_job(process_config, experiment, scan, "ang")
    log_queue = None
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger(cfg.num_threads)
    aistart = setup_initial_ai(cfg.pyfaiponi)
    set_ai_slits(aistart, cfg.slitratios)
    pyfai_info = pyfai_settings(
        setup=cfg.setup,
        radialrange=cfg.radialrange,
        unit_ip_name="exit_angle_horz_deg",
        unit_oop_name="exit_angle_vert_deg",
        shapedataout=np.array(cfg.qmapbins),
        polarization=cfg.polarization,
        fullranges=cfg.fullranges,
    )
    t0 = time()
    pool_function = worker_unpack("static_exit")
    scan_angles = get_scanangles(experiment, scan)
    args_iter = setup_args_iter(
        scan, pyfai_info, scan_angles, cfg, aistart, shared=False
    )
    print(pyfai_info)
    mapped_data, mapaxisinfo, *_ = run_single_scan_pool(
        pool_function=pool_function, args_iter=args_iter, num_threads=cfg.num_threads
    )
    outdata = check_data_shape(mapped_data, scan)
    save_hf_map_static(hf, cfg, t0, "exit_angles", outdata, mapaxisinfo[0], scan)
    if cfg.debuglogging:
        log_queue.put_nowait(None)  # End the queue
        listener.join()  # Stop the listener


def pyfai_static_exitangles_new(
    experiment: Experiment, hf, scan, process_config: SimpleNamespace
):

    cfg = setup_job(process_config, experiment, scan, "ang")
    log_queue = None
    if cfg.debuglogging:
        logger, listener, log_queue = setup_debug_logger(cfg.num_threads)

    t0 = time()
    cfg.multi = False
    cfg.unit_qip_name = "exit_angle_horz_deg"  # "2th_deg"  # "qtot_A^-1"# "qip_A^-1"
    cfg.unit_qoop_name = "exit_angle_vert_deg"
    # "2th_deg"
    cfg.aistart = setup_initial_ai(cfg.pyfaiponi)
    cfg.batchsize = 30

    ctx = get_context("fork")

    all_maps, all_xlabels, all_ylabels, scan_mask, all_mapaxisinfo = [
        [],
        [],
        [],
        [],
        [],
    ]

    experiment.load_curve_values(scan)
    imageindices = get_full_indices(scan, cfg.scanlength, cfg.scalegamma)
    batches, num_batches, completed = get_batch_details(
        cfg.multi, imageindices, cfg.batchsize
    )

    d5i_full, all_inc_angles, gamdelvals = setup_pool_info(
        cfg, experiment, scan, imageindices
    )
    current_ai = setup_copy_ai(cfg.aistart, cfg.slitratios)
    pool_function = worker_unpack("static_exit")

    alphacritical = cfg.alphacritical
    setup = cfg.setup
    newrots = [
        calc_rots_from_gamdel(
            gamdelvals[ind_n], all_inc_angles[ind_n][0], alphacritical, setup
        )
        for ind_n in np.arange(len(batches))
    ]
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

    with ctx.Pool(processes=cfg.num_threads) as pool:
        for partial in pool.starmap(pool_function, args_iter, chunksize=1):
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


# def pyfai_static_qmap_new(
#     experiment: Experiment, hf, scan, process_config: SimpleNamespace
# ):

#     cfg = setup_job(process_config, experiment, scan, "q")
#     log_queue = None
#     if cfg.debuglogging:
#         logger, listener, log_queue = setup_debug_logger(cfg.num_threads)

#     t0 = time()
#     cfg.multi = False
#     cfg.unit_qip_name = "qip_A^-1"  # "qip_A^-1"# "qip_A^-1""2th_deg"  #
#     cfg.unit_qoop_name = "qoop_A^-1"  # "qoop_A^-1"
#     print(f"starting process pool with num_threads={cfg.num_threads}")
#     cfg.aistart = setup_initial_ai(cfg.pyfaiponi)
#     cfg.batchsize = 30

#     ctx = get_context("fork")
#     all_maps, all_xlabels, all_ylabels, scan_mask, all_mapaxisinfo = [
#         [],
#         [],
#         [],
#         [],
#         [],
#     ]

#     experiment.load_curve_values(scan)
#     imageindices = get_full_indices(scan, cfg.scanlength, cfg.scalegamma)
#     batches, num_batches, completed = get_batch_details(
#         cfg.multi, imageindices, cfg.batchsize
#     )

#     d5i_full, all_inc_angles, gamdelvals = setup_pool_info(
#         cfg, experiment, scan, imageindices
#     )
#     current_ai = setup_copy_ai(cfg.aistart, cfg.slitratios)
#     pool_function = worker_unpack("static_qmap")

#     alphacritical = cfg.alphacritical
#     setup = cfg.setup
#     newrots = [
#         calc_rots_from_gamdel(
#             gamdelvals[ind_n], all_inc_angles[ind_n][0], alphacritical, setup
#         )
#         for ind_n in np.arange(len(batches))
#     ]
#     args_iter = [
#         [
#             ind,
#             d5i_full[ind_n],
#             scan.metadata,
#             current_ai,
#             newrots[ind_n],
#             cfg,
#             log_queue,
#             ind_n,
#         ]
#         for ind_n, ind in enumerate(batches)
#     ]
#     with ctx.Pool(processes=cfg.num_threads) as pool:
#         for partial in pool.starmap(pool_function, args_iter, chunksize=1):
#             if completed == 0:
#                 scan_mask.append(partial[4])
#             all_maps.append(partial[0])
#             all_xlabels.append(partial[1])
#             all_ylabels.append(partial[2])
#             all_mapaxisinfo.append(partial[3])
#             completed += 1
#             if completed % 10 == 0 or completed == num_batches:
#                 print(f"  completed {completed}/{num_batches} batches", flush=True)
#     print("finished process pool")
#     inlist = [all_maps, all_xlabels, all_ylabels]
#     outlist = check_data_shape(inlist, scan)
#     save_hf_map_static(hf, cfg, t0, "qpara_qperp", outlist[0], all_mapaxisinfo[0], scan)
#     if cfg.debuglogging:
#         log_queue.put_nowait(None)  # End the queue
#         listener.join()  # Stop the listener


# def pyfai_moving_ivsq_smm_new(experiment: Experiment, hf, scanlist, process_config):
#     """
#     calculate q_para vs q_perp map for a moving detector scan
#     """

#     cfg = setup_job(process_config, experiment, scanlist, "ang")
#     log_queue = None
#     if cfg.debuglogging:
#         logger, listener, log_queue = setup_debug_logger(cfg.num_threads)

#     cfg.multi = False
#     cfg.unit_qip_name = "2th_deg"  # "qip_A^-1"# "qip_A^-1""2th_deg"  #
#     cfg.unit_qoop_name = "2th_deg"
#     # "2th_deg"
#     cfg.aistart = setup_initial_ai(cfg.pyfaiponi)
#     cfg.batchsize = 30
#     cfg.shapedataout = np.abs(cfg.ivqbins)
#     t0 = time()
#     ctx = get_context("fork")
#     with SharedMemoryManager() as smm:
#         shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
#             smm, cfg.shapedataout
#         )
#         start_time = time()
#         for scanind, scan in enumerate(cfg.scanlistnew):
#             experiment.load_curve_values(scan)

#             imageindices = get_full_indices(scan, cfg.scanlength, cfg.scalegamma)
#             batches, num_batches, completed = get_batch_details(
#                 cfg.multi, imageindices, cfg.batchsize
#             )
#             # Prepare batches
#             d5i_full, all_inc_angles, gamdelvals = setup_pool_info(
#                 cfg, experiment, scan, imageindices
#             )
#             current_ai = setup_copy_ai(cfg.aistart, cfg.slitratios)

#             pool_function = worker_unpack("move_ivsq")

#             alphacritical = cfg.alphacritical
#             setup = cfg.setup
#             newrots = [
#                 calc_rots_from_gamdel(
#                     gamdelvals[ind_n], all_inc_angles[ind_n][0], alphacritical, setup
#                 )
#                 for ind_n in np.arange(len(batches))
#             ]

#             args_iter = [
#                 [
#                     ind,
#                     d5i_full[ind_n],
#                     scan.metadata,
#                     current_ai,
#                     newrots[ind_n],
#                     cfg,
#                     log_queue,
#                     ind_n,
#                     True,
#                 ]
#                 for ind_n, ind in enumerate(batches)
#             ]
#             with ctx.Pool(
#                 processes=cfg.num_threads,
#                 initializer=pyfai_init_worker,
#                 initargs=(
#                     lock,
#                     shm_intensities.name,
#                     shm_counts.name,
#                     cfg.shapedataout,
#                 ),
#             ) as pool:
#                 mapaxisinfolist = pool.starmap(pool_function, args_iter)

#         ints_final = np.zeros(cfg.shapedataout, dtype=np.float32)
#         counts_final = np.zeros(cfg.shapedataout, dtype=np.float32)
#         shmI = SharedMemory(name=shm_intensities.name)
#         shmC = SharedMemory(name=shm_counts.name)

#         intensity_view = np.ndarray(cfg.shapedataout, dtype=np.float32, buffer=shmI.buf)
#         count_view = np.ndarray(cfg.shapedataout, dtype=np.float32, buffer=shmC.buf)

#         ints_final += intensity_view
#         counts_final += count_view

#     if cfg.debuglogging:
#         log_queue.put_nowait(None)  # End the queue
#         listener.join()  # Stop the listener
#     outaxisinfo = mapaxisinfolist[0]
#     tth_vals_final = outaxisinfo[0]
#     tth_string = outaxisinfo[1]
#     q_final = [calcq(val, experiment.incident_wavelength) for val in tth_vals_final]
#     save_1d_integration(
#         hf, cfg, ints_final, counts_final, tth_vals_final, q_final, tth_string
#     )


# def pyfai_static_ivsq_new(
#     experiment: Experiment, hf, scan, process_config: SimpleNamespace
# ):
#     """
#     calculate Intensity Vs Q 1d profile from static detector scan
#     """
#     cfg = setup_job(process_config, experiment, scan, "ang")
#     log_queue = None
#     if cfg.debuglogging:
#         logger, listener, log_queue = setup_debug_logger(cfg.num_threads)

#     cfg.multi = False
#     cfg.unit_qip_name = "2th_deg"  # "qip_A^-1"# "qip_A^-1""2th_deg"  #
#     cfg.unit_qoop_name = "2th_deg"
#     # "2th_deg"
#     cfg.aistart = setup_initial_ai(cfg.pyfaiponi)
#     cfg.batchsize = 30
#     cfg.shapedataout = np.abs(cfg.ivqbins)
#     t0 = time()
#     ctx = get_context("fork")
#     experiment.load_curve_values(scan)
#     imageindices = get_full_indices(scan, cfg.scanlength, cfg.scalegamma)
#     batches, num_batches, completed = get_batch_details(
#         cfg.multi, imageindices, cfg.batchsize
#     )
#     # Prepare batches
#     d5i_full, all_inc_angles, gamdelvals = setup_pool_info(
#         cfg, experiment, scan, imageindices
#     )
#     current_ai = setup_copy_ai(cfg.aistart, cfg.slitratios)

#     pool_function = worker_unpack("static_ivsq")

#     alphacritical = cfg.alphacritical
#     setup = cfg.setup
#     newrots = [
#         calc_rots_from_gamdel(
#             gamdelvals[ind_n], all_inc_angles[ind_n][0], alphacritical, setup
#         )
#         for ind_n in np.arange(len(batches))
#     ]

#     args_iter = [
#         [
#             ind,
#             d5i_full[ind_n],
#             scan.metadata,
#             current_ai,
#             newrots[ind_n],
#             cfg,
#             log_queue,
#             ind_n,
#         ]
#         for ind_n, ind in enumerate(batches)
#     ]
#     scan_masks, axis_name, all_ints, all_two_ths, all_qs = [[], [], [], [], []]
#     with ctx.Pool(processes=cfg.num_threads) as pool:
#         for partial in pool.starmap(pool_function, args_iter, chunksize=1):
#             if completed == 0:
#                 scan_masks.append(partial[2])
#                 axis_name.append(partial[3])
#             all_ints.append(partial[0])
#             all_two_ths.append(partial[1])
#             all_qs.append(
#                 [calcq(val, experiment.incident_wavelength) for val in partial[1]]
#             )
#             completed += 1
#             if completed % 10 == 0 or completed == num_batches:
#                 print(f"  completed {completed}/{num_batches} batches", flush=True)

#     inlist = [all_ints, all_qs, all_two_ths]
#     outlist = check_data_shape(inlist, scan)
#     outlist.append(axis_name[0])
#     save_masks(hf, scan_masks[0])

#     save_1d_integration_static(cfg, hf, outlist, scan)
#     if cfg.debuglogging:
#         log_queue.put_nowait(None)  # End the queue
#         listener.join()  # Stop the listener


# def pyfai_moving_qmap_smm_new(experiment: Experiment, hf, scanlist, process_config):
#     """
#     calculate q_para vs q_perp map for a moving detector scan
#     """

#     cfg = setup_job(process_config, experiment, scanlist, "q")
#     log_queue = None
#     if cfg.debuglogging:
#         logger, listener, log_queue = setup_debug_logger(cfg.num_threads)

#     cfg.multi = False
#     cfg.unit_qip_name = "qip_A^-1"  # "2th_deg"  # "qtot_A^-1"# "qip_A^-1"
#     cfg.unit_qoop_name = "qoop_A^-1"
#     # "2th_deg"
#     cfg.aistart = setup_initial_ai(cfg.pyfaiponi)
#     cfg.batchsize = 30

#     t0 = time()
#     ctx = get_context("fork")
#     with SharedMemoryManager() as smm:
#         cfg.shapedataout = (cfg.qmapbins[1], cfg.qmapbins[0])
#         shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
#             smm, cfg.shapedataout
#         )
#         start_time = time()
#         for scanind, scan in enumerate(cfg.scanlistnew):
#             experiment.load_curve_values(scan)
#             imageindices = get_full_indices(scan, cfg.scanlength, cfg.scalegamma)
#             batches, num_batches, completed = get_batch_details(
#                 cfg.multi, imageindices, cfg.batchsize
#             )
#             # Prepare batches
#             d5i_full, all_inc_angles, gamdelvals = setup_pool_info(
#                 cfg, experiment, scan, imageindices
#             )
#             current_ai = setup_copy_ai(cfg.aistart, cfg.slitratios)

#             pool_function = worker_unpack("move_qmap")

#             alphacritical = cfg.alphacritical
#             setup = cfg.setup
#             newrots = [
#                 calc_rots_from_gamdel(
#                     gamdelvals[ind_n], all_inc_angles[ind_n][0], alphacritical, setup
#                 )
#                 for ind_n in np.arange(len(batches))
#             ]

#             args_iter = [
#                 [
#                     ind,
#                     d5i_full[ind_n],
#                     scan.metadata,
#                     current_ai,
#                     newrots[ind_n],
#                     cfg,
#                     log_queue,
#                     ind_n,
#                     True,
#                 ]
#                 for ind_n, ind in enumerate(batches)
#             ]
#             with ctx.Pool(
#                 processes=cfg.num_threads,
#                 initializer=pyfai_init_worker,
#                 initargs=(
#                     lock,
#                     shm_intensities.name,
#                     shm_counts.name,
#                     cfg.shapedataout,
#                 ),
#             ) as pool:
#                 mapaxisinfolist = pool.starmap(pool_function, args_iter)

#         ints_final = np.zeros(cfg.shapedataout, dtype=np.float32)
#         counts_final = np.zeros(cfg.shapedataout, dtype=np.float32)
#         shmI = SharedMemory(name=shm_intensities.name)
#         shmC = SharedMemory(name=shm_counts.name)

#         intensity_view = np.ndarray(cfg.shapedataout, dtype=np.float32, buffer=shmI.buf)
#         count_view = np.ndarray(cfg.shapedataout, dtype=np.float32, buffer=shmC.buf)

#         ints_final += intensity_view
#         counts_final += count_view

#     if cfg.debuglogging:
#         log_queue.put_nowait(None)  # End the queue
#         listener.join()  # Stop the listener

#     save_hf_map(
#         hf,
#         "qpara_qperp",
#         ints_final,
#         counts_final,
#         mapaxisinfolist[0],
#         t0,
#         cfg,
#     )

# def pyfai_moving_exitangles_smm_new(
#     experiment: Experiment, hf, scanlist, process_config
# ):
#     """
#     calculate q_para vs q_perp map for a moving detector scan
#     """

#     cfg = setup_job(process_config, experiment, scanlist, "ang")
#     log_queue = None
#     if cfg.debuglogging:
#         logger, listener, log_queue = setup_debug_logger(cfg.num_threads)

#     t0 = time()
#     cfg.multi = False
#     cfg.unit_qip_name = "exit_angle_horz_deg"  # "2th_deg"  # "qtot_A^-1"# "qip_A^-1"
#     cfg.unit_qoop_name = "exit_angle_vert_deg"
#     # "2th_deg"
#     cfg.aistart = setup_initial_ai(cfg.pyfaiponi)
#     cfg.batchsize = 30

#     ctx = get_context("fork")
#     with SharedMemoryManager() as smm:
#         cfg.shapedataout = (cfg.qmapbins[1], cfg.qmapbins[0])
#         shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
#             smm, cfg.shapedataout
#         )
#         start_time = time()
#         for scanind, scan in enumerate(cfg.scanlistnew):
#             experiment.load_curve_values(scan)
#             imageindices = get_full_indices(scan, cfg.scanlength, cfg.scalegamma)
#             batches, num_batches, completed = get_batch_details(
#                 cfg.multi, imageindices, cfg.batchsize
#             )

#             d5i_full, all_inc_angles, gamdelvals = setup_pool_info(
#                 cfg, experiment, scan, imageindices
#             )
#             current_ai = setup_copy_ai(cfg.aistart, cfg.slitratios)
#             pool_function = worker_unpack("move_exit")

#             alphacritical = cfg.alphacritical
#             setup = cfg.setup
#             newrots = [
#                 calc_rots_from_gamdel(
#                     gamdelvals[ind_n], all_inc_angles[ind_n][0], alphacritical, setup
#                 )
#                 for ind_n in np.arange(len(batches))
#             ]
#             args_iter = [
#                 [
#                     ind,
#                     d5i_full[ind_n],
#                     scan.metadata,
#                     current_ai,
#                     newrots[ind_n],
#                     cfg,
#                     log_queue,
#                     ind_n,
#                     True,
#                 ]
#                 for ind_n, ind in enumerate(batches)
#             ]

#             with ctx.Pool(
#                 processes=cfg.num_threads,
#                 initializer=pyfai_init_worker,
#                 initargs=(
#                     lock,
#                     shm_intensities.name,
#                     shm_counts.name,
#                     cfg.shapedataout,
#                 ),
#             ) as pool:
#                 mapaxisinfolist = pool.starmap(pool_function, args_iter)

#         ints_final = np.zeros(cfg.shapedataout, dtype=np.float32)
#         counts_final = np.zeros(cfg.shapedataout, dtype=np.float32)
#         shmI = SharedMemory(name=shm_intensities.name)
#         shmC = SharedMemory(name=shm_counts.name)

#         intensity_view = np.ndarray(cfg.shapedataout, dtype=np.float32, buffer=shmI.buf)
#         count_view = np.ndarray(cfg.shapedataout, dtype=np.float32, buffer=shmC.buf)

#         ints_final += intensity_view
#         counts_final += count_view

#     if cfg.debuglogging:
#         log_queue.put_nowait(None)  # End the queue
#         listener.join()  # Stop the listener

#     save_hf_map(
#         hf,
#         "exit_angles",
#         ints_final,
#         counts_final,
#         mapaxisinfolist[0],
#         t0,
#         cfg,
#     )
