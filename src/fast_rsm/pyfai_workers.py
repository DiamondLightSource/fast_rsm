import copy
import logging
import logging.handlers
import sys
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from pyFAI import units
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.integrator.fiber import FiberIntegrator

from fast_rsm.angle_pixel_q import gamdel2rots
from fast_rsm.experiment import Experiment
from fast_rsm.image import Image
from fast_rsm.logging_config import get_logger

# ==============common functions


def get_pyfai_ai(
    experiment: Experiment, aistart, slitratios, alphacritical, inc_angle, gamdelval
):
    """
    get components need for mapping with pyFAI
    """

    gamval, delval = gamdelval
    if (-np.degrees(inc_angle) > alphacritical) & (experiment.setup == "DCD"):
        # if above critical angle, account for direct beam adding to delta
        rots = gamdel2rots(gamval, delval + np.degrees(-inc_angle))
    else:
        rots = gamdel2rots(gamval, delval)

    out_ai = copy.deepcopy(aistart)
    out_ai.rot1, out_ai.rot2, out_ai.rot3 = rots

    if experiment.setup == "vertical":
        out_ai.rot1 = rots[1]
        out_ai.rot2 = -rots[0]

    if slitratios[0] is not None:
        out_ai.pixel1 *= slitratios[0]
        out_ai.poni1 *= slitratios[0]

    if slitratios[1] is not None:
        out_ai.pixel2 *= slitratios[1]
        out_ai.poni2 *= slitratios[1]
    return out_ai


def setup_start_ai(aistart, slitratios):
    out_ai = copy.deepcopy(aistart)
    if slitratios[0] is not None:
        out_ai.pixel1 *= slitratios[0]
        out_ai.poni1 *= slitratios[0]

    if slitratios[1] is not None:
        out_ai.pixel2 *= slitratios[1]
        out_ai.poni2 *= slitratios[1]
    return out_ai


def get_pyfai_limits(limits_in):
    radial_limits = (
        limits_in[0] * (1.0 + (0.05 * -(np.sign(limits_in[0])))),
        limits_in[1] * (1.0 + (0.05 * (np.sign(limits_in[1])))),
    )
    azimuthal_limits = (
        limits_in[2] * (1.0 + (0.05 * -(np.sign(limits_in[2])))),
        limits_in[3] * (1.0 + (0.05 * (np.sign(limits_in[3])))),
    )
    return [
        radial_limits[0],
        radial_limits[1],
        azimuthal_limits[0],
        azimuthal_limits[1],
    ]


def get_pyfai_image_data(setup: str, metadata, idx):

    # outimage = scan.load_image(i)
    outimage = Image(metadata, idx, load_image=True)
    mask_nan = np.isnan(outimage.data).astype(bool)
    mask_negative = outimage.data < 0
    fullmask = np.logical_or(mask_nan, mask_negative)
    if setup == "vertical":
        return np.rot90(outimage.data, -1), np.rot90(fullmask, -1)

    return np.array(outimage.data), fullmask


def setup_stat_worker(
    imageindex,
    all_inc_angles,
    gamdelvals,
    d5i_full,
):
    inc_angle, inc_angle_out = all_inc_angles[imageindex]
    gamdelval = gamdelvals[imageindex]
    d5i_data = d5i_full[imageindex]

    return inc_angle, inc_angle_out, d5i_data, gamdelval


@lru_cache
def setup_ip_oop_units(
    unit_qip_name, unit_qoop_name, sample_orientation, inc_angle_out=0
):
    unit_ip = units.get_unit_fiber(
        unit_qip_name,
        sample_orientation=sample_orientation,
        incident_angle=inc_angle_out,
    )
    unit_oop = units.get_unit_fiber(
        unit_qoop_name,
        sample_orientation=sample_orientation,
        incident_angle=inc_angle_out,
    )
    return unit_ip, unit_oop


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


def calculate_2d_map(
    pyfai_info: pyfai_settings,
    ai: AzimuthalIntegrator,
    img_data: np.ndarray,
    norm_data: float | int,
):
    map2d = ai.integrate2d(
        img_data,
        npt_rad=pyfai_info.shapedataout[1],
        npt_azim=pyfai_info.shapedataout[0],
        unit=(pyfai_info.unit_ip, pyfai_info.unit_oop),
        radial_range=(pyfai_info.fullranges[0], pyfai_info.fullranges[1]),
        azimuth_range=(pyfai_info.fullranges[2], pyfai_info.fullranges[3]),
        method=pyfai_info.method,
        polarization_factor=pyfai_info.polarization,
        normalization_factor=norm_data,
    )
    mapaxisinfo = [
        map2d.azimuthal,
        map2d.radial,
        str(map2d.azimuthal_unit),
        str(map2d.radial_unit),
    ]
    return map2d, mapaxisinfo


def calculate_2d_map_fiber(
    pyfai_info: pyfai_settings,
    fi: FiberIntegrator,
    img_data: np.ndarray,
    norm_data: float | int,
):
    map2d = fi.integrate2d_fiber(
        img_data,
        npt_oop=pyfai_info.shapedataout[1],
        npt_ip=pyfai_info.shapedataout[0],
        unit_ip=pyfai_info.unit_ip_name,
        oop_range=pyfai_info.azimuthal_sector - 90,
        unit_oop=pyfai_info.unit_oop_name,
        ip_range=pyfai_info.radialrange,
        sample_orientation=pyfai_info.sample_orientation,
        polarization_factor=pyfai_info.polarization,
        method=pyfai_info.method,
        normalization_factor=norm_data,
    )
    mapaxisinfo = [
        map2d.outofplane,
        map2d.inplane,
        str(map2d.oop_unit),
        str(map2d.ip_unit),
    ]
    return map2d, mapaxisinfo


def calculate_1d_fiber(
    pyfai_info: pyfai_settings,
    fi: FiberIntegrator,
    img_data: np.ndarray,
    norm_data: float | int,
):

    result1d = fi.integrate_fiber(
        img_data,
        npt_oop=int(pyfai_info.shapedataout),
        npt_ip=int(pyfai_info.shapedataout),
        unit_ip=pyfai_info.unit_ip_name,
        oop_range=pyfai_info.azimuthal_sector - 90,
        unit_oop=pyfai_info.unit_oop_name,
        ip_range=pyfai_info.radialrange,
        vertical_integration=True,
        sample_orientation=pyfai_info.sample_orientation,
        polarization_factor=pyfai_info.polarization,
        method=pyfai_info.method,
        normalization_factor=norm_data,
    )
    mapaxisinfo = [
        result1d.integrated,
        str(result1d._unit),
    ]
    return result1d, mapaxisinfo


def calculate_1d(
    pyfai_info: pyfai_settings,
    ai: AzimuthalIntegrator,
    img_data: np.ndarray,
    norm_data: float | int,
):

    result1d = ai.integrate1d(
        img_data,
        pyfai_info.shapedataout,
        unit=pyfai_info.unit_ip_name,
        normalization_factor=norm_data,
        method=pyfai_info.method,
        radial_range=pyfai_info.radialrange,
        polarization_factor=pyfai_info.polarization,
    )
    mapaxisinfo = [
        result1d.radial,
        str(result1d._unit),
    ]
    return result1d, mapaxisinfo


def get_sector_mask(ai, shape, sector_ranges):
    if sector_ranges is None:
        return np.zeros(shape)
    chivals = ai.center_array(unit="chi_rad")
    chi_min, chi_max = ai.normalize_azimuth_range(sector_ranges)
    return np.logical_or(chivals > chi_max, chivals < chi_min)


def get_time_logger(queue, logn):
    LOGGER_DEBUG = "fastrsm_debug"
    # LOGGER_ERROR = "fastrsm_error"
    # debug_logger = get_debug_logger()
    sys.stdout.reconfigure(line_buffering=True)

    time_logger = get_logger(LOGGER_DEBUG)
    if queue is not None:
        # print('found queue')
        qh = logging.handlers.QueueHandler(queue)
        root_logger = get_logger(LOGGER_DEBUG)
        root_logger.addHandler(qh)
        root_logger.setLevel(logging.DEBUG)
        time_logger = root_logger.getChild(f"child_{logn}")
        time_logger.debug(f"created logger for child_{logn}")

    return time_logger


def pyfai_init_worker(lock, shm_intensities_name, shm_counts_name, shmshape):
    """
    intialiser for pyfai mappings
    """
    global LOCK
    global SHM_INTENSITY
    global INTENSITY_ARRAY
    global SHM_COUNT
    global COUNT_ARRAY

    SHM_INTENSITY = SharedMemory(name=shm_intensities_name)
    SHM_COUNT = SharedMemory(name=shm_counts_name)
    INTENSITY_ARRAY = np.ndarray(
        shape=shmshape, dtype=np.float32, buffer=SHM_INTENSITY.buf
    )
    COUNT_ARRAY = np.ndarray(shape=shmshape, dtype=np.float32, buffer=SHM_COUNT.buf)
    LOCK = lock


# -----------------------------
# Dispatcher
# -----------------------------
def worker_unpack(worker_type):
    function_map = {
        "move_ivsq": pyfai_1dmap_worker,  # pyfai_1dmap_worker,
        "move_qmap": pyfai_2dmap_worker,  # pyfai_move_qmap_worker_new,
        "move_exit": pyfai_2dmap_worker,
        "static_ivsq": pyfai_1dmap_worker,
        "static_qmap": pyfai_2dmap_worker,
        "static_exit": pyfai_2dmap_worker,
        "static_ivschi": pyfai_1dmap_worker,
        "static_chimap": pyfai_2dmap_worker,
    }
    # worker_type = args[0]
    return function_map[worker_type]
    # worker_args = args[1:]
    # top-level adapter to avoid lambda pickling issues
    # return worker_function(*worker_args)


def save_intcountshm_withlock(result):
    global INTENSITY_ARRAY, COUNT_ARRAY
    with LOCK:
        INTENSITY_ARRAY += result._sum_signal
        COUNT_ARRAY += result.count.astype(dtype=np.int32)


def set_ai_rots(rots, current_ai, setup):
    """
    get components need for mapping with pyFAI
    """
    current_ai.rot1, current_ai.rot2, current_ai.rot3 = rots

    if setup == "vertical":
        current_ai.rot1 = rots[1]
        current_ai.rot2 = -rots[0]


def set_ai_mask(current_ai, img_data_shape, img_mask, azimuthal_sector) -> np.ndarray:
    # if current_ai.mask is None:
    #     current_ai.mask = np.array(img_mask, copy=True)
    #     mask = current_ai.mask
    # else:
    #     np.copyto(mask, img_mask)

    sector_mask = get_sector_mask(current_ai, img_data_shape, azimuthal_sector)
    current_ai.mask = np.logical_or(img_mask, sector_mask)
    return np.array([img_mask, sector_mask])


def pyfai_1dmap_worker(
    pyfai_info,
    imageind,
    d5i,
    metadata,
    start_ai,
    newrot,
    log_queue,
    logn=None,
    shared=False,
) -> tuple:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
    """

    ind = imageind
    current_ai = copy.deepcopy(start_ai)

    set_ai_rots(newrot, current_ai, pyfai_info.setup)
    img_data, img_mask = get_pyfai_image_data(pyfai_info.setup, metadata, ind)
    sector_mask = get_sector_mask(
        current_ai, img_data.shape, pyfai_info.azimuthal_sector
    )
    current_ai.mask = np.logical_or(img_mask, sector_mask)

    if isinstance(current_ai, FiberIntegrator):
        map_func = calculate_1d_fiber
    elif isinstance(current_ai, AzimuthalIntegrator):
        map_func = calculate_1d
    res1d, mapaxisinfo = map_func(
        pyfai_info,
        current_ai,
        img_data,
        d5i,
    )
    if shared:
        save_intcountshm_withlock(res1d)
        return mapaxisinfo

    return (
        res1d.intensity,
        mapaxisinfo,
        [current_ai.mask, img_mask, sector_mask],
    )


def pyfai_2dmap_worker(
    pyfai_info,
    imageind,
    d5i,
    metadata,
    start_ai,
    newrot,
    log_queue,
    logn=None,
    shared=False,
) -> tuple:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
    """

    ind = imageind
    current_ai = copy.deepcopy(start_ai)

    set_ai_rots(newrot, current_ai, pyfai_info.setup)
    img_data, img_mask = get_pyfai_image_data(pyfai_info.setup, metadata, ind)
    sector_mask = get_sector_mask(
        current_ai, img_data.shape, pyfai_info.azimuthal_sector
    )
    current_ai.mask = np.logical_or(img_mask, sector_mask)
    pyfai_info.unit_ip, pyfai_info.unit_oop = setup_ip_oop_units(
        pyfai_info.unit_ip_name,
        pyfai_info.unit_oop_name,
        pyfai_info.sample_orientation,
    )

    if isinstance(current_ai, FiberIntegrator):
        map_func = calculate_2d_map_fiber
    elif isinstance(current_ai, AzimuthalIntegrator):
        map_func = calculate_2d_map
    map2d, mapaxisinfo = map_func(
        pyfai_info,
        current_ai,
        img_data,
        d5i,
    )
    if shared:
        save_intcountshm_withlock(map2d)
        return mapaxisinfo

    return (map2d[0], mapaxisinfo, current_ai.mask)


# def pyfai_1dmap_worker(
#     imageind,
#     d5i,
#     metadata,
#     current_ai,
#     newrot,
#     cfg,
#     log_queue,
#     logn=None,
#     shared=False,
# ) -> tuple:
#     """
#     calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
#     """

#     ind = imageind

#     mask = current_ai.mask
#     method = ("no", "csr", "cython")
#     # alphacritical = cfg.alphacritical
#     setup = cfg.setup

#     polarization, ivqbins, radialrange, unit_ip = (
#         cfg.polarization,
#         cfg.ivqbins,
#         cfg.radialrange,
#         cfg.unit_qip_name,
#     )
#     set_ai_rots(newrot, current_ai, setup)
#     img_data, img_mask = get_pyfai_image_data(setup, metadata, ind)

#     if current_ai.mask is None:
#         current_ai.mask = np.array(img_mask, copy=True)
#         mask = current_ai.mask
#     else:
#         np.copyto(mask, img_mask)
#     sector_mask = get_sector_mask(current_ai, img_data.shape, cfg.azimuthal_sector)
#     current_ai.mask = np.logical_or(img_mask, sector_mask)

#     res1d, mapaxisinfo = calculate_1d(
#         ivqbins, radialrange, unit_ip, polarization, current_ai, img_data, d5i, method
#     )
#     if shared:
#         save_intcountshm_withlock(res1d)
#         return mapaxisinfo

#     return (
#         res1d.intensity,
#         mapaxisinfo[0],
#         [current_ai.mask, img_mask, sector_mask],
#         mapaxisinfo[1],
#     )


# def pyfai_2dmap_worker(
#     imageind,
#     d5i,
#     metadata,
#     current_ai,
#     newrot,
#     cfg,
#     log_queue,
#     logn=None,
#     shared=False,
# ) -> tuple:
#     """
#     calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
#     """

#     # global INTENSITY_ARRAY, COUNT_ARRAY
#     ind = imageind

#     mask = current_ai.mask
#     mapunits = setup_ip_oop_units(
#         cfg.unit_qip_name,
#         cfg.unit_qoop_name,
#         cfg.sample_orientation,
#     )
#     unit_ip = mapunits[0]  # shared_mem_setup.UNIT_IP
#     unit_oop = mapunits[1]  # shared_mem_setup.UNIT_OOP
#     method = ("no", "csr", "cython")
#     # alphacritical = cfg.alphacritical
#     setup = cfg.setup

#     qmapbins, fullranges, polarization = (
#         cfg.qmapbins,
#         cfg.fullranges,
#         cfg.polarization,
#     )
#     set_ai_rots(newrot, current_ai, setup)
#     img_data, img_mask = get_pyfai_image_data(setup, metadata, ind)
#     if current_ai.mask is None:
#         current_ai.mask = np.array(img_mask, copy=True)
#         mask = current_ai.mask
#     else:
#         np.copyto(mask, img_mask)

#     map2d, mapaxisinfo = calculate_2d_map(
#         current_ai,
#         img_data,
#         unit_ip,
#         unit_oop,
#         method,
#         d5i,
#         qmapbins,
#         fullranges,
#         polarization,
#     )
#     if shared:
#         save_intcountshm_withlock(map2d)
#         return mapaxisinfo

#     return (map2d[0], map2d[1], map2d[2], mapaxisinfo, current_ai.mask)
