import copy
import logging
import logging.handlers
import sys
from types import SimpleNamespace

import numpy as np
from pyFAI import units

import fast_rsm.shared_mem_setup as shared_mem_setup
from fast_rsm.angle_pixel_q import gamdel2rots
from fast_rsm.experiment import Experiment
from fast_rsm.image import Image
from fast_rsm.logging_config import get_debug_logger, get_logger

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
    mask = np.isnan(outimage.data).astype(bool)
    if setup == "vertical":
        return np.rot90(outimage.data, -1), np.rot90(mask, -1)

    return np.array(outimage.data), mask


def set_ai_rots(inc_angle, gamdelval, current_ai, setup, alphacritical):
    """
    get components need for mapping with pyFAI
    """
    gamval, delval = gamdelval
    if (-np.degrees(inc_angle) > alphacritical) & (setup == "DCD"):
        # if above critical angle, account for direct beam adding to delta
        rots = gamdel2rots(gamval, delval + np.degrees(-inc_angle))
    else:
        rots = gamdel2rots(gamval, delval)

    current_ai.rot1, current_ai.rot2, current_ai.rot3 = rots

    if setup == "vertical":
        current_ai.rot1 = rots[1]
        current_ai.rot2 = -rots[0]


def setup_stat_worker(
    metadata,
    imageindex,
    current_ai,
    setup,
    all_inc_angles,
    gamdelvals,
    d5i_full,
    alphacritical,
):
    inc_angle, inc_angle_out = all_inc_angles[imageindex]
    gamdelval = gamdelvals[imageindex]
    set_ai_rots(inc_angle, gamdelval, current_ai, setup, alphacritical)
    d5i_data = d5i_full[imageindex]
    img_data, img_mask = get_pyfai_image_data(setup, metadata, imageindex)

    return img_data, img_mask, inc_angle_out, d5i_data


# def setup_stat_worker(experiment: Experiment, cfg, scan, imageindex):
#     inc_angle, inc_angle_out = cfg.all_inc_angles[imageindex]
#     gamdelval = cfg.gamdelvals[imageindex]
#     current_ai = get_pyfai_ai(
#         experiment, cfg.aistart, cfg.slitratios, cfg.alphacritical, inc_angle, gamdelval
#     )
#     d5i_data = cfg.d5i_full[imageindex]
#     img_data, img_mask = get_pyfai_image_data(experiment, scan, imageindex)
#     method = ("no", "histogram", "cython")
#     return current_ai, img_data, img_mask, method, inc_angle_out, d5i_data


def setup_ip_oop_units(cfg, inc_angle_out=0):
    unit_ip = units.get_unit_fiber(
        cfg.unit_qip_name,
        sample_orientation=cfg.sample_orientation,
        incident_angle=inc_angle_out,
    )
    unit_oop = units.get_unit_fiber(
        cfg.unit_qoop_name,
        sample_orientation=cfg.sample_orientation,
        incident_angle=inc_angle_out,
    )
    return unit_ip, unit_oop


def calculate_2d_map(
    ai, img_data, unit_ip, unit_oop, method, d5i, qmapbins, fullranges, polarization
):
    map2d = ai.integrate2d(
        img_data,
        qmapbins[0],
        qmapbins[1],
        unit=(unit_ip, unit_oop),
        radial_range=(fullranges[0], fullranges[1]),
        azimuth_range=(fullranges[2], fullranges[3]),
        method=method,
        polarization_factor=polarization,
        normalization_factor=d5i,
    )
    mapaxisinfo = [
        map2d.azimuthal,
        map2d.radial,
        str(map2d.azimuthal_unit),
        str(map2d.radial_unit),
    ]
    return map2d, mapaxisinfo


def calculate_1d(cfg, ai, img_data, norm_data, method):
    result1d = ai.integrate1d(
        img_data,
        cfg.ivqbins,
        unit=cfg.unit_qip_name,
        normalization_factor=norm_data,
        correctSolidAngle=True,
        method=method,
        radial_range=(cfg.radialrange[0] - 0.5, cfg.radialrange[1] + 0.5),
        polarization_factor=cfg.polarization,
    )
    return result1d


def get_sector_mask(ai, shape, sector_ranges):
    if sector_ranges is None:
        return np.zeros(shape)
    chivals = ai.chiArray(shape)
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


# -----------------------------
# Dispatcher
# -----------------------------
def worker_unpack(args):
    function_map = {
        "move_ivq": pyfai_move_ivsq_worker_new,
        "move_qmap": pyfai_move_qmap_worker_minimum,  # pyfai_move_qmap_worker_new,
        "move_exit": pyfai_move_exitangles_worker_new,
        "static_ivq": pyfai_stat_ivsq_worker_new,
        "static_exit": pyfai_stat_exitangles_worker_new,
        "static_qmap": pyfai_stat_qmap_worker_new,
    }
    worker_type = args[0]
    worker_function = function_map[worker_type]
    worker_args = args[1:]
    # top-level adapter to avoid lambda pickling issues
    return worker_function(*worker_args)


# ==========moving workers
# --------------------------------------------
# ======= incident angle message=======
# decided to remove incident from units for now, as a sample
# incident angle for sxrd does not affect the exit angle of interest, just the specific set of crystallites that are in diffraction condition
# DCD is more complicated and will check later - currently offsets delta if incident angle is above critical angle
# --------------------------------------------------


def pyfai_move_ivsq_worker_new(
    experiment: Experiment, imageindices, scan, process_config, queue=None, logn=None
) -> None:
    """
    calculate 1d intensity vs q profile for moving detector scan using pyFAI

    """

    cfg = process_config
    d5i_data = []
    unit_tth, unit_oop = setup_ip_oop_units(cfg)
    # time_logger = get_time_logger(queue, logn)
    # time_logger.debug(do_time_check(f"start ivq worker {logn}"))
    fullresult = np.zeros(cfg.ivqbins)
    fullcounts = np.zeros(cfg.ivqbins)  #

    # time_logger.debug(do_time_check(f"start loop of child_{logn}"))
    for i, ind in enumerate(imageindices):
        current_ai, img_data, img_mask, method, inc_angle_out, d5i_data = (
            setup_stat_worker(experiment, cfg, scan, ind)
        )
        current_ai.mask = img_mask

        # =============================================
        # see incident angle message at top of worker section line 80
        # =============================================
        # unit_tth.incident_angle = inc_angle_out
        # unit_oop.incident_angle = inc_angle_out

        single_result = calculate_1d(cfg, current_ai, img_data, d5i_data, method)

        fullresult += single_result.sum_signal
        fullcounts += single_result.count
    # time_logger.debug(do_time_check(f"stop loop of child_{logn}"))
    return fullresult, fullcounts, single_result.radial, img_mask


def pyfai_move_qmap_worker_minimum(
    imageindices, metadata, unique_scan_vals, logn=None
) -> None:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
    """

    logger = get_debug_logger()
    cfg = shared_mem_setup.CFG
    # cfg.unit_ip, cfg.unit_oop = setup_ip_oop_units(
    #     cfg.unit_qip_name, cfg.unit_qoop_name, cfg.sample_orientation
    # )
    # logger.debug("loaded cfg in work minimum")
    cfg.d5i_full, cfg.all_inc_angles, cfg.gamdelvals = unique_scan_vals
    d5i_data = []
    # inc_angle = np.radians(experiment.incident_angle)

    # time_logger = get_time_logger(queue, logn)
    # time_logger.debug(do_time_check(f"start qmap worker {logn}"))
    # time_logger.debug(do_time_check(f"start loop of image child_{logn}"))

    fullresult = np.zeros((cfg.qmapbins[1], cfg.qmapbins[0]), dtype=np.float32)
    fullcounts = np.zeros((cfg.qmapbins[1], cfg.qmapbins[0]), dtype=np.float32)
    # #

    current_ai = setup_start_ai(cfg.aistart, cfg.slitratios)

    mask = current_ai.mask

    unit_ip = shared_mem_setup.UNIT_IP
    unit_oop = shared_mem_setup.UNIT_OOP
    method = ("no", "csr", "cython")
    alphacritical = cfg.alphacritical
    setup = cfg.setup
    all_inc_angles = cfg.all_inc_angles
    gamdelvals = cfg.gamdelvals
    d5i_full = cfg.d5i_full
    qmapbins, fullranges, polarization = (
        cfg.qmapbins,
        cfg.fullranges,
        cfg.polarization,
    )
    logger.debug(
        f"reached start of loop for batch {logn}: images {imageindices[0]}-{imageindices[-1]} "
    )
    for i, ind in enumerate(imageindices):
        img_data, img_mask, inc_angle_out, d5i_data = setup_stat_worker(
            metadata,
            ind,
            current_ai,
            setup,
            all_inc_angles,
            gamdelvals,
            d5i_full,
            alphacritical,
        )

        if mask is None:
            current_ai.mask = np.array(img_mask, copy=True)
            mask = current_ai.mask
        else:
            np.copyto(mask, img_mask)

        # method = ("no", "histogram", "cython")

        # =============================================
        # see incident angle message at top of worker section line 80
        # =============================================
        # unit_tth.incident_angle = inc_angle_out
        # unit_oop.incident_angle = inc_angle_out
        # logger.debug(f"start map calculation {ind}")
        single_result, axisinfo = calculate_2d_map(
            current_ai,
            img_data,
            unit_ip,
            unit_oop,
            method,
            d5i_data,
            qmapbins,
            fullranges,
            polarization,
        )
        if ind % 10 == 0:
            logger.debug(f"ended map calculation {ind}")

        fullresult += single_result.sum_signal
        fullcounts += single_result.count
    logger.debug(
        f"reached end of loop for batch {logn}: images {imageindices[0]}-{imageindices[-1]} "
    )

    # time_logger.debug(do_time_check(f"stop loop of image child_{logn}"))
    return fullresult, fullcounts, axisinfo, img_mask


def pyfai_move_qmap_worker_new(
    experiment: Experiment, imageindices, scan, process_config, queue=None, logn=None
) -> None:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
    """
    logger = get_debug_logger()
    cfg = process_config

    d5i_data = []
    inc_angle = np.radians(experiment.incident_angle)
    unit_ip, unit_oop = setup_ip_oop_units(cfg)
    cfg.current_ai = setup_start_ai(cfg.aistart, cfg.slitratios)
    # time_logger = get_time_logger(queue, logn)
    # time_logger.debug(do_time_check(f"start qmap worker {logn}"))
    # time_logger.debug(do_time_check(f"start loop of image child_{logn}"))

    fullresult = np.zeros((cfg.qmapbins[1], cfg.qmapbins[0]))
    fullcounts = np.zeros((cfg.qmapbins[1], cfg.qmapbins[0]))  #
    logger.debug(f"starting batch {logn}: {imageindices[0]}-{imageindices[-1]}")
    for i, ind in enumerate(imageindices):
        img_data, img_mask, inc_angle_out, d5i_data = setup_stat_worker(
            experiment, cfg, scan, ind
        )
        cfg.current_ai.mask = img_mask
        method = ("no", "csr", "cython")

        # =============================================
        # see incident angle message at top of worker section line 80
        # =============================================
        # unit_tth.incident_angle = inc_angle_out
        # unit_oop.incident_angle = inc_angle_out

        single_result, axisinfo = calculate_2d_map(
            cfg, cfg.current_ai, img_data, unit_ip, unit_oop, method, d5i_data
        )

        fullresult += single_result.sum_signal
        fullcounts += single_result.count
    logger.debug(f"finished batch {imageindices[0]}-{imageindices[-1]}")

    # time_logger.debug(do_time_check(f"stop loop of image child_{logn}"))
    return fullresult, fullcounts, axisinfo, img_mask


def pyfai_move_exitangles_worker_new(
    experiment: Experiment, imageindices, scan, process_config, queue=None, logn=None
) -> None:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
    """
    cfg = process_config
    # time_logger = get_time_logger(queue, logn)
    d5i_data = []
    unit_ip, unit_oop = setup_ip_oop_units(cfg)

    fullresult = np.zeros((cfg.qmapbins[1], cfg.qmapbins[0]))
    fullcounts = np.zeros((cfg.qmapbins[1], cfg.qmapbins[0]))  #
    # time_logger.debug(do_time_check(f'start loop of child_{logn}'))
    for i, ind in enumerate(imageindices):
        current_ai, img_data, img_mask, method, inc_angle_out, d5i_data = (
            setup_stat_worker(experiment, cfg, scan, ind)
        )

        # =============================================
        # see incident angle message at top of worker section line 80
        # =============================================
        # unit_tth.incident_angle = inc_angle_out
        # unit_oop.incident_angle = inc_angle_out

        current_ai.mask = img_mask
        single_result, axisinfo = calculate_2d_map(
            cfg, current_ai, img_data, unit_ip, unit_oop, method, d5i_data
        )

        fullresult += single_result.sum_signal
        fullcounts += single_result.count
    # time_logger.debug(do_time_check(f'stop loop of child_{logn}'))
    return fullresult, fullcounts, axisinfo, img_mask


# ============static workers


def pyfai_stat_ivsq_worker_new(
    experiment: Experiment, imageindex, scan, process_config: SimpleNamespace
) -> None:
    """
    calculate Intensity Vs Q profile for static detector scan data using pyFAI Fiber integrator
    """
    cfg = process_config
    current_ai, img_data, img_mask, method, inc_angle_out, d5i_data = setup_stat_worker(
        experiment, cfg, scan, imageindex
    )
    sector_mask = get_sector_mask(current_ai, img_data.shape, cfg.azimuthal_sector)
    current_ai.mask = np.logical_or(img_mask, sector_mask)

    tth, intensity = calculate_1d(cfg, current_ai, img_data, d5i_data, method)
    mask_list = [current_ai.mask, img_mask, sector_mask]
    return intensity, tth, mask_list


def pyfai_stat_exitangles_worker_new(
    experiment: Experiment, imageindex, scan, process_config: SimpleNamespace
) -> None:
    """
    calculate exit angle map for static detector scan data using pyFAI Fiber integrator
    """
    cfg = process_config
    current_ai, img_data, img_mask, method, inc_angle_out, d5i_data = setup_stat_worker(
        experiment, cfg, scan, imageindex
    )
    current_ai.mask = img_mask
    unit_ip, unit_oop = setup_ip_oop_units(cfg, inc_angle_out)
    map2d, mapaxisinfo = calculate_2d_map(
        cfg, current_ai, img_data, unit_ip, unit_oop, method, d5i_data
    )

    return map2d[0], map2d[1], map2d[2], mapaxisinfo, img_mask


def pyfai_stat_qmap_worker_new(
    experiment: Experiment, imageindex, scan, process_config: SimpleNamespace
) -> None:
    """
    calculate exit angle map for static detector scan data using pyFAI Fiber integrator
    """
    cfg = process_config
    current_ai, img_data, img_mask, method, inc_angle_out, d5i_data = setup_stat_worker(
        experiment, cfg, scan, imageindex
    )
    current_ai.mask = img_mask
    unit_ip, unit_oop = setup_ip_oop_units(cfg, inc_angle_out)
    method = ("no", "csr", "cython")
    map2d, mapaxisinfo = calculate_2d_map(
        cfg, current_ai, img_data, unit_ip, unit_oop, method, d5i_data
    )

    return map2d[0], map2d[1], map2d[2], mapaxisinfo, img_mask
