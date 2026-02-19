import sys
import copy
from types import SimpleNamespace

import logging
import logging.handlers
from logging.handlers import QueueHandler
from pyFAI.multi_geometry import MultiGeometry
from fast_rsm.experiment import Experiment,calctheta,calcq,gamdel2rots
import pyFAI
from pyFAI import units
import numpy as np
import pyFAI.detectors
import pyFAI.calibrant

from fast_rsm.angle_pixel_q import calcq,gamdel2rots
from fast_rsm.logging_config import get_debug_logger,listener_process,get_logger,do_time_check


LOGGER_DEBUG = 'fastrsm_debug'
LOGGER_ERROR = 'fastrsm_error'
debug_logger = get_debug_logger()
sys.stdout.reconfigure(line_buffering=True)


#==============common functions

def get_pyfai_image_data(experiment: Experiment,scan,i):

    outimage=scan.load_image(i)
    mask=np.isnan(outimage.data).astype(bool)
    if experiment.setup == 'vertical':
        return np.rot90(outimage.data, -1),np.rot90(mask,-1)
    
    return np.array(outimage.data),mask

def get_pyfai_ai(experiment: Experiment,aistart, slitratios, alphacritical,inc_angle,gamdelval):
    """
    get components need for mapping with pyFAI
    """

    gamval,delval=gamdelval
    if (-np.degrees(inc_angle) >
            alphacritical) & (experiment.setup == 'DCD'):
        # if above critical angle, account for direct beam adding to delta
        rots = gamdel2rots(gamval, delval + np.degrees(-inc_angle))
    else:
        rots =  gamdel2rots(gamval, delval+ np.degrees(-inc_angle))

    out_ai = copy.deepcopy(aistart)
    out_ai.rot1, out_ai.rot2, out_ai.rot3 = rots
    

    if experiment.setup == 'vertical':
        out_ai.rot1 = rots[1]
        out_ai.rot2 = -rots[0]

    if slitratios[0] is not None:
        out_ai.pixel1 *= slitratios[0]
        out_ai.poni1 *= slitratios[0]

    if slitratios[1] is not None:
        out_ai.pixel2 *= slitratios[1]
        out_ai.poni2 *= slitratios[1]
    return  out_ai


def get_pyfai_limits(limits_in):
    radial_limits = (limits_in[0] * (1.0 + (0.05 * -(np.sign(limits_in[0])))),
                     limits_in[1] * (1.0 + (0.05 * (np.sign(limits_in[1])))))
    azimuthal_limits = (limits_in[2] * (1.0 + (0.05 * -(np.sign(limits_in[2])))),
                        limits_in[3] * (1.0 + (0.05 * (np.sign(limits_in[3])))))
    return [radial_limits[0], radial_limits[1],
                  azimuthal_limits[0], azimuthal_limits[1]]
    
def setup_stat_worker(experiment: Experiment,cfg,scan, imageindex):
    inc_angle,inc_angle_out=cfg.all_inc_angles[imageindex]
    gamdelval=cfg.gamdelvals[imageindex]
    current_ai=get_pyfai_ai(experiment,cfg.aistart, cfg.slitratios, cfg.alphacritical,inc_angle,gamdelval)
    d5i_data=cfg.d5i_full[imageindex]
    img_data,img_mask=get_pyfai_image_data(experiment,scan,imageindex)
    method=("no", "histogram", "cython")
    return current_ai,img_data,img_mask,method,inc_angle_out,d5i_data

def setup_ip_oop_units(cfg,inc_angle_out=0):
    unit_ip = units.get_unit_fiber(
        cfg.unit_qip_name, sample_orientation=cfg.sample_orientation, incident_angle=inc_angle_out)
    unit_oop = units.get_unit_fiber(
        cfg.unit_qoop_name, sample_orientation=cfg.sample_orientation, incident_angle=inc_angle_out)
    return unit_ip,unit_oop

def calculate_2d_map(cfg,ai,img_data,unit_ip,unit_oop,method,d5i):
    map2d = ai.integrate2d(img_data, cfg.qmapbins[0], cfg.qmapbins[1], \
    unit=(unit_ip, unit_oop), radial_range=(cfg.fullranges[0], cfg.fullranges[1]),\
     azimuth_range=(cfg.fullranges[2], cfg.fullranges[3]), method=method, polarization_factor=cfg.polarization,\
        normalization_factor=d5i)
    mapaxisinfo = [map2d.azimuthal, map2d.radial, str(
        map2d.azimuthal_unit), str(map2d.radial_unit)]
    return map2d,mapaxisinfo

def calculate_1d(cfg,ai,img_data,norm_data,method):
    result1d=ai.integrate1d(img_data,cfg.ivqbins,unit = cfg.unit_qip_name ,normalization_factor=norm_data,\
                                             correctSolidAngle=True, method=method,radial_range=(cfg.radialrange[0]-0.5, cfg.radialrange[1]+0.5),\
                                             polarization_factor=cfg.polarization)
    return result1d

def get_sector_mask(ai,shape,sector_ranges):
    if sector_ranges is None:
        return np.zeros(shape)
    chivals=ai.chiArray(shape)
    chi_min,chi_max=ai.normalize_azimuth_range(sector_ranges)
    return np.logical_or(chivals>chi_max,chivals<chi_min)

def get_time_logger(queue,logn):
    time_logger=get_logger(LOGGER_DEBUG)
    if queue is not None:
        #print('found queue')
        qh = logging.handlers.QueueHandler(queue)
        root_logger = get_logger(LOGGER_DEBUG)
        root_logger.addHandler(qh)
        root_logger.setLevel(logging.DEBUG)
        time_logger=root_logger.getChild(f'child_{logn}')
        time_logger.debug(f'created logger for child_{logn}')
    
    return time_logger

#==========moving workers


def pyfai_move_ivsq_worker_new(experiment: Experiment, imageindices,
                           scan, process_config,queue=None,logn=None) -> None:
    """
    calculate 1d intensity vs q profile for moving detector scan using pyFAI

    """
    
    cfg = process_config
    d5i_data=[]
    unit_tth,unit_oop=setup_ip_oop_units(cfg)
    time_logger=get_time_logger(queue,logn)
    time_logger.debug(do_time_check(f'start ivq worker {logn}'))
    fullresult=np.zeros(cfg.ivqbins)
    fullcounts=np.zeros(cfg.ivqbins)#

    time_logger.debug(do_time_check(f'start loop of child_{logn}'))
    for i,ind in enumerate(imageindices):
        current_ai,img_data,img_mask,method,inc_angle_out,d5i_data=setup_stat_worker(experiment,cfg,scan,ind)
        current_ai.mask=img_mask
        unit_tth.incident_angles=inc_angle_out
        unit_oop.incident_angle=inc_angle_out
        single_result=calculate_1d(cfg,current_ai,img_data,d5i_data,method)

        fullresult+=single_result.sum_signal
        fullcounts+=single_result.count
    time_logger.debug(do_time_check(f'stop loop of child_{logn}'))
    return fullresult,fullcounts,single_result.radial,img_mask

def pyfai_move_qmap_worker_new(experiment: Experiment, imageindices,
                           scan, process_config,queue=None,logn=None) -> None:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
    """
    cfg = process_config
    
    d5i_data=[]
    inc_angle=np.radians(experiment.incident_angle)
    unit_ip,unit_oop=setup_ip_oop_units(cfg)
    
    time_logger=get_time_logger(queue,logn)
    time_logger.debug(do_time_check(f'start qmap worker {logn}'))
    time_logger.debug(do_time_check(f'start loop of image child_{logn}'))
    
    fullresult=np.zeros((cfg.qmapbins[1],cfg.qmapbins[0]))
    fullcounts=np.zeros((cfg.qmapbins[1],cfg.qmapbins[0]))#
    
    for i,ind in enumerate(imageindices):
        current_ai,img_data,img_mask,method,inc_angle_out,d5i_data=setup_stat_worker(experiment,cfg,scan,ind)
        current_ai.mask=img_mask
        method=("no", "csr", "cython")
        unit_ip.incident_angle=inc_angle_out
        unit_oop.incident_angle=inc_angle_out
        single_result,axisinfo=calculate_2d_map(cfg,current_ai,img_data,unit_ip,unit_oop,method,d5i_data)

        fullresult+=single_result.sum_signal
        fullcounts+=single_result.count
    time_logger.debug(do_time_check(f'stop loop of image child_{logn}'))
    return fullresult,fullcounts,axisinfo,img_mask

def pyfai_move_exitangles_worker_new(experiment: Experiment, imageindices,
                           scan, process_config,queue=None,logn=None) -> None:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
    """
    cfg = process_config
    time_logger=get_time_logger(queue,logn)
    d5i_data=[]
    unit_ip,unit_oop=setup_ip_oop_units(cfg)
    
    
    fullresult=np.zeros((cfg.qmapbins[1],cfg.qmapbins[0]))
    fullcounts=np.zeros((cfg.qmapbins[1],cfg.qmapbins[0]))#
    #time_logger.debug(do_time_check(f'start loop of child_{logn}'))
    for i,ind in enumerate(imageindices):
        current_ai,img_data,img_mask,method,inc_angle_out,d5i_data=setup_stat_worker(experiment,cfg,scan,ind)
        unit_ip.incident_angle=inc_angle_out
        unit_oop.incident_angle=inc_angle_out
        current_ai.mask=img_mask
        single_result,axisinfo=calculate_2d_map(cfg,current_ai,img_data,unit_ip,unit_oop,method,d5i_data)

        fullresult+=single_result.sum_signal
        fullcounts+=single_result.count
    #time_logger.debug(do_time_check(f'stop loop of child_{logn}'))
    return fullresult,fullcounts,axisinfo,img_mask

#============static workers

def pyfai_stat_ivsq_worker_new(experiment: Experiment, imageindex, scan,
                           process_config: SimpleNamespace) -> None:
    """
    calculate Intensity Vs Q profile for static detector scan data using pyFAI Fiber integrator
    """
    cfg = process_config
    current_ai,img_data,img_mask,method,inc_angle_out,d5i_data=setup_stat_worker(experiment,cfg,scan,imageindex)
    sector_mask=get_sector_mask(current_ai,img_data.shape,cfg.azimuthal_sector)
    current_ai.mask=np.logical_or(img_mask,sector_mask)
    
    tth,intensity = calculate_1d(cfg,current_ai,img_data,d5i_data,method)
    mask_list=[current_ai.mask,img_mask,sector_mask]
    return intensity,tth,mask_list

def pyfai_stat_exitangles_worker_new(experiment: Experiment, imageindex, scan,\
                                  process_config: SimpleNamespace) -> None:
    
    """
    calculate exit angle map for static detector scan data using pyFAI Fiber integrator
    """
    cfg = process_config
    current_ai,img_data,img_mask,method,inc_angle_out,d5i_data=setup_stat_worker(experiment,cfg,scan,imageindex)
    current_ai.mask=img_mask
    unit_ip,unit_oop=setup_ip_oop_units(cfg,inc_angle_out)
    map2d,mapaxisinfo=calculate_2d_map(cfg,current_ai,img_data,unit_ip,unit_oop,method,d5i_data)

    return map2d[0], map2d[1], map2d[2], mapaxisinfo,img_mask

def pyfai_stat_qmap_worker_new(experiment: Experiment, imageindex, scan,\
                                  process_config: SimpleNamespace) -> None:
    
    """
    calculate exit angle map for static detector scan data using pyFAI Fiber integrator
    """
    cfg = process_config
    current_ai,img_data,img_mask,method,inc_angle_out,d5i_data=setup_stat_worker(experiment,cfg,scan,imageindex)
    current_ai.mask=img_mask
    unit_ip,unit_oop=setup_ip_oop_units(cfg,inc_angle_out)
    method=("no", "csr", "cython")
    map2d,mapaxisinfo=calculate_2d_map(cfg,current_ai,img_data,unit_ip,unit_oop,method,d5i_data)

    return map2d[0], map2d[1], map2d[2], mapaxisinfo,img_mask
