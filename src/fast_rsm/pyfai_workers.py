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
    



#==========moving workers


def pyfai_move_ivsq_worker_new(experiment: Experiment, imageindices,
                           scan, process_config,queue=None,logn=None) -> None:
    """
    calculate 1d intensity vs q profile for moving detector scan using pyFAI

    """
    
    cfg = process_config

    
    time_logger=get_logger(LOGGER_DEBUG)
    if queue is not None:
        #print('found queue')
        qh = logging.handlers.QueueHandler(queue)
        root_logger = get_logger(LOGGER_DEBUG)
        root_logger.addHandler(qh)
        root_logger.setLevel(logging.DEBUG)
        time_logger=root_logger.getChild(f'child_{logn}')
        time_logger.debug(f'created logger for child_{logn}')
    time_logger.debug(do_time_check(f'start ivq worker {logn}'))
    
    #ais = []
    #img_data_list = []
    d5i_data=[]
    unit_tth = units.get_unit_fiber(
        cfg.unit_qip_name, sample_orientation=cfg.sample_orientation, incident_angle=0)
    unit_oop = units.get_unit_fiber(
        cfg.unit_qoop_name, sample_orientation=cfg.sample_orientation, incident_angle=0)
    
   
    fullresult=np.zeros(cfg.ivqbins)
    fullcounts=np.zeros(cfg.ivqbins)#
    time_logger.debug(do_time_check(f'start loop of image child_{logn}'))
    for i,ind in enumerate(imageindices):
        inc_angle,inc_angle_out=cfg.all_inc_angles[ind]
        # unit_oop.set_incident_angle(inc_angle_out)
        # unit_ip.set_incident_angle(inc_angle_out)
        gamdelval=cfg.gamdelvals[ind]
        current_ai=get_pyfai_ai(experiment,cfg.aistart, cfg.slitratios, cfg.alphacritical,inc_angle,gamdelval)
        d5i_data=cfg.d5i_full[ind]
        #current_ai.rot2=np.radians(34.8)

        #flat_img_data,dummy_ai=load_flat_test_image()
        # img_mask=dummy_ai.mask
        # current_ai.mask=dummy_ai.mask

        img_data,img_mask=get_pyfai_image_data(experiment,scan,ind)

        #DEBUG - section for creating normalise image of ones
        #img_data[img_data.astype(float)==0.0]=1
        #ones_img_data=np.divide(img_data,img_data,out=np.zeros(np.shape(img_data)),where=img_data.astype(float)>0.0)

        current_ai.mask=img_mask
        method=("no", "histogram", "cython")
#) 
#
        #single_result=current_ai.integrate1d(img_data,cfg.ivqbins,unit = cfg.unit_qip_name ,normalization_factor=d5i_data,correctSolidAngle=True, method=method,radial_range=(cfg.radialrange[0]-0.5, cfg.radialrange[1]+0.5))
        #outrange=(cfg.radialrange[0]-0.5, cfg.radialrange[1]+0.5)
        qranges=np.array([calcq(val,experiment.incident_wavelength) for val in cfg.fullranges])
        range_adjust=[-0.5,0.5]
        single_result=current_ai.integrate_fiber(img_data,  npt_ip=cfg.ivqbins, unit_ip=cfg.unit_qip_name, ip_range=qranges[0:2]+range_adjust,
                        npt_oop=cfg.ivqbins, unit_oop=cfg.unit_qoop_name,oop_range=qranges[2:]+range_adjust,
                        sample_orientation=cfg.sample_orientation,
                        normalization_factor=d5i_data,vertical_integration=True)
        #single_result=current_ai.integrate1d(img_data,cfg.ivqbins,unit = cfg.unit_qip_name ,method=method,radial_range=(cfg.radialrange[0], cfg.radialrange[1]))
        fullresult+=single_result.sum_signal
        fullcounts+=single_result.count
    time_logger.debug(do_time_check(f'stop loop of image child_{logn}'))
    return fullresult,fullcounts,single_result.radial,img_mask
    #return result1d.sum_signal,result1d.count,single_result.radial


def pyfai_move_qmap_worker_new(experiment: Experiment, imageindices,
                           scan, process_config,queue=None,logn=None) -> None:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI
    """
    cfg = process_config

    
    time_logger=get_logger(LOGGER_DEBUG)
    if queue is not None:
        #print('found queue')
        qh = logging.handlers.QueueHandler(queue)
        root_logger = get_logger(LOGGER_DEBUG)
        root_logger.addHandler(qh)
        root_logger.setLevel(logging.DEBUG)
        time_logger=root_logger.getChild(f'child_{logn}')
        time_logger.debug(f'created logger for child_{logn}')
    time_logger.debug(do_time_check(f'start ivq worker {logn}'))
    
    #ais = []
    #img_data_list = []
    d5i_data=[]
    unit_tth = units.get_unit_fiber(
        cfg.unit_qip_name, sample_orientation=cfg.sample_orientation, incident_angle=0)
    unit_oop = units.get_unit_fiber(
        cfg.unit_qoop_name, sample_orientation=cfg.sample_orientation, incident_angle=0)
    
    
    fullresult=np.zeros(cfg.qmapbins)
    fullcounts=np.zeros(cfg.qmapbins)#
    time_logger.debug(do_time_check(f'start loop of image child_{logn}'))
    for i,ind in enumerate(imageindices):
        inc_angle,inc_angle_out=cfg.all_inc_angles[ind]
        # unit_oop.set_incident_angle(inc_angle_out)
        # unit_ip.set_incident_angle(inc_angle_out)
        gamdelval=cfg.gamdelvals[ind]
        current_ai=get_pyfai_ai(experiment,cfg.aistart, cfg.slitratios, cfg.alphacritical,inc_angle,gamdelval)
        d5i_data=cfg.d5i_full[ind]


        img_data,img_mask=get_pyfai_image_data(experiment,scan,ind)
        current_ai.mask=img_mask
        method=("no", "histogram", "cython")
#)

        #single_result=current_ai.integrate1d(img_data,cfg.ivqbins,unit = cfg.unit_qip_name ,normalization_factor=d5i_data,correctSolidAngle=True, method=method,radial_range=(cfg.radialrange[0]-0.5, cfg.radialrange[1]+0.5))
        outrange=(cfg.radialrange[0]-0.5, cfg.radialrange[1]+0.5)

        single_result=current_ai.integrate2d_fiber(img_data,  npt_ip=cfg.qmapbins[0], unit_ip=cfg.unit_qip_name, ip_range=outrange, npt_oop=cfg.qmapbins[1], unit_oop=cfg.unit_qoop_name, oop_range=outrange,sample_orientation=cfg.sample_orientation, normalization_factor=d5i_data)
        #single_result=current_ai.integrate1d(img_data,cfg.ivqbins,unit = cfg.unit_qip_name ,method=method,radial_range=(cfg.radialrange[0], cfg.radialrange[1]))
        fullresult+=single_result.sum_signal
        fullcounts+=single_result.count
    time_logger.debug(do_time_check(f'stop loop of image child_{logn}'))
    return fullresult,fullcounts,single_result.radial,single_result.azimuthal,img_mask



def pyfai_move_ivsq_worker_old(experiment: Experiment, imageindices,
                           scan, process_config) -> None:
    """
    calculate 1d intensity vs q profile for moving detector scan using pyFAI

    """
    cfg = process_config
    do_time_check('start ivq worker')

    global INTENSITY_ARRAY, COUNT_ARRAY

    print("started move ivq worker")
    print(f"shape qi =  {cfg.shapeqi}")
    print(f"indices length= {len(imageindices)}")
    # , type_="pyFAI.integrator.fiber.FiberIntegrator")
    aistart = pyFAI.load(cfg.pyfaiponi)
    # 15-07-2025  fiber integrator not currently working with multigeomtery
    totaloutqi = np.zeros(cfg.shapeqi)
    totaloutcounts = np.zeros(cfg.shapeqi)

    unit_qip_name = "2th_deg"  # "qtot_A^-1"# "qip_A^-1"
    unit_qoop_name = "qoop_A^-1"
    if hasattr(scan.metadata.data_file.nx_entry,'d5i'):
        d5i_full=np.array(scan.metadata.data_file.nx_entry.d5i.data)
    else:
        d5i_full=np.ones(scan.metadata.data_file.scan_length)
    sample_orientation = 1
    groupnum = 25
    groups = [imageindices[i:i + groupnum]
              for i in range(0, len(imageindices), groupnum)]
    for m,group in enumerate(groups):
        do_time_check(f'start group {m}')
        ais = []
        img_data_list = []
        d5i_data=[]
        #do_time_check(f'start get components group {m}')
        for i in group:
            unit_tth_ip, unit_qoop, img_data, current_ai, ai_limits =\
             get_pyfai_components(experiment, i, sample_orientation, unit_qip_name,\
             unit_qoop_name, aistart, cfg.slitratios, cfg.alphacritical,\
              scan, [0, 1, 0, 1])
            d5i_data.append(d5i_full[i])
            img_data_list.append(img_data)
            ais.append(current_ai)
            if i==0:
                print(f"shape image 1 = {np.shape(img_data)}")
        
        #do_time_check(f'stop get components group {m}')
        #do_time_check(f'start multigeometry calculation group {m}')
        mg = MultiGeometry(ais, unit=unit_tth_ip,
                           wavelength=experiment.incident_wavelength,
                           radial_range=(
                               cfg.radialrange[0], cfg.radialrange[1]))
        method=("no", "histogram", "cython") #- still issue of tails
        #method = pyFAI.method_registry.IntegrationMethod.parse("full", dim=1)
        result1d = mg.integrate1d(img_data_list, cfg.ivqbins,method=method,normalization_factor=d5i_data)
        result_solid=mg.integrate1d([ai.solidAngleArray() for ai in ais], cfg.ivqbins,normalization_factor=d5i_data,correctSolidAngle=False)#,method=method)
        #do_time_check(f'stop multigeometry calculation group {m}')
        q_from_theta = [calcq(
            val, experiment.incident_wavelength) for val in result1d.radial]
        # theta_from_q= [calctheta(val, experiment.incident_wavelength) \
        # for val in result1d.radial]
        #np.divide(result1d.sum_signal,result_solid.intensity,where=result_solid.intensity.astype(float)!=0.0)
        totaloutqi[0] += result1d.sum_signal
        totaloutqi[1] = q_from_theta
        totaloutqi[2] = result1d.radial

        totaloutcounts[0] += result1d.count#[1 if val>0 else 0 for val in result1d.count]  # [int(val) for val in I>0]
        totaloutcounts[1] += result_solid.intensity
        totaloutcounts[2] += result_solid.sum_signal  # theta_from_q#
        do_time_check(f'end group {m}')
    # with lock:
    #     INTENSITY_ARRAY[0] += totaloutqi[0]
    #     INTENSITY_ARRAY[1:] = totaloutqi[1:]
    #     COUNT_ARRAY[0] += totaloutcounts[0]
    #     COUNT_ARRAY[1:] += totaloutcounts[1:]
    return totaloutqi, totaloutcounts


def pyfai_move_qmap_worker_old(experiment: Experiment, imageindices,
                           scan, process_config) -> None:
    """
    calculate 2d q_para Vs q_perp map for moving detector scan using pyFAI

    """
    global INTENSITY_ARRAY, COUNT_ARRAY
    cfg = process_config
    aistart = pyFAI.load(
        cfg.pyfaiponi,
        type_="pyFAI.integrator.fiber.FiberIntegrator")

    shapemap = cfg.shapeqpqp
    totalqpqpmap = np.zeros((shapemap[0], shapemap[1]))
    totalqpqpcounts = np.zeros((shapemap[0], shapemap[1]))
    unit_qip_name = "qip_A^-1"
    unit_qoop_name = "qoop_A^-1"

    sample_orientation = 1

    groupnum = 15

    groups = [imageindices[i:i + groupnum]
              for i in range(0, len(imageindices), groupnum)]
    for group in groups:
        ais = []
        img_data_list = []
        for i in group:
            unit_qip, unit_qoop, img_data, out_ai, ai_limits = \
                get_pyfai_components(experiment, i, sample_orientation,\
                unit_qip_name, unit_qoop_name, aistart, cfg.slitratios,\
                cfg.alphacritical, scan, cfg.qlimitsout)

            img_data_list.append(img_data)
            ais.append(out_ai)

        for current_n, current_ai in enumerate(ais):
            current_img = img_data_list[current_n]
            map2d = current_ai.integrate2d(current_img, cfg.qmapbins[0],
                                           cfg.qmapbins[1], unit=(
                                               unit_qip, unit_qoop),
                                           radial_range=(
                                               ai_limits[0], ai_limits[1]),
                                           azimuth_range=(
                                               ai_limits[2], ai_limits[3]),
                                           method=("no", "csr", "cython"))

            totalqpqpmap += map2d.sum_signal
            totalqpqpcounts += map2d.count

    mapaxisinfo = [map2d.azimuthal, map2d.radial, str(
        map2d.azimuthal_unit), str(map2d.radial_unit)]
    with lock:
        INTENSITY_ARRAY += totalqpqpmap
        COUNT_ARRAY += totalqpqpcounts.astype(dtype=np.int32)
    return mapaxisinfo


def pyfai_move_exitangles_worker_old(experiment: Experiment, imageindices, scan, process_config) -> None:
    """
    calculate exit angle map for moving detector scan using pyFAI

    """
    cfg = process_config
    global INTENSITY_ARRAY, COUNT_ARRAY
    aistart = pyFAI.load(
        cfg.pyfaiponi,
        type_="pyFAI.integrator.fiber.FiberIntegrator")

    shapemap = cfg.shapeexhexv
    totalexhexvmap = np.zeros((shapemap[0], shapemap[1]))
    totalexhexvcounts = np.zeros((shapemap[0], shapemap[1]))
    unit_qip_name = "exit_angle_horz_deg"
    unit_qoop_name = "exit_angle_vert_deg"
    sample_orientation = 1

    groupnum = 15
    groups = [imageindices[i:i + groupnum]
              for i in range(0, len(imageindices), groupnum)]
    for group in groups:
        ais = []
        img_data_list = []
        for i in group:
            unit_qip, unit_qoop, img_data, out_ai, ai_limits = \
             get_pyfai_components(
                experiment, i, sample_orientation, unit_qip_name,\
                unit_qoop_name, aistart, cfg.slitratios, cfg.alphacritical, \
                scan, cfg.anglimitsout)

            img_data_list.append(img_data)
            ais.append(out_ai)

        for current_n, current_ai in enumerate(ais):
            current_img = img_data_list[current_n]
            map2d = current_ai.integrate2d(current_img, shapemap[1], shapemap[0],
                                           unit=(unit_qip, unit_qoop),
                                           radial_range=(
                                               ai_limits[0], ai_limits[1]),
                                           azimuth_range=(
                                               ai_limits[2], ai_limits[3]),
                                           method=("no", "csr", "cython"))
            totalexhexvmap += map2d.sum_signal
            totalexhexvcounts += map2d.count

    mapaxisinfo = [map2d.azimuthal, map2d.radial, str(
        map2d.azimuthal_unit), str(map2d.radial_unit)]
    with lock:
        INTENSITY_ARRAY += totalexhexvmap
        COUNT_ARRAY += totalexhexvcounts
    return mapaxisinfo


#============static workers

def pyfai_stat_exitangles_worker(experiment: Experiment, imageindex, scan,\
                                  process_config: SimpleNamespace) -> None:
    """
    calculate exit angle map for static detector scan data using pyFAI Fiber integrator
    """
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    cfg = process_config
    index = imageindex
    aistart = pyFAI.load(
        cfg.pyfaiponi,
        type_="pyFAI.integrator.fiber.FiberIntegrator")

    sample_orientation = 1
    unit_qip_name = "exit_angle_horz_deg"
    unit_qoop_name = "exit_angle_vert_deg"

    unit_qip, unit_qoop, img_data, current_ai, ai_limits = get_pyfai_components(
        experiment, index, sample_orientation, unit_qip_name, unit_qoop_name,
        aistart, cfg.slitratios, cfg.alphacritical, scan, cfg.anglimits)

    map2d = current_ai.integrate2d(img_data, cfg.qmapbins[0], cfg.qmapbins[1], \
    unit=(unit_qip, unit_qoop), radial_range=(ai_limits[0], ai_limits[1]),\
     azimuth_range=(ai_limits[2], ai_limits[3]), method=("no", "csr", "cython"))
    mapaxisinfo = [map2d.azimuthal, map2d.radial, str(
        map2d.azimuthal_unit), str(map2d.radial_unit)]

    return map2d[0], map2d[1], map2d[2], mapaxisinfo


def pyfai_stat_qmap_worker(experiment, imageindex, scan,
                           process_config: SimpleNamespace) -> None:
    """
    calculate q_para Vs q_perp map for static detector scan data using pyFAI Fiber integrator
    """
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    cfg = process_config
    index = imageindex
    aistart = pyFAI.load(
        cfg.pyfaiponi,
        type_="pyFAI.integrator.fiber.FiberIntegrator")

    sample_orientation = 1

    unit_qip_name = "qip_A^-1"
    unit_qoop_name = "qoop_A^-1"

    unit_qip, unit_qoop, img_data, current_ai, ai_limits = get_pyfai_components(
        experiment, index, sample_orientation, unit_qip_name,
        unit_qoop_name, aistart, cfg.slitratios, cfg.alphacritical, scan, cfg.qlimits)

    map2d = current_ai.integrate2d(img_data, cfg.qmapbins[0], cfg.qmapbins[1],\
        unit=(unit_qip, unit_qoop), radial_range=(ai_limits[0], ai_limits[1]),\
    azimuth_range=(ai_limits[2], ai_limits[3]), method=("no", "csr", "cython"))
    mapaxisinfo = [map2d.azimuthal, map2d.radial, str(
        map2d.azimuthal_unit), str(map2d.radial_unit)]
    return map2d[0], map2d[1], map2d[2], mapaxisinfo


def pyfai_stat_ivsq_worker(experiment: Experiment, imageindex, scan,
                           process_config: SimpleNamespace) -> None:
    """
    calculate Intensity Vs Q profile for static detector scan data using pyFAI Fiber integrator
    """
    cfg = process_config
    index = imageindex
    # , type_="pyFAI.integrator.fiber.FiberIntegrator")
    aistart = pyFAI.load(cfg.pyfaiponi)
    sample_orientation = 1

    unit_qip_name = "qtot_A^-1"
    unit_qoop_name = "qoop_A^-1"
    unit_q_tot, unit_qoop, img_data, current_ai, ai_limits = get_pyfai_components(
        experiment, index, sample_orientation, unit_qip_name,
        unit_qoop_name, aistart, cfg.slitratios, cfg.alphacritical, scan, [0, 1, 0, 1])

    tth, intensity = current_ai.integrate1d_ng(img_data,
                                          cfg.ivqbins,
                                          unit="2th_deg", polarization_factor=1,\
                                            radial_range=(
                               cfg.radialrange[0], cfg.radialrange[1]))
    qvals = [calcq(tthval, experiment.incident_wavelength)
             for tthval in tth]

    return intensity, tth, qvals

