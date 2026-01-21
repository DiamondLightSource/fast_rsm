"""
Module for the functions used to interface with pyFAI package
"""
import os,sys

from types import SimpleNamespace

from datetime import datetime
from time import time
from multiprocessing import current_process, Lock, Pool, get_context, Manager,Process #Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from typing import  List
import copy
import yaml
from pyFAI.multi_geometry import MultiGeometry
import pyFAI
from pyFAI import units
import numpy as np
import pyFAI.detectors

import pyFAI.calibrant
from fast_rsm.rsm_metadata import RSMMetadata
from fast_rsm.scan import Scan, chunk, check_shared_memory
from fast_rsm.experiment import Experiment, gamdel2rots,calctheta,calcq,do_savedats,do_savetiffs
from fast_rsm.logging_config import get_debug_logger,listener_process,get_logger,do_time_check

from fast_rsm.pyfai_workers import pyfai_move_ivsq_worker_old,pyfai_move_qmap_worker_old, pyfai_move_qmap_worker_new,pyfai_move_ivsq_worker_new, pyfai_move_exitangles_worker_old,pyfai_stat_exitangles_worker,pyfai_stat_ivsq_worker,pyfai_stat_qmap_worker

from fast_rsm.angle_pixel_q import calcq,calctheta,gamdel2rots

LOGGER_DEBUG = 'fastrsm_debug'
LOGGER_ERROR = 'fastrsm_error'

# def init_worker_logger(log_queue, level=logging.INFO):
#     """
#     Runs in EACH worker process when the pool starts.
#     Attaches a QueueHandler to    Attaches a QueueHandler to the named loggers so worker logs go into the parent's queue.
#     """
#     for name in (LOGGER_DEBUG):
#         lg = logging.getLogger(name)
#         lg.setLevel(level)
#         lg.handlers[:] = []               # avoid duplicated handlers in workers
#         lg.addHandler(QueueHandler(log_queue))


debug_logger = get_debug_logger()
sys.stdout.reconfigure(line_buffering=True)




# ----------------------------
# Tuning: set BLAS/OpenMP threads to 1 to avoid oversubscription
# (important when using multiprocessing)
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "1")


# ====general functions



def find_bad_image_paths(scan: Scan):
    badpaths=[]
    for num,end in enumerate(scan.metadata.data_file.raw_image_paths):
        if not end.endswith('.tif'):
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
    ponioutpath = fr'{outpath}/fast_rsm_{datetime_str}.poni'
    with open(ponioutpath, 'w', encoding='utf-8') as f:
        f.write('# PONI file created by fast_rsm\n#\n')
        f.write('poni_version: 2\n')
        f.write('Detector: Detector\n')
        f.write('Detector_config: {"pixel1":')
        pixel_line = (
            f'{experiment.pixel_size}, '
            f'"pixel2": {experiment.pixel_size}, '
            f'"max_shape": [{image2dshape[0]}, {image2dshape[1]}]'
        )
        f.write(pixel_line)
        f.write('}\n')
        f.write(f'Distance: {experiment.detector_distance}\n')
        if beam_centre == 0:
            poni1 = (image2dshape[0] - offset) * experiment.pixel_size
            poni2 = image2dshape[1] * experiment.pixel_size
        elif (offset == 0) & (experiment.setup != 'vertical'):
            poni1 = (beam_centre[0]) * experiment.pixel_size
            poni2 = beam_centre[1] * experiment.pixel_size
        else:  # (offset == 0) & (experiment.setup == 'vertical'):
            poni1 = beam_centre[1] * experiment.pixel_size
            poni2 = (image2dshape[0] - beam_centre[0]) * experiment.pixel_size

        f.write(f'Poni1: {poni1}\n')
        f.write(f'Poni2: {poni2}\n')
        f.write('Rot1: 0.0\n')
        f.write('Rot2: 0.0\n')
        f.write('Rot3: 0.0\n')
        f.write(f'Wavelength: {experiment.incident_wavelength}')
    return ponioutpath

def get_full_indices( scan, process_config: SimpleNamespace):
    cfg = process_config
    fullrange = np.arange(0, cfg.scanlength, cfg.scalegamma)
    selectedindices = [
            n for n in fullrange if n not in scan.skip_images]
    return selectedindices   

def get_input_args(experiment, scan, process_config: SimpleNamespace):
    """
    create the input arguments for processing depending on process
    configuration
    """
    cfg = process_config
    fullrange = np.arange(0, cfg.scanlength, cfg.scalegamma)
    selectedindices = [
            n for n in fullrange if n not in scan.skip_images]
    if cfg.multi:
        inputindices = chunk(selectedindices, cfg.num_threads)
    else:
        inputindices = selectedindices

    input_args = [[experiment, indices, scan, cfg]
                  for indices in inputindices]
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

    qstep = round(calcq(1.00, experiment.incident_wavelength) -
                  calcq(1.01, experiment.incident_wavelength), 4)
    binshor = abs(round(((qlimits[1] - qlimits[0]) / qstep) * 1.05))
    binsver = abs(round(((qlimits[3] - qlimits[2]) / qstep) * 1.05))
    return (binshor, binsver)

def get_corner_thetas(process_config: SimpleNamespace):
    """
    calculate theta angles given inplane and out-of-plane angles to the detector
    corners. 
    """
    cfg = process_config
    corner_indexes=[[0,2],[0,3],[1,2],[1,3]]
    cfg.fullranges90=[(val,0) if val <=90 else (val-90,90) for val in np.abs(cfg.fullranges)]

    corner_items=[[cfg.fullranges90[ind][0] for ind in pair] for pair in corner_indexes]
    corner_values=np.radians(corner_items)
    corner_diagonal_angles=np.degrees(np.arctan([np.sqrt(np.tan(cv[0])**2+ np.tan(cv[1])**2) \
                            for cv in corner_values]))
    absranges = [np.abs(dval)+cfg.fullranges90[i][1] for i,dval in enumerate(corner_diagonal_angles)]
    radmax = np.max(absranges)
    return absranges,radmax



def pyfai_init_worker(l, shm_intensities_name, shm_counts_name, shmshape):
    """
    intialiser for pyfai mappings
    """
    global lock
    global SHM_INTENSITY
    global INTENSITY_ARRAY
    global SHM_COUNT
    global COUNT_ARRAY
    SHM_INTENSITY = SharedMemory(name=shm_intensities_name)
    SHM_COUNT = SharedMemory(name=shm_counts_name)
    INTENSITY_ARRAY = np.ndarray(
        shape=shmshape, dtype=np.float32, buffer=SHM_INTENSITY.buf)
    COUNT_ARRAY = np.ndarray(
        shape=shmshape, dtype=np.float32, buffer=SHM_COUNT.buf)
    lock = l


def get_inc_angles_out(experiment: Experiment,index):
    if np.size(experiment.incident_angle) > 1:
        inc_angle = -np.radians(experiment.incident_angle[index])
    elif isinstance(experiment.incident_angle, np.float64):
        inc_angle = -np.radians(experiment.incident_angle)
    else:
        inc_angle = -np.radians(experiment.incident_angle[0])

    if experiment.setup == 'DCD':
        inc_angle_out = 0  # debug setting incident angle to 0
    else:
        inc_angle_out = inc_angle
    
    return inc_angle,inc_angle_out

def get_gam_del_vals(experiment: Experiment, index):
    gamval = 0
    delval = 0
    if np.size(experiment.gammadata) > 1:
        gamval = -np.array(experiment.two_theta_start).ravel()[index]
    elif np.size(experiment.gammadata) == 1:
        gamval = -np.array(experiment.two_theta_start).ravel()
    if np.size(experiment.deltadata) > 1:
        delval = np.array(experiment.deltadata).ravel()[index]
    elif np.size(experiment.deltadata) == 1:
        delval = np.array(experiment.deltadata).ravel()

    if (-np.degrees(inc_angle) >
            alphacritical) & (experiment.setup == 'DCD'):
        # if above critical angle, account for direct beam adding to delta
        rots = gamdel2rots(gamval, delval + np.degrees(-inc_angle))
    else:
        rots = gamdel2rots(gamval, delval)

    my_ai = copy.deepcopy(aistart)
    my_ai.rot1, my_ai.rot2, my_ai.rot3 = rots

    if experiment.setup == 'vertical':
        my_ai.rot1 = rots[1]
        my_ai.rot2 = -rots[0]

    if slitratios[0] is not None:
        my_ai.pixel1 *= slitratios[0]
        my_ai.poni1 *= slitratios[0]

    if slitratios[1] is not None:
        my_ai.pixel2 *= slitratios[1]
        my_ai.poni2 *= slitratios[1]
    if experiment.setup == 'vertical':
        img_data = np.rot90(scan.load_image(i).data, -1)
    else:
        img_data = np.array(scan.load_image(i).data)

    radial_limits = (limits_in[0] * (1.0 + (0.05 * -(np.sign(limits_in[0])))),
                     limits_in[1] * (1.0 + (0.05 * (np.sign(limits_in[1])))))
    azimuthal_limits = (limits_in[2] * (1.0 + (0.05 * -(np.sign(limits_in[2])))),
                        limits_in[3] * (1.0 + (0.05 * (np.sign(limits_in[3])))))
    limits_out = [radial_limits[0], radial_limits[1],
                  azimuthal_limits[0], azimuthal_limits[1]]

    return unit_ip, unit_oop, img_data, my_ai, limits_out


def pyfai_setup_limits(experiment: Experiment, scanlist, limitfunction, slitratios):
    """
    calculate setup values needed for pyfai calculations
    """
    # pylint: disable=attribute-defined-outside-init
    if isinstance(scanlist, Scan):
        scanlistnew = [scanlist]
    else:
        scanlistnew = scanlist

    limhor = None
    limver = None
    for scan in scanlistnew:
        experiment.load_curve_values(scan)


        if slitratios is not None:
            slitvertratio,slithorratio=slitratios
        else:
            slitvertratio=slithorratio=None
        
        scanlimhor = limitfunction(
            'hor',
            slithorratio=slithorratio)
        scanlimver = limitfunction(
            'vert',
            slitvertratio=slitvertratio)

        scanlimits = [
            scanlimhor[0],
            scanlimhor[1],
            scanlimver[0],
            scanlimver[1]]
        if limhor is None:
            limhor = scanlimits[0:2]
            limver = scanlimits[2:]
        else:
            limhor = combine_ranges(limhor, scanlimits[0:2])
            limver = combine_ranges(limver, scanlimits[2:])

    outlimits = [limhor[0], limhor[1], limver[0], limver[1]]
    if experiment.setup == 'vertical':
        experiment.beam_centre = [
            experiment.beam_centre[1],
            experiment.beam_centre[0]]
        experiment.beam_centre[1] = experiment.imshape[0] - \
            experiment.beam_centre[1]

    datacheck = 'data' in list(scan.metadata.data_file.nx_detector)
    localpathcheck = 'local_image_paths' in \
        scan.metadata.data_file.__dict__.keys()
    intcheck = isinstance(scan.metadata.data_file.scan_length, int)
    if datacheck & intcheck:
        scanlength = np.shape(
            scan.metadata.data_file.nx_detector.data[:, 1, :])[0]
        scanlength = min(scanlength, scan.metadata.data_file.scan_length)
    elif datacheck:
        scanlength = np.shape(
            scan.metadata.data_file.nx_detector.data[:, 1, :])[0]
    elif localpathcheck:
        scanlength = len(scan.metadata.data_file.local_image_paths)
    else:
        scanlength = scan.metadata.data_file.scan_length
    
    #check for scans finished early
    if not scan.metadata.data_file.has_hdf5_data:
        badimagecheck=find_bad_image_paths(scan)
        if len(badimagecheck)>0:
            scanlength-=len(badimagecheck)



def init_pyfai_process_pool(
        locks: List[Lock],
        num_threads: int,
        metadata: RSMMetadata,
        shapeqi: tuple,
    shapecake: tuple,
        shapeqpqpmap: tuple,
        output_file_name: str = None
) -> None:
    """
    Initializes a processing pool to have a global shared lock.

    Args:
        locks:
            A list of the locks that will be shared between spawned processes.
        num_threads:
            The total number of processes that are being spawned in the pool.
        shape:
            Passed if you want to make PYFAI_QI and CAKE arrays global.
    """
    # pylint: disable=global-variable-undefined.

    # Make a global lock for the shared memory block used in parallel code.
    global LOCKS

    # Some metadata that a worker thread should always have access to.
    global NUM_THREADS
    global METADATA

    # Not always necessary and may be set to None.
    global OUTPUT_FILE_NAME

    # These are numpy arrays whose buffer corresponds to the shared memory
    # buffer. It's more convenient to access these later than to directly work
    # with the shared memory buffer.
    global PYFAI_QI
    global CAKE
    global QPQPMAP

    # We want to keep track of what we've called our shared memory arrays.
    global SHARED_PYFAI_QI_NAME
    global SHARED_CAKE_NAME
    global SHARED_QPQPMAP_NAME
    # Why do we need to make the shared memory blocks global, if we're giving
    # global access to them via the numpy 'PYFAI_QI' and 'CAKE' arrays? The answer
    # is that we need the shared memory arrays to remain in scope, or they'll be
    # freed.
    global SHARED_PYFAI_QI
    global SHARED_CAKE
    global SHARED_QPQPMAP

    LOCKS = locks
    NUM_THREADS = num_threads
    METADATA = metadata

    OUTPUT_FILE_NAME = output_file_name

    # Work out how many bytes we're going to need by making a dummy array.
    arrqi = np.ndarray(shape=shapeqi, dtype=np.float32)
    arrcake = np.ndarray(shape=shapecake, dtype=np.float32)
    arrqpqpmap = np.ndarray(shape=shapeqpqpmap, dtype=np.float32)

    # Construct the shared memory buffers.
    SHARED_PYFAI_QI_NAME = f'pyfai_qi_{current_process().name}'
    SHARED_CAKE_NAME = f'cake_{current_process().name}'
    SHARED_QPQPMAP_NAME = f'qpqpmap_{current_process().name}'

    check_shared_memory(SHARED_PYFAI_QI_NAME)
    check_shared_memory(SHARED_CAKE_NAME)
    check_shared_memory(SHARED_QPQPMAP_NAME)

    SHARED_PYFAI_QI = SharedMemory(
        name=SHARED_PYFAI_QI_NAME, create=True, size=arrqi.nbytes)
    SHARED_CAKE = SharedMemory(
        name=SHARED_CAKE_NAME, create=True, size=arrcake.nbytes)
    SHARED_QPQPMAP = SharedMemory(
        name=SHARED_QPQPMAP_NAME, create=True, size=arrqpqpmap.nbytes)

    # Construct the global references to the shared memory arrays.
    PYFAI_QI = np.ndarray(shapeqi, dtype=np.float32,
                          buffer=SHARED_PYFAI_QI.buf)
    CAKE = np.ndarray(shapecake, dtype=np.float32, buffer=SHARED_CAKE.buf)
    QPQPMAP = np.ndarray(shapeqpqpmap, dtype=np.float32,
                         buffer=SHARED_QPQPMAP.buf)

    # Initialize the shared memory arrays.
    PYFAI_QI.fill(0)
    CAKE.fill(0)
    QPQPMAP.fill(0)

    print(f"Finished initializing worker {current_process().name}.")


def start_smm(smm, memshape):
    """
    start up the shared memory manager and associated data arrays
    """
    shm_intensities = smm.SharedMemory(
        size=np.zeros(memshape, dtype=np.float32).nbytes)
    shm_counts = smm.SharedMemory(
        size=np.zeros(
            memshape,
            dtype=np.float32).nbytes)
    arrays_arr = np.ndarray(
        memshape,
        dtype=np.float32,
        buffer=shm_intensities.buf)
    counts_arr = np.ndarray(
        memshape,
        dtype=np.float32,
        buffer=shm_counts.buf)
    arrays_arr.fill(0)
    counts_arr.fill(0)
    l = Lock()
    return shm_intensities, shm_counts, arrays_arr, counts_arr, l

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def load_pyfai_calib_image(ai):
    LaB6 = pyFAI.calibrant.get_calibrant("LaB6")

    det = pyFAI.detectors.Maxipix()  #choose detector with the same pixel size 5.5e-5
    newshape=(515,2069)
    det.shape=newshape
    det.max_shape=newshape
    ai.detector=det
    return LaB6.fake_calibration_image(ai)

def load_flat_test_image():
    dummy_ai=pyFAI.load('/dls/science/users/rpy65944/output/fast_rsm_2026-01-08_13h56m28s.poni')
    dummy_img=load_pyfai_calib_image(dummy_ai)
    flat_image=np.ones(np.shape(dummy_img))
    flat_image[dummy_ai.mask==1]=0
    return flat_image,dummy_ai

def worker_unpack(args):
    function_map={'move_ivq':pyfai_move_ivsq_worker_new,
                  'move_qmap': pyfai_move_qmap_worker_new}
    worker_function=function_map[args[0]]
    worker_args=args[1:]
    # top-level adapter to avoid lambda pickling issues
    return worker_function(*worker_args)

def combine_ranges(range1, range2):
    """
    combines two ranges to give the widest possible range
    """
    return (min(range1[0], range2[0]), max(range1[1], range2[1]))


# ====data saving functions

def save_integration(experiment, hf, twothetas, q_angs,
                     intensities, configs, counts_arr,arrays_arr,scan=0):
    """
    save 1d Intensity Vs Q profile to hdf5 file
    """
    dset = hf.create_group("integrations")
    dset.create_dataset("configs", data=str(configs))
    dset.create_dataset("2thetas", data=twothetas)
    dset.create_dataset("Q_angstrom^-1", data=q_angs)
    dset.create_dataset("Intensity", data=intensities)
    dset.create_dataset("counts",data=counts_arr)
    dset.create_dataset("sum_signal",data=arrays_arr)

 
    if "scanfields" not in hf.keys():
        save_scan_field_values(hf, scan)
    if experiment.savedats is True:
        do_savedats(hf, intensities, q_angs, twothetas)


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
        do_savetiffs(hf,
                                qperp_qpara_map[0],
                                qperp_qpara_map[1],
                                qperp_qpara_map[2])


# oblines, pythonlocation, globalvals):
def save_config_variables(hf, process_config):
    """
    save all variables in the configuration file to the output hdf5 file
    """
    cfg = process_config
    config_group = hf.create_group('i07configuration')
    outdict = vars(cfg)
    with open(cfg.default_config_path, "r",encoding='utf-8') as f:
        default_config_dict = yaml.load(f,Loader=yaml.FullLoader)
    # add in extra to defaults that arent set by user, so that parsing
    # defaults finds it
    default_config_dict['ubinfo'] = 0
    default_config_dict['pythonlocation'] = 0
    default_config_dict['joblines'] = 0
    for key in default_config_dict:
        if key == 'ubinfo':
            for i, coll in enumerate(outdict['ubinfo']):
                ubgroup = config_group.create_group(f'ubinfo_{i+1}')
                ubgroup.create_dataset(
                    f'lattice_{i+1}', data=coll['diffcalc_lattice'])
                ubgroup.create_dataset(f'u_{i+1}', data=coll['diffcalc_u'])
                ubgroup.create_dataset(f'ub_{i+1}', data=coll['diffcalc_ub'])
            continue
        val = outdict[key]
        if val is None:
            val = 'None'
        config_group.create_dataset(f"{key}", data=val)


def save_scan_field_values(hf, scan):
    """
    saves scanfields recorded in nexus file to hdf5 output
    """

    try:
        rank = scan.metadata.data_file.diamond_scan.scan_rank.nxdata
        fields = scan.metadata.data_file.diamond_scan.scan_fields
        scanned = [x.decode('utf-8').split('.')[0]
                   for x in fields[:rank].nxdata]
        scannedvalues = [
            np.unique(
                scan.metadata.data_file.nx_instrument[field].value)for field in scanned]
        scannedvaluesout = [scannedvals[~np.isnan(
            scannedvals)] for scannedvals in scannedvalues]
    except BaseException:
        scanned, scannedvaluesout = None, None

    dset = hf.create_group("scanfields")
    if scan != 0:
        if scanned is not None:
            for i, field in enumerate(scanned):
                dset.create_dataset(
                    f"dim{i}_{field}", data=scannedvaluesout[i])


def save_hf_map(experiment: Experiment, hf, mapname, sum_array,
                counts_array, mapaxisinfo, start_time, process_config):
    cfg = process_config
    norm_array = np.divide(
        sum_array,
        counts_array,
        out=np.copy(sum_array),
        where=counts_array != 0.0)
    end_time = time()
    times = [start_time, end_time]
    dset = hf.create_group(f"{mapname}")
    dset.create_dataset(f"{mapname}_map", data=norm_array)
    dset.create_dataset("map_para", data=mapaxisinfo[1])
    dset.create_dataset("map_para_unit", data=mapaxisinfo[3])
    # list(reversed(mapaxisinfo[0])))
    dset.create_dataset("map_perp", data=mapaxisinfo[0])
    dset.create_dataset("map_perp_unit", data=mapaxisinfo[2])
    dset.create_dataset("map_perp_indices", data=[0, 1, 2])
    dset.create_dataset("map_para_indices", data=[0, 1, 3])

    if cfg.savetiffs:
        do_savetiffs(hf, norm_array, mapaxisinfo[1], mapaxisinfo[0])

    minutes = (times[1] - times[0]) / 60
    print(f'total calculation took {minutes}  minutes')


# ====moving detector processing
def pyfai_moving_qmap_smm_old(experiment: Experiment, hf, scanlist, process_config):
    """
    calculate q_para vs q_perp map for a moving detector scan
    """

    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    cfg = process_config
    qpqp_array_total = 0
    qpqp_counts_total = 0

    cfg.qlimitsout, cfg.scanlength, cfg.scanlistnew = \
    pyfai_setup_limits(experiment,scanlist, experiment.calcqlim, cfg.slitratios)

    cfg.multi = True
    with SharedMemoryManager() as smm:

        cfg.shapeqpqp = (cfg.qmapbins[1], cfg.qmapbins[0])
        shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
            smm, cfg.shapeqpqp)
        start_time = time()
        for scanind, scan in enumerate(cfg.scanlistnew):
            cfg.qlimits, cfg.scanlength, scanlistnew = \
            pyfai_setup_limits(experiment,scan, experiment.calcqlim, cfg.slitratios)
            cfg.scalegamma = 1
            cfg.scan_ind=scanind
            input_args = get_input_args(experiment, scan, cfg)
            print(
                f'starting process pool with num_threads=\
                {cfg.num_threads} for scan {scanind+1}/{len(cfg.scanlistnew)}')

            with Pool(cfg.num_threads,initializer=pyfai_init_worker,\
                initargs=\
                (lock, shm_intensities.name, shm_counts.name, cfg.shapeqpqp)) as pool:
                mapaxisinfolist = pool.starmap(
                    pyfai_move_qmap_worker_old, input_args)
            print(
                f'finished process pool for scan {scanind+1}/{len(cfg.scanlistnew)}')

    mapaxisinfo = mapaxisinfolist[0]
    qpqp_array_total = arrays_arr
    qpqp_counts_total = counts_arr
    end_time = time()
    minutes = (end_time - start_time) / 60
    
    save_hf_map(experiment, hf, "qpara_qperp", qpqp_array_total, qpqp_counts_total,
                mapaxisinfo, start_time, cfg)
    save_config_variables(hf, cfg)
    hf.close()
    return mapaxisinfo

def pyfai_moving_ivsq_smm_old(experiment: Experiment, hf, scanlist, process_config):
    """
    calculate 1d Intensity Vs Q profile for a moving detector scan
    """

    cfg = process_config
    cfg.fullranges, cfg.scanlength, cfg.scanlistnew =\
     pyfai_setup_limits(experiment,scanlist, experiment.calcanglim, cfg.slitratios)
    absranges,radmax=get_corner_thetas(cfg)


    if cfg.radialrange is None:
        centre_check={1:True,0:False,2:False}
        hor_centre=centre_check[np.sum([(val>0) for val in cfg.fullranges[0:2]])]
        ver_centre=centre_check[np.sum([(val>0) for val in cfg.fullranges[2:]])]
        if hor_centre and ver_centre:
            cfg.radialrange = (0, radmax)
        elif hor_centre:
            cfg.radialrange = (min(abs(np.array(cfg.fullranges[2:]))), radmax)
        elif ver_centre:
            cfg.radialrange = (min(abs(np.array(cfg.fullranges[0:2]))), radmax)
        else:
            cfg.radialrange = (min(absranges),radmax)
        

    if cfg.ivqbins is None:
        cfg.ivqbins = int(
            np.ceil((cfg.radialrange[1] - cfg.radialrange[0]) /
                cfg.radialstepval))
    cfg.multi = True
    do_time_check('start shared memory')
    #with SharedMemoryManager() as smm:

    cfg.shapeqi = (3, np.abs(cfg.ivqbins))
    #shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
     #   smm, cfg.shapeqi)

    all_qi = []
    all_counts = []
    do_time_check('NEW start process pool')
    with Pool(2) as pool: #cfg.num_threads)
        for scanind, scan in enumerate(cfg.scanlistnew):
            qlimits, scanlength, scanlistnew = \
            pyfai_setup_limits(experiment,scan, experiment.calcqlim, cfg.slitratios)
            start_time = time()
            cfg.scalegamma = 1
            cfg.scan_ind=scanind
            input_args = get_input_args(experiment, scan, cfg)[0:2]
            print(
                f'starting processing with NEW num_threads=\
                {cfg.num_threads} for scan {scanind+1}/{len(cfg.scanlistnew)}')

        # with Pool(cfg.num_threads,
        #             initializer=pyfai_init_worker,
        #             initargs=(lock, shm_intensities.name, shm_counts.name, cfg.shapeqi)) as pool:
            partials=pool.starmap(pyfai_move_ivsq_worker_old, input_args)
            print(f'finished processing scan {scanind+1}/{len(cfg.scanlistnew)}')
 

            all_qi.append(np.add.reduce([p[0] for p in partials]))
            all_counts.append(np.add.reduce([p[1] for p in partials]))
    do_time_check('stop process pool')

    qi_final = np.add.reduce(all_qi)
    counts_final = np.add.reduce(all_counts)

    qi_array = np.divide(
        qi_final[0],
        counts_final[0],
        out=np.copy(
            qi_final[0]),
        where=counts_final[0].astype(float) != 0.0)
    end_time = time()
    minutes = (end_time - start_time) / 60
    print(f'total calculation took {minutes}  minutes')

    dset = hf.create_group("integrations")
    dset.create_dataset("Intensity", data=qi_array)
    dset.create_dataset("Q_angstrom^-1", data=qi_final[1])
    dset.create_dataset("2thetas", data=qi_final[2])
    # dset.create_dataset("counts",data=counts_arr[0])
    # dset.create_dataset("sum_signal",data=arrays_arr[0])
    # dset.create_dataset("solid_intensity",data=counts_arr[1])
    # dset.create_dataset("solid_sum_signal",data=counts_arr[2])

    if cfg.savedats:
        do_savedats(hf, qi_array, qi_final[1], qi_final[2])
    save_config_variables(hf, cfg)
    hf.close()


def pyfai_moving_exitangles_smm(experiment: Experiment, hf, scanlist, process_config):
    """
    calculate exit angle map with moving detector
    """
    cfg = process_config

    exhexv_array_total = 0
    exhexv_counts_total = 0
    cfg.anglimitsout, cfg.scanlength, cfg.scanlistnew = \
    pyfai_setup_limits(experiment,scanlist, experiment.calcanglim, cfg.slitratios)
    cfg.multi = True
    with SharedMemoryManager() as smm:

        cfg.shapeexhexv = (cfg.qmapbins[1], cfg.qmapbins[0])
        shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
            smm, cfg.shapeexhexv)
        start_time = time()
        for scanind, scan in enumerate(cfg.scanlistnew):

            cfg.anglimits, cfg.scanlength, scanlistnew = \
            pyfai_setup_limits(experiment,scan, experiment.calcanglim, cfg.slitratios)
            cfg.scalegamma = 1
            input_args = get_input_args(experiment, scan, cfg)
            print(f'starting process pool with num_threads=\
                  {cfg.num_threads} for scan {scanind+1}/{len(cfg.scanlistnew)}')

            with Pool(cfg.num_threads, initializer=pyfai_init_worker, \
            initargs=(lock, shm_intensities.name, shm_counts.name, cfg.shapeexhexv)) as pool:
                mapaxisinfolist = pool.starmap(
                    pyfai_move_exitangles_worker_old, input_args)
            print(
                f'finished process pool for scan {scanind+1}/{len(cfg.scanlistnew)}')

    mapaxisinfo = mapaxisinfolist[0]
    exhexv_array_total = arrays_arr
    exhexv_counts_total = counts_arr
    save_hf_map(experiment, hf, "exit_angles", exhexv_array_total, exhexv_counts_total,
                mapaxisinfo, start_time, cfg)
    save_config_variables(hf, cfg)
    hf.close()
    return mapaxisinfo



# def init_worker_logger(log_queue, level=logging.INFO):
#     root = logging.getLogger('fastrsm_debug')
#     root.setLevel(level)
#     root.handlers[:] = []
#     root.addHandler(logging.handlers.QueueHandler(log_queue))
#     root.propagate = False


# def setup_parent_logging():
#     log_queue = Queue()
#     stream = logging.StreamHandler()
#     stream.setFormatter(logging.Formatter("%(asctime)s [pid=%(process)d] %(levelname)s: %(message)s"))
#     listener = logging.handlers.QueueListener(log_queue, stream)
#     listener.start()
#     return log_queue

def start_listener():
    manager = Manager()
    log_queue = manager.Queue()
    listener = Process(target=listener_process,
                                        args=(log_queue, get_logger, LOGGER_DEBUG))
    listener.start()
    return listener,log_queue

def pyfai_moving_qmap_smm_new(experiment: Experiment, hf, scanlist, process_config):
    """
    calculate q_para vs q_perp map for a moving detector scan
    """

    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    logger=get_logger(LOGGER_DEBUG)
    manager = Manager()
    log_queue = manager.Queue()
    listener = Process(target=listener_process,
                                        args=(log_queue, get_logger, LOGGER_DEBUG))
    listener.start()
    cfg = process_config
    ctx = get_context("spawn")
    cfg = process_config
    qpqp_array_total = 0
    qpqp_counts_total = 0

    cfg.qlimitsout, cfg.scanlength, cfg.scanlistnew = \
    pyfai_setup_limits(experiment,scanlist, experiment.calcqlim, cfg.slitratios)
    intensity_results_per_scan = []
    count_results_per_scan = []
    para_results_per_scan=[]
    perp_results_per_scan=[]    
    t0 = time()
    cfg.multi = True

    logger.debug(do_time_check('NEW start process pool'))
    cfg.unit_qip_name = "2th_deg"  # "qtot_A^-1"# "qip_A^-1"
    cfg.unit_qoop_name = "qoop_A^-1"
    cfg.sample_orientation = 1
    batchsize=15
    ctx = get_context("spawn")
    with ctx.Pool(processes=cfg.num_threads ) as pool:
         
         for scanind, scan in enumerate(cfg.scanlistnew):# chunksize=1 makes sense here: each task is already “large” (25 images)
            
            cfg.aistart = pyFAI.load(cfg.pyfaiponi,type_="pyFAI.integrator.fiber.FiberIntegrator")
            cfg.d5i_full=get_d5i_values(scan)
            imageindices=get_full_indices(scan,cfg)
            cfg.gamdelvals=[get_gam_del_vals(experiment,ind) for ind in imageindices]
            cfg.all_inc_angles=[get_inc_angles_out(experiment,ind) for ind in imageindices]
            batches = list(chunked(imageindices, batchsize))
            num_batches = len(batches)
            args_iter = (('move_qmap',experiment, batch, scan, cfg,log_queue,logn) for logn,batch in enumerate(batches))
            accumulator_intensity = np.zeros((1, cfg.qmapbins), dtype=np.float32)
            accumulator_count = np.zeros((1, cfg.qmapbins), dtype=np.float32)
            accumulator_para = np.zeros((1, cfg.qmapbins[0]), dtype=np.float32)
            accumulator_perp = np.zeros((1, cfg.qmapbins[1]), dtype=np.float32)

            accumulator_mask=[]
            completed = 0           
            
            for partial in pool.imap_unordered(worker_unpack, args_iter, chunksize=1):
                if (completed==0)&(scanind==0):
                    accumulator_mask=partial[4]
                accumulator_intensity += partial[0]
                accumulator_count += partial[1]
                accumulator_para += partial[2]
                accumulator_para += partial[3]
                
                completed += 1
                if completed % 10 == 0 or completed == num_batches:
                    print(f"  completed {completed}/{num_batches} batches", flush=True)
            intensity_results_per_scan.append(accumulator_intensity)
            count_results_per_scan.append(accumulator_count)
            para_results_per_scan.append(accumulator_para)
            perp_results_per_scan.append(accumulator_perp)
    
    log_queue.put_nowait(None) # End the queue
    listener.join() # Stop the listener

    logger.debug(do_time_check('stop process pool'))

    qmap_final=np.sum(intensity_results_per_scan,axis=0)
    counts_final = np.sum(count_results_per_scan,axis=0)
    para_vals_final=para_results_per_scan[0]
    perp_vals_final=para_results_per_scan[0]
    qmap_array = np.divide(
        qmap_final[0],
        counts_final[0],
        out=np.copy(
            qmap_final[0]),
        where=counts_final[0].astype(float) > 0.0)
    end_time = time()
    minutes = (end_time - t0) / 60
    print(f'total calculation took {minutes}  minutes')
        
    dset = hf.create_group("qpara_qperp")
    dset.create_dataset("qpara_qperp_map", data=qmap_array)
    dset.create_dataset("map_para", data=para_vals_final)
    dset.create_dataset("map_perp", data=perp_vals_final)
    dset.create_dataset("map_perp_indices", data=[0, 1, 2])
    dset.create_dataset("map_para_indices", data=[0, 1, 3])

    # if "scanfields" not in hf.keys():
    #     save_scan_field_values(hf, scan)
    # if cfg.savetiffs:
    #     do_savetiffs(hf, qmap_array, para_vals_final, perp_vals_final)
    save_config_variables(hf, cfg)
    hf.close()
            
   
    # with SharedMemoryManager() as smm:

    #     cfg.shapeqpqp = (cfg.qmapbins[1], cfg.qmapbins[0])
    #     shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
    #         smm, cfg.shapeqpqp)
    #     start_time = time()
    #     for scanind, scan in enumerate(cfg.scanlistnew):
    #         cfg.qlimits, cfg.scanlength, scanlistnew = \
    #         pyfai_setup_limits(experiment,scan, experiment.calcqlim, cfg.slitratios)
    #         cfg.scalegamma = 1
    #         cfg.scan_ind=scanind
    #         input_args = get_input_args(experiment, scan, cfg)
    #         print(
    #             f'starting process pool with num_threads=\
    #             {cfg.num_threads} for scan {scanind+1}/{len(cfg.scanlistnew)}')

    #         with Pool(cfg.num_threads,initializer=pyfai_init_worker,\
    #             initargs=\
    #             (lock, shm_intensities.name, shm_counts.name, cfg.shapeqpqp)) as pool:
    #             mapaxisinfolist = pool.starmap(
    #                 pyfai_move_qmap_worker_old, input_args)
    #         print(
    #             f'finished process pool for scan {scanind+1}/{len(cfg.scanlistnew)}')

    # mapaxisinfo = mapaxisinfolist[0]
    # qpqp_array_total = arrays_arr
    # qpqp_counts_total = counts_arr
    # end_time = time()
    # minutes = (end_time - start_time) / 60
    
    # save_hf_map(experiment, hf, "qpara_qperp", qpqp_array_total, qpqp_counts_total,
    #             mapaxisinfo, start_time, cfg)
    # save_config_variables(hf, cfg)
    # hf.close()
    # return mapaxisinfo

def check_full_1d_radial_range(experiment: Experiment,process_config,absranges, radmax):
    cfg=process_config
    centre_check={1:True,0:False,2:False}
    hor_centre=centre_check[np.sum([(val>0) for val in cfg.fullranges[0:2]])]
    ver_centre=centre_check[np.sum([(val>0) for val in cfg.fullranges[2:]])]
    if hor_centre and ver_centre:
        radialrange = (0, radmax)
    elif hor_centre:
        radialrange = (min(abs(np.array(cfg.fullranges[2:]))), radmax)
    elif ver_centre:
        radialrange = (min(abs(np.array(cfg.fullranges[0:2]))), radmax)
    else:
        radialrange = (min(absranges),radmax)
    if str(cfg.unit_qip_name).startswith("q"):
        return [calcq(val,experiment.incident_wavelength) for val in radialrange]
    return radialrange

def get_d5i_values(scan):
    if hasattr(scan.metadata.data_file.nx_entry,'d5i'):
        d5i_full=np.array(scan.metadata.data_file.nx_entry.d5i.data)
    else:
        d5i_full=np.ones(scan.metadata.data_file.scan_length)
    return d5i_full

def pyfai_moving_ivsq_smm_new(experiment: Experiment, hf, scanlist, process_config):
    """
    calculate 1d Intensity Vs Q profile for a moving detector scan
    """

    logger=get_logger(LOGGER_DEBUG)
    listener,log_queue=start_listener()
    cfg = process_config
    ctx = get_context("spawn")
    cfg.fullranges, cfg.scanlength, cfg.scanlistnew =\
     pyfai_setup_limits(experiment,scanlist, experiment.calcanglim, cfg.slitratios)
    absranges,radmax=get_corner_thetas(cfg)
    #num_threads = int(cfg.num_threads)  # e.g., 40
    intensity_results_per_scan = []
    count_results_per_scan = []
    qtot_results_per_scan=[]
    t0 = time()
    cfg.unit_qip_name =  "qip_A^-1"# "qip_A^-1""2th_deg"  #
    cfg.unit_qoop_name = "qoop_A^-1"
    if cfg.radialrange is None:
        cfg.radialrange=check_full_1d_radial_range(experiment,cfg,absranges,radmax)

        
    if cfg.ivqbins is None:
        cfg.ivqbins = int(
            np.ceil((cfg.radialrange[1] - cfg.radialrange[0]) /
                cfg.radialstepval))
    
    cfg.multi = True

    cfg.shapeqi = (1, np.abs(cfg.ivqbins))
    cfg.scalegamma = 1

    logger.debug(do_time_check('NEW start process pool'))

    cfg.sample_orientation = 1
    batchsize=15
    ctx = get_context("spawn")
    with ctx.Pool(processes=cfg.num_threads ) as pool:
        for scanind, scan in enumerate(cfg.scanlistnew):# chunksize=1 makes sense here: each task is already “large” (25 images)
            
            cfg.aistart = pyFAI.load(cfg.pyfaiponi,type_="pyFAI.integrator.fiber.FiberIntegrator")
            cfg.d5i_full=get_d5i_values(scan)
            imageindices=get_full_indices(scan,cfg)
            cfg.gamdelvals=[get_gam_del_vals(experiment,ind) for ind in imageindices]
            cfg.all_inc_angles=[get_inc_angles_out(experiment,ind) for ind in imageindices]
            
            batches = list(chunked(imageindices, batchsize))
            num_batches = len(batches)
            args_iter = (('move_ivq',experiment, batch, scan, cfg,log_queue,logn) for logn,batch in enumerate(batches))
            
            accumulator_intensity = np.zeros((1, cfg.ivqbins), dtype=np.float32)
            accumulator_count = np.zeros((1, cfg.ivqbins), dtype=np.float32)
            accumulator_theta = np.zeros((1, cfg.ivqbins), dtype=np.float32)
            accumulator_mask=[]
            completed = 0
            #accumulator_mask=[]
            for partial in pool.imap_unordered(worker_unpack, args_iter, chunksize=1):
                if (completed==0)&(scanind==0):
                    accumulator_mask=partial[3]
                accumulator_intensity += partial[0]
                accumulator_count += partial[1]
                accumulator_theta += partial[2]
                
                completed += 1
                if completed % 10 == 0 or completed == num_batches:
                    print(f"  completed {completed}/{num_batches} batches", flush=True)
                        
            intensity_results_per_scan.append(accumulator_intensity)
            count_results_per_scan.append(accumulator_count)
            qtot_results_per_scan.append(accumulator_theta/completed)
            print(f"[scan {scanind+1}] finished.")

    log_queue.put_nowait(None) # End the queue
    listener.join() # Stop the listener

    logger.debug(do_time_check('stop process pool'))

    qi_final = np.sum(intensity_results_per_scan,axis=0)
    counts_final = np.sum(count_results_per_scan,axis=0)
    qtot_vals_final=qtot_results_per_scan[0]
    #qvals_final=[calcq(val, experiment.incident_wavelength) for val in theta_vals_final]
    thetas_final=[calctheta(val,experiment.incident_wavelength) for val in qtot_vals_final]


    qi_array = np.divide(
        qi_final[0],
        counts_final[0],
        out=np.copy(
            qi_final[0]),
        where=counts_final[0].astype(float) > 0.0)
    end_time = time()
    minutes = (end_time - t0) / 60
    print(f'total calculation took {minutes}  minutes')

    dset = hf.create_group("integrations")
    dset.create_dataset("Intensity", data=np.expand_dims(qi_array,0))
    dset.create_dataset("Q_angstrom^-1", data=qtot_vals_final)
    dset.create_dataset("2thetas", data=thetas_final)
    dset.create_dataset("mask", data=accumulator_mask)
    # dset.create_dataset("counts",data=counts_arr[0])
    # dset.create_dataset("sum_signal",data=arrays_arr[0])
    # dset.create_dataset("solid_intensity",data=counts_arr[1])
    # dset.create_dataset("solid_sum_signal",data=counts_arr[2])

    if cfg.savedats:
        do_savedats(hf, qi_array, qvals_final, theta_vals_final)
    save_config_variables(hf, cfg)
    hf.close()

# ====static detector processing

def pyfai_static_exitangles(experiment: Experiment, hf, scan,
                            process_config: SimpleNamespace):
    """
    calculate the map of vertical exit angle Vs horizontal exit angle using pyFAI

    Parameters
    ----------
    hf : hdf5 file
        open hdf5 file to write data to.
    scan : scan object
        scan to be analysed.
    num_threads : int
        number of threads used in calculation.
    pyfaiponi : string
        path to PONI file.
    ivqbins : int
        number of bins for I Vs Q profile.
    qmapbins : array, optional
        number of x,y bins for map. The default is 0.

    Returns
    -------
    None.

    """
    cfg = process_config
    start_time = time()
    cfg.anglimits, cfg.scanlength, cfg.scanlistnew = \
    pyfai_setup_limits(experiment,scan, experiment.calcanglim, cfg.slitratios)
    # calculate map bins if not specified using resolution of 0.01 degrees
    if cfg.qmapbins == 0:
        cfg.qmapbins = get_qmapbins(cfg.qlimits, experiment)
    cfg.scalegamma = 1

    print(f'starting process pool with num_threads={ cfg.num_threads}')
    all_maps = []
    all_xlabels = []
    all_ylabels = []
    all_mapaxisinfo = []
    cfg.multi = False
    cfg.scan_ind=0
    with Pool(processes=cfg.num_threads) as pool:
        input_args = get_input_args(experiment, scan, cfg)
        results = pool.starmap(pyfai_stat_exitangles_worker, input_args)
        maps = [result[0] for result in results]
        xlabels = [result[1] for result in results]
        ylabels = [result[2] for result in results]
        mapaxisinfo = [result[3] for result in results]
        all_maps.append(maps)
        all_xlabels.append(xlabels)
        all_ylabels.append(ylabels)
        all_mapaxisinfo.append(mapaxisinfo)

    print('finished process pool')

    signal_shape = np.shape(scan.metadata.data_file.default_signal)
    if len(signal_shape) > 1:
        savemaps = reshape_to_signalshape(all_maps[0], signal_shape)
    else:
        savemaps = all_maps[0]
    if "scanfields" not in hf.keys():
        save_scan_field_values(hf, scan)
    save_hf_map(experiment,
                hf, "exit_angles", savemaps, np.ones(np.shape(savemaps)),
                all_mapaxisinfo[0][0],
                start_time, cfg)
    save_config_variables(hf, cfg)
    hf.close()


def pyfai_static_qmap(experiment: Experiment, hf, scan, process_config: SimpleNamespace):
    """
    calculate 2d q_para vs q_perp mape for a static detector scan
    """
    cfg = process_config
    cfg.qlimits, cfg.scanlength, cfg.scanlistnew =\
     pyfai_setup_limits(experiment, scan, experiment.calcqlim, cfg.slitratios)

    # calculate map bins if not specified using resolution of 0.01 degrees
    if cfg.qmapbins == 0:
        cfg.qmapbins = get_qmapbins(cfg.qlimits, experiment)
    cfg.scalegamma = 1

    print(f'starting process pool with num_threads={cfg.num_threads}')
    all_maps = []
    all_xlabels = []
    all_ylabels = []
    cfg.multi = False
    cfg.scan_ind=0
    with Pool(processes=cfg.num_threads) as pool:
        input_args = get_input_args(experiment, scan, cfg)
        results = pool.starmap(pyfai_stat_qmap_worker, input_args)
        maps = [result[0] for result in results]
        xlabels = [result[1] for result in results]
        ylabels = [result[2] for result in results]
        all_maps.append(maps)
        all_xlabels.append(xlabels)
        all_ylabels.append(ylabels)

    print('finished process pool')

    signal_shape = np.shape(scan.metadata.data_file.default_signal)
    outlist = [all_maps[0], all_xlabels[0], all_ylabels[0]]
    if len(signal_shape) > 1:
        outlist = [
            reshape_to_signalshape(
                arr, signal_shape) for arr in outlist]

    dset = hf.create_group("qpara_qperp")
    dset.create_dataset("qpara_qperp_map", data=outlist[0])
    dset.create_dataset("map_para", data=outlist[1])
    dset.create_dataset("map_perp", data=outlist[2])
    dset.create_dataset("map_perp_indices", data=[0, 1, 2])
    dset.create_dataset("map_para_indices", data=[0, 1, 3])

    if "scanfields" not in hf.keys():
        save_scan_field_values(hf, scan)
    if cfg.savetiffs:
        do_savetiffs(hf, outlist[0], outlist[1], outlist[2])
    save_config_variables(hf, cfg)
    hf.close()


def pyfai_static_ivsq(experiment: Experiment, hf, scan, process_config: SimpleNamespace):
    """
    calculate Intensity Vs Q 1d profile from static detector scan
    """
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    cfg = process_config
    cfg.fullranges, cfg.scanlength, cfg.scanlistnew = \
     pyfai_setup_limits(experiment, scan, experiment.calcanglim, cfg.slitratios)
    
    absranges,radmax=get_corner_thetas(cfg)

    centre_check={1:True,0:False,2:False}
    hor_centre=centre_check[np.sum([(val>0) for val in cfg.fullranges[0:2]])]
    ver_centre=centre_check[np.sum([(val>0) for val in cfg.fullranges[2:]])]
    if hor_centre and ver_centre:
        cfg.radialrange = (0, radmax)
    elif hor_centre:
        cfg.radialrange = (min(abs(np.array(cfg.fullranges[2:]))), radmax)
    elif ver_centre:
        cfg.radialrange = (min(abs(np.array(cfg.fullranges[0:2]))), radmax)
    else:
        cfg.radialrange = (min(absranges),radmax)

    if cfg.ivqbins is None:
        cfg.ivqbins = int(
            np.ceil((cfg.radialrange[1] - cfg.radialrange[0]) /
                cfg.radialstepval))
    cfg.scalegamma = 1
    print(f'starting process pool with num_threads={cfg.num_threads}')
    all_ints = []
    all_two_ths = []
    all_qs = []
    cfg.multi = False
    cfg.scan_ind=0
    with Pool(processes=cfg.num_threads) as pool:
        input_args = get_input_args(experiment, scan, cfg)
        results = pool.starmap(pyfai_stat_ivsq_worker, input_args)
        intensities = [result[0] for result in results]
        two_th_vals = [result[1] for result in results]
        q_vals = [result[2] for result in results]
        all_ints.append(intensities)
        all_two_ths.append(two_th_vals)
        all_qs.append(q_vals)

    print('finished process pool')

    signal_shape = np.shape(scan.metadata.data_file.default_signal)
    outlist = [all_ints[0], all_qs[0], all_two_ths[0]]
    if len(signal_shape) > 1:
        outlist = [
            reshape_to_signalshape(
                arr, signal_shape) for arr in outlist]

    dset = hf.create_group("integrations")
    dset.create_dataset("Intensity", data=outlist[0])
    dset.create_dataset("Q_angstrom^-1", data=outlist[1])
    dset.create_dataset("2thetas", data=outlist[2])
    if "scanfields" not in hf.keys():
        save_scan_field_values(hf, scan)
    if cfg.savedats is True:
        do_savedats(hf, outlist[0], outlist[1], outlist[2])
    save_config_variables(hf, cfg)
    hf.close()
