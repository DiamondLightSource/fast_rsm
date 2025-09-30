"""
Module for the functions used to interface with pyFAI package
"""
from types import SimpleNamespace
import logging
from datetime import datetime
from time import time
from multiprocessing import current_process, Lock, Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from typing import  List
import copy
import yaml
from pyFAI.multi_geometry import MultiGeometry
import pyFAI
from pyFAI import units
import numpy as np

from diffraction_utils import Frame  # I10Nexus, Vector3,
# from diffraction_utils.diffractometers import I10RasorDiffractometer

from fast_rsm.rsm_metadata import RSMMetadata
from fast_rsm.scan import Scan, chunk, check_shared_memory


logger = logging.getLogger("fastrsm")

# ====general functions


def combine_ranges(range1, range2):
    """
    combines two ranges to give the widest possible range
    """
    return (min(range1[0], range2[0]), max(range1[1], range2[1]))


def createponi(experiment, outpath, offset=0):
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


def get_input_args(experiment, scan, process_config: SimpleNamespace):
    """
    create the input arguments for processing depending on process
    configuration
    """
    cfg = process_config
    fullrange = np.arange(0, cfg.scanlength, cfg.scalegamma)
    selectedindices = [
        n for n in fullrange if n not in cfg.skipimages]
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

    qstep = round(experiment.calcq(1.00, experiment.incident_wavelength) -
                  experiment.calcq(1.01, experiment.incident_wavelength), 4)
    binshor = abs(round(((qlimits[1] - qlimits[0]) / qstep) * 1.05))
    binsver = abs(round(((qlimits[3] - qlimits[2]) / qstep) * 1.05))
    return (binshor, binsver)

# ====save functions


def save_integration(experiment, hf, twothetas, q_angs,
                     intensities, configs, scan=0):
    """
    save 1d Intensity Vs Q profile to hdf5 file
    """
    dset = hf.create_group("integrations")
    dset.create_dataset("configs", data=str(configs))
    dset.create_dataset("2thetas", data=twothetas)
    dset.create_dataset("Q_angstrom^-1", data=q_angs)
    dset.create_dataset("Intensity", data=intensities)
    if "scanfields" not in hf.keys():
        save_scan_field_values(hf, scan)
    if experiment.savedats is True:
        experiment.do_savedats(hf, intensities, q_angs, twothetas)


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
        experiment.do_savetiffs(hf,
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
    with open(cfg.default_config_path, "r") as f:
        default_config_dict = yaml.safe_load(f,Loader=yaml.FullLoader)
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


def save_hf_map(experiment, hf, mapname, sum_array,
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
        experiment.do_savetiffs(hf, norm_array, mapaxisinfo[1], mapaxisinfo[0])

    minutes = (times[1] - times[0]) / 60
    print(f'total calculation took {minutes}  minutes')


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


def get_pyfai_components(experiment, i, sample_orientation, unit_ip_name,
                         unit_oop_name, aistart, slitratios, alphacritical, scan, limits_in):
    """
    get components need for mapping with pyFAI
    """
    if np.size(experiment.incident_angle) > 1:
        inc_angle = -np.radians(experiment.incident_angle[i])
    elif isinstance(experiment.incident_angle, np.float64):
        inc_angle = -np.radians(experiment.incident_angle)
    else:
        inc_angle = -np.radians(experiment.incident_angle[0])

    if experiment.setup == 'DCD':
        inc_angle_out = 0  # debug setting incident angle to 0
    else:
        inc_angle_out = inc_angle

    unit_ip = units.get_unit_fiber(
        unit_ip_name, sample_orientation=sample_orientation, incident_angle=inc_angle_out)
    unit_oop = units.get_unit_fiber(
        unit_oop_name, sample_orientation=sample_orientation, incident_angle=inc_angle_out)

    gamval = 0
    delval = 0
    if np.size(experiment.gammadata) > 1:
        gamval = -np.array(experiment.two_theta_start).ravel()[i]
    if np.size(experiment.gammadata) == 1:
        gamval = -np.array(experiment.two_theta_start).ravel()
    if np.size(experiment.deltadata) > 1:
        delval = np.array(experiment.deltadata).ravel()[i]
    if np.size(experiment.deltadata) == 1:
        delval = np.array(experiment.deltadata).ravel()

    if (-np.degrees(inc_angle) >
            alphacritical) & (experiment.setup == 'DCD'):
        # if above critical angle, account for direct beam adding to delta
        rots = experiment.gamdel2rots(gamval, delval + np.degrees(-inc_angle))
    # elif (experiment.setup=='DCD'):
    #     rots = experiment.gamdel2rots(gamval, delval)
    else:
        rots = experiment.gamdel2rots(gamval, delval)

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


def pyfai_setup_limits(experiment, scanlist, limitfunction, slitratios):
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
        dcd_sample_dist = 1e-3 * scan.metadata.diffractometer._dcd_sample_distance
        if experiment.setup == 'DCD':
            tthdirect = -1 * \
                np.degrees(np.arctan(experiment.projectionx / dcd_sample_dist))
        else:
            tthdirect = 0

        experiment.two_theta_start = experiment.gammadata - tthdirect

        if slitratios is not None:
            scanlimhor = limitfunction(
                'hor',
                vertsetup=(
                    experiment.setup == 'vertical'),
                slithorratio=slitratios[1])
            scanlimver = limitfunction(
                'vert',
                vertsetup=(
                    experiment.setup == 'vertical'),
                slitvertratio=slitratios[0])
        else:
            scanlimhor = limitfunction(
                'hor', vertsetup=(
                    experiment.setup == 'vertical'))
            scanlimver = limitfunction(
                'vert', vertsetup=(
                    experiment.setup == 'vertical'))

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

    return outlimits, scanlength, scanlistnew


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


# ====moving detector processing
def pyfai_moving_exitangles_smm(experiment, hf, scanlist, process_config):
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

            cfg.anglimits, cfg.scanlength, cfg.scanlistnew = \
            pyfai_setup_limits(experiment,scan, experiment.calcanglim, cfg.slitratios)
            cfg.scalegamma = 1
            input_args = get_input_args(experiment, scan, cfg)
            print(f'starting process pool with num_threads=\
                  {cfg.num_threads} for scan {scanind+1}/{len(cfg.scanlistnew)}')

            with Pool(cfg.num_threads, initializer=pyfai_init_worker, \
            initargs=(lock, shm_intensities.name, shm_counts.name, cfg.shapeexhexv)) as pool:
                mapaxisinfolist = pool.starmap(
                    pyfai_move_exitangles_worker, input_args)
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


def pyfai_moving_qmap_smm(experiment, hf, scanlist, process_config):
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
            cfg.qlimits, cfg.scanlength, cfg.scanlistnew = \
            pyfai_setup_limits(experiment,scan, experiment.calcqlim, cfg.slitratios)
            cfg.scalegamma = 1
            input_args = get_input_args(experiment, scan, cfg)
            print(
                f'starting process pool with num_threads=\
                {cfg.num_threads} for scan {scanind+1}/{len(cfg.scanlistnew)}')

            with Pool(cfg.num_threads,initializer=pyfai_init_worker,\
                initargs=\
                (lock, shm_intensities.name, shm_counts.name, cfg.shapeqpqp)) as pool:
                mapaxisinfolist = pool.starmap(
                    pyfai_move_qmap_worker, input_args)
            print(
                f'finished process pool for scan {scanind+1}/{len(cfg.scanlistnew)}')

    mapaxisinfo = mapaxisinfolist[0]
    qpqp_array_total = arrays_arr
    qpqp_counts_total = counts_arr
    end_time = time()
    minutes = (end_time - start_time) / 60
    print(f'total calculation took {minutes}  minutes')
    save_hf_map(experiment, hf, "qpara_qperp", qpqp_array_total, qpqp_counts_total,
                mapaxisinfo, start_time, cfg)
    save_config_variables(hf, cfg)
    hf.close()
    return mapaxisinfo


def pyfai_moving_ivsq_smm(experiment, hf, scanlist, process_config):
    """
    calculate 1d Intensity Vs Q profile for a moving detector scan
    """

    cfg = process_config
    cfg.fullranges, cfg.scanlength, cfg.scanlistnew =\
     pyfai_setup_limits(experiment,scanlist, experiment.calcanglim, cfg.slitratios)
    absranges = np.abs(cfg.fullranges)
    radmax = np.max(absranges)
    con1 = np.abs(
        cfg.fullranges[0]) < np.abs(
        cfg.fullranges[0] -
        cfg.fullranges[1])
    con2 = np.abs(
        cfg.fullranges[2]) < np.abs(
        cfg.fullranges[2] -
        cfg.fullranges[3])

    if (con1) & (con2):
        cfg.radrange = (0, radmax)

    elif con1:
        cfg.radrange = np.sort([absranges[2], absranges[3]])
    elif con2:
        cfg.radrange = np.sort([absranges[0], absranges[1]])
    else:
        cfg.radrange = (np.max([absranges[0], absranges[2]]),
                        np.max([absranges[1], absranges[3]]))

    cfg.nqbins = int(
        np.ceil(
            (cfg.radrange[1] -
             cfg.radrange[0]) /
            cfg.radialstepval))
    cfg.multi = True
    with SharedMemoryManager() as smm:

        cfg.shapeqi = (3, np.abs(cfg.nqbins))
        shm_intensities, shm_counts, arrays_arr, counts_arr, lock = start_smm(
            smm, cfg.shapeqi)

        for scanind, scan in enumerate(cfg.scanlistnew):
            qlimits, scanlength, scanlistnew = \
            pyfai_setup_limits(experiment,scan, experiment.calcqlim, cfg.slitratios)
            start_time = time()
            cfg.scalegamma = 1
            input_args = get_input_args(experiment, scan, cfg)
            print(
                f'starting process pool with num_threads=\
                {cfg.num_threads} for scan {scanind+1}/{len(cfg.scanlistnew)}')

            with Pool(cfg.num_threads,
                      initializer=pyfai_init_worker,
                      initargs=(lock, shm_intensities.name, shm_counts.name, cfg.shapeqi)) as pool:
                pool.starmap(pyfai_move_ivsq_worker, input_args)
            print(
                f'finished process pool for scan {scanind+1}/{len(cfg.scanlistnew)}')
    qi_array = np.divide(
        arrays_arr[0],
        counts_arr[0],
        out=np.copy(
            arrays_arr[0]),
        where=counts_arr[0] != 0)
    end_time = time()
    minutes = (end_time - start_time) / 60
    print(f'total calculation took {minutes}  minutes')

    dset = hf.create_group("integrations")
    dset.create_dataset("Intensity", data=qi_array)
    dset.create_dataset("Q_angstrom^-1", data=arrays_arr[1])
    dset.create_dataset("2thetas", data=arrays_arr[2])

    if cfg.savedats:
        experiment.do_savedats(hf, qi_array, arrays_arr[1], arrays_arr[2])
    save_config_variables(hf, cfg)
    hf.close()


def pyfai_move_qmap_worker(experiment, imageindices,
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

    if cfg.qlimits is None:
        qlimhor = experiment.calcqlim('hor', vertsetup=(
            experiment.setup == 'vertical'), slithorratio=cfg.slitratios[0])
        qlimver = experiment.calcqlim('vert', vertsetup=(
            experiment.setup == 'vertical'), slitvertratio=cfg.slitratios[1])
        cfg.qlimits = [qlimhor[0], qlimhor[1], qlimver[0], qlimver[1]]

    sample_orientation = 1

    groupnum = 15

    groups = [imageindices[i:i + groupnum]
              for i in range(0, len(imageindices), groupnum)]
    for group in groups:
        ais = []
        img_data_list = []
        for i in group:
            unit_qip, unit_qoop, img_data, my_ai, ai_limits = \
                get_pyfai_components(experiment, i, sample_orientation,\
                unit_qip_name, unit_qoop_name, aistart, cfg.slitratios,\
                cfg.alphacritical, scan, cfg.qlimits)

            img_data_list.append(img_data)
            ais.append(my_ai)

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


def pyfai_move_ivsq_worker(experiment, imageindices,
                           scan, process_config) -> None:
    """
    calculate 1d intensity vs q profile for moving detector scan using pyFAI

    """
    cfg = process_config
    global INTENSITY_ARRAY, COUNT_ARRAY

    # , type_="pyFAI.integrator.fiber.FiberIntegrator")
    aistart = pyFAI.load(cfg.pyfaiponi)
    # 15-07-2025  fiber integrator not currently working with multigeomtery
    totaloutqi = np.zeros(cfg.shapeqi)
    totaloutcounts = np.zeros(cfg.shapeqi)

    unit_qip_name = "2th_deg"  # "qtot_A^-1"# "qip_A^-1"
    unit_qoop_name = "qoop_A^-1"

    sample_orientation = 1
    groupnum = 15
    groups = [imageindices[i:i + groupnum]
              for i in range(0, len(imageindices), groupnum)]
    for group in groups:
        ais = []
        img_data_list = []
        for i in group:
            unit_tth_ip, unit_qoop, img_data, my_ai, ai_limits =\
             get_pyfai_components(experiment, i, sample_orientation, unit_qip_name,\
             unit_qoop_name, aistart, cfg.slitratios, cfg.alphacritical,\
              scan, [0, 1, 0, 1])

            img_data_list.append(img_data)
            ais.append(my_ai)

        ivqbins = cfg.shapeqi[1]
        mg = MultiGeometry(ais, unit=unit_tth_ip,
                           wavelength=experiment.incident_wavelength,
                           radial_range=(
                               cfg.radrange[0], cfg.radrange[1]))
        result1d = mg.integrate1d(img_data_list, ivqbins)
        q_from_theta = [experiment.calcq(
            val, experiment.incident_wavelength) for val in result1d.radial]
        # theta_from_q= [experiment.calctheta(val, experiment.incident_wavelength) \
        # for val in result1d.radial]

        totaloutqi[0] += result1d.sum_signal
        totaloutqi[1] = q_from_theta
        totaloutqi[2] = result1d.radial

        totaloutcounts[0] += result1d.count  # [int(val) for val in I>0]
        totaloutcounts[1] = q_from_theta
        totaloutcounts[2] = result1d.radial  # theta_from_q#
    with lock:
        INTENSITY_ARRAY[0] += totaloutqi[0]
        INTENSITY_ARRAY[1:] = totaloutqi[1:]
        COUNT_ARRAY[0] += totaloutcounts[0]
        COUNT_ARRAY[1:] = totaloutcounts[1:]


def pyfai_move_exitangles_worker(
        experiment, imageindices, scan, process_config) -> None:
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
            unit_qip, unit_qoop, img_data, my_ai, ai_limits = \
             get_pyfai_components(
                experiment, i, sample_orientation, unit_qip_name,\
                unit_qoop_name, aistart, cfg.slitratios, cfg.alphacritical, \
                scan, cfg.anglimits)

            img_data_list.append(img_data)
            ais.append(my_ai)

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


# ====static detector processing
def pyfai_stat_exitangles_worker(
        experiment, imageindex, scan, process_config: SimpleNamespace) -> None:
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

    unit_qip, unit_qoop, img_data, my_ai, ai_limits = get_pyfai_components(
        experiment, index, sample_orientation, unit_qip_name, unit_qoop_name,
        aistart, cfg.slitratios, cfg.alphacritical, scan, cfg.anglimits)

    map2d = my_ai.integrate2d(img_data, cfg.qmapbins[0], cfg.qmapbins[1], \
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

    unit_qip, unit_qoop, img_data, my_ai, ai_limits = get_pyfai_components(
        experiment, index, sample_orientation, unit_qip_name,
        unit_qoop_name, aistart, cfg.slitratios, cfg.alphacritical, scan, cfg.qlimits)

    map2d = my_ai.integrate2d(img_data, cfg.qmapbins[0], cfg.qmapbins[1],\
        unit=(unit_qip, unit_qoop), radial_range=(ai_limits[0], ai_limits[1]),\
    azimuth_range=(ai_limits[2], ai_limits[3]), method=("no", "csr", "cython"))
    mapaxisinfo = [map2d.azimuthal, map2d.radial, str(
        map2d.azimuthal_unit), str(map2d.radial_unit)]
    return map2d[0], map2d[1], map2d[2], mapaxisinfo


def pyfai_stat_ivsq_worker(experiment, imageindex, scan,
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
    unit_q_tot, unit_qoop, img_data, my_ai, ai_limits = get_pyfai_components(
        experiment, index, sample_orientation, unit_qip_name,
        unit_qoop_name, aistart, cfg.slitratios, cfg.alphacritical, scan, [0, 1, 0, 1])

    tth, intensity = my_ai.integrate1d_ng(img_data,
                                          cfg.ivqbins,
                                          unit="2th_deg", polarization_factor=1)
    qvals = [experiment.calcq(tthval, experiment.incident_wavelength)
             for tthval in tth]

    return intensity, tth, qvals


def pyfai_static_exitangles(experiment, hf, scan,
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


def pyfai_static_qmap(experiment, hf, scan, process_config: SimpleNamespace):
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
        experiment.do_savetiffs(hf, outlist[0], outlist[1], outlist[2])
    save_config_variables(hf, cfg)
    hf.close()


def pyfai_static_ivsq(experiment, hf, scan, process_config: SimpleNamespace):
    """
    calculate Intensity Vs Q 1d profile from static detector scan
    """
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    cfg = process_config
    cfg.qlimits, cfg.scanlength, cfg.scanlistnew = \
     pyfai_setup_limits(experiment, scan, experiment.calcqlim, cfg.slitratios)

    # calculate map bins if not specified using resolution of 0.01 degrees
    if cfg.qmapbins == 0:
        cfg.qmapbins = get_qmapbins(cfg.qlimits, experiment)
    cfg.scalegamma = 1

    print(f'starting process pool with num_threads={cfg.num_threads}')
    all_ints = []
    all_two_ths = []
    all_qs = []
    cfg.multi = False
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
        experiment.do_savedats(hf, outlist[0], outlist[1], outlist[2])
    save_config_variables(hf, cfg)
    hf.close()
