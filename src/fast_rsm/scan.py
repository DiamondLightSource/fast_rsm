"""
This module contains the scan class, that is used to store all of the
information relating to a reciprocal space scan.
"""

# pylint: disable=protected-access

import traceback
from multiprocessing import current_process
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Lock
from pathlib import Path
from typing import Union, Tuple, List, Dict

import numpy as np

from diffraction_utils import I10Nexus, Vector3, Frame
from diffraction_utils.diffractometers import I10RasorDiffractometer

from . import io
from .binning import weighted_bin_3d
from .image import Image
from .rsm_metadata import RSMMetadata
from .writing import linear_bin_to_vtk

from pyFAI.multi_geometry import MultiGeometry
import pyFAI
import copy



def check_shared_memory(shared_mem_name: str) -> None:
    """
    Make sure that a shared memory array is not open. Clear the shared memory
    and print a warning if it is open.

    Args:
        shared_mem_name:
            Name of the shared memory array to check.
    """
    # Make sure that we don't leak this memory more than once.
    try:
        shm = SharedMemory(shared_mem_name,
                           size=100)  # Totally arbitrary number.
        shm.close()
        shm.unlink()
        print(f"Had to unlink *leaked* shared memory '{shared_mem_name}'...")
    except FileNotFoundError:
        # Don't do anything if we couldn't open 'arr'.
        pass


def init_process_pool(
        locks: List[Lock],
        num_threads: int,
        metadata: RSMMetadata,
        frame: Frame,
        shape: tuple,
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
            Passed if you want to make RSM and count arrays global.
    """
    # pylint: disable=global-variable-undefined.

    # Make a global lock for the shared memory block used in parallel code.
    global LOCKS

    # Some metadata that a worker thread should always have access to.
    global NUM_THREADS
    global METADATA
    global FRAME

    # Not always necessary and may be set to None.
    global OUTPUT_FILE_NAME

    # These are numpy arrays whose buffer corresponds to the shared memory
    # buffer. It's more convenient to access these later than to directly work
    # with the shared memory buffer.
    global RSM
    global COUNT

    # We want to keep track of what we've called our shared memory arrays.
    global SHARED_RSM_NAME
    global SHARED_COUNT_NAME

    # Why do we need to make the shared memory blocks global, if we're giving
    # global access to them via the numpy 'RSM' and 'COUNT' arrays? The answer
    # is that we need the shared memory arrays to remain in scope, or they'll be
    # freed.
    global SHARED_RSM
    global SHARED_COUNT

    LOCKS = locks
    NUM_THREADS = num_threads
    METADATA = metadata
    FRAME = frame

    OUTPUT_FILE_NAME = output_file_name

    # Work out how many bytes we're going to need by making a dummy array.
    arr = np.ndarray(shape=shape, dtype=np.float32)

    # Construct the shared memory buffers.
    SHARED_RSM_NAME = f'rsm_{current_process().name}'
    SHARED_COUNT_NAME = f'count_{current_process().name}'
    check_shared_memory(SHARED_RSM_NAME)
    check_shared_memory(SHARED_COUNT_NAME)
    SHARED_RSM = SharedMemory(
        name=SHARED_RSM_NAME, create=True, size=arr.nbytes)
    SHARED_COUNT = SharedMemory(
        name=SHARED_COUNT_NAME, create=True, size=arr.nbytes)

    # Construct the global references to the shared memory arrays.
    RSM = np.ndarray(shape, dtype=np.float32, buffer=SHARED_RSM.buf)
    COUNT = np.ndarray(shape, dtype=np.uint32, buffer=SHARED_COUNT.buf)

    # Initialize the shared memory arrays.
    RSM.fill(0)
    COUNT.fill(0)

    print(f"Finished initializing worker {current_process().name}.")

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
    arrcake =np.ndarray(shape=shapecake, dtype=np.float32)
    arrqpqpmap=np.ndarray(shape=shapeqpqpmap, dtype=np.float32)

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
    PYFAI_QI = np.ndarray(shapeqi, dtype=np.float32, buffer=SHARED_PYFAI_QI.buf)
    CAKE = np.ndarray(shapecake, dtype=np.float32, buffer=SHARED_CAKE.buf)
    QPQPMAP  = np.ndarray(shapeqpqpmap, dtype=np.float32, buffer=SHARED_QPQPMAP.buf)

    # Initialize the shared memory arrays.
    PYFAI_QI.fill(0)
    CAKE.fill(0)
    QPQPMAP.fill(0)

    print(f"Finished initializing worker {current_process().name}.")
    


def pyfai_qmap_qvsI(experiment,imageindex,scan,two_theta_start,pyfaiponi,qmapbins,ivqbins)->None:
    index=imageindex
    aistart=pyFAI.load(pyfaiponi)     
    unit_qip = "qip_A^-1"
    unit_qoop = "qoop_A^-1"
    
    qlimhor=experiment.calcqlim( 'hor')
    qlimver=experiment.calcqlim( 'vert')
    qlimits=[qlimhor[0],qlimhor[1],qlimver[0],qlimver[1]]

    my_ai = copy.deepcopy(aistart)
    if np.size(experiment.gammadata)>1:  
        my_ai.rot1 = np.radians(-two_theta_start[index])
    if np.size(experiment.deltadata)>1:
        my_ai.rot2 =-np.radians(experiment.deltadata[index])

    #my_ai.poni1=(experiment.imshape[0] - experiment.beam_centre[0]) * experiment.pixel_size
   # print(my_ai.get_config())
    #img_data=np.flipud(scan.load_image(index).data)
    img_data=scan.load_image(index).data
   # print(qlimits)
    map2d = my_ai.integrate2d(img_data, qmapbins[0],qmapbins[1], unit=(unit_qip, unit_qoop),radial_range=(qlimits[0]*1,qlimits[1]*1),azimuth_range=(10.5*qlimits[2],10.5*qlimits[3]), method=("no", "csr", "cython"))
    
    tth,I = my_ai.integrate1d_ng(img_data,
                            ivqbins,
                            unit="2th_deg",polarization_factor=1)
    

    Q,I = my_ai.integrate1d_ng(img_data,
                        ivqbins,
                       unit="q_A^-1",polarization_factor=1)
    
    
    return map2d[0],map2d[1],map2d[2],I,tth,Q
  
def pyfaicalcint(experiment,imageindices,scan,shapecake,shapeqi,shapeqpqp,two_theta_start,pyfaiponi,radrange,radstepval,qmapbins)->None:
    choiceims=imageindices
    #ais=[]
   
    aistart=pyFAI.load(pyfaiponi)    
    #aistart.set_mask(scan.metadata.edfmask)
    totaloutqi=np.zeros((3,shapeqi[2]))
    totaloutcounts=np.zeros((3,shapeqi[2]))
    
    totalcake=np.zeros(shapecake[1:])
    totalcount=np.zeros(shapecake[1:])
    
    totalqpqpmap=np.zeros(shapeqpqp[1:])
    totalqpqpcounts=np.zeros(shapeqpqp[1:])
    unit_qip = "qip_A^-1"
    unit_qoop = "qoop_A^-1"

    runningtotal=0
    groupnum=15
    qlimhor=experiment.calcqlim( 'hor')
    qlimver=experiment.calcqlim( 'vert')
    qlimits=[qlimhor[0],qlimhor[1],qlimver[0],qlimver[1]]
    

    groups=[choiceims[i:i+groupnum] for i in range(0,len(choiceims),groupnum)]
    for group in groups:
        ais=[]
        for i in group:
            my_ai = copy.deepcopy(aistart)
            if np.size(experiment.gammadata)>1:  
                my_ai.rot1 = np.radians(-two_theta_start[i])
            if np.size(experiment.deltadata)>1:
                my_ai.rot2 =-np.radians(experiment.deltadata[i])
            ais.append(my_ai)
        #print(f'loading image {i}')
        img_data=[np.flipud(scan.load_image(i).data) for i in group]
        
        for current_n,current_ai  in enumerate(ais):
            current_img=img_data[current_n]
            map2d = current_ai.integrate2d(current_img, qmapbins[0],qmapbins[1], unit=(unit_qip, unit_qoop),
                                   radial_range=(qlimits[0]*1.05,qlimits[1]*1.05),azimuth_range=(10.5*qlimits[2],10.5*qlimits[3]), method=("no", "csr", "cython"))
            totalqpqpmap+=map2d.sum_signal
            totalqpqpcounts+=map2d.count
            #print(np.max(map2d.sum_signal))
        
        
        nbins=int(np.ceil((radrange[1]-radrange[0])/radstepval))
        Qrange=(experiment.calcq(radrange[0],experiment.incident_wavelength),\
                experiment.calcq(radrange[1],experiment.incident_wavelength))

        mg = MultiGeometry(ais,  unit="2th_deg",wavelength=experiment.incident_wavelength,radial_range=(0,75))
        result2d=mg.integrate2d(img_data, shapecake[2],shapecake[1])

        mg = MultiGeometry(ais,  unit="2th_deg",wavelength=experiment.incident_wavelength,radial_range=radrange)

        result1d= mg.integrate1d(img_data, nbins)
        two_theta_arr=result1d.radial
        I_arr=result1d.sum_signal
        I_counts=result1d.count
        mg= MultiGeometry(ais,  unit= "q_A^-1",wavelength=experiment.incident_wavelength,radial_range=Qrange)
        result1d= mg.integrate1d(img_data, nbins)        
        q_arr=result1d[0]
        qmin,qmax=q_arr.min(),q_arr.max()
        thmin,thmax=two_theta_arr.min(),two_theta_arr.max()

        
        
        totaloutqi[0]+=I_arr
        totaloutqi[1]=q_arr
        totaloutqi[2]=two_theta_arr
        
        totaloutcounts[0]+=I_counts
        totaloutcounts[1]=1
        totaloutcounts[2]=1
        
        
        cakeout_array=np.zeros(np.shape(result2d[0]))
        cakeout_array[0:int(shapecake[1]/2),:]=result2d[0][int(shapecake[1]/2):shapecake[1],:]
        cakeout_array[int(shapecake[1]/2):shapecake[1],:]=result2d[0][0:int(shapecake[1]/2),:]
        countresult=np.array(cakeout_array>0).astype(int)


        
        totalcake+=cakeout_array
        totalcount+=countresult

    cakeaxisinfo=[result2d.azimuthal,result2d.radial,str(result2d.azimuthal_unit),str(result2d.radial_unit)]
    mapaxisinfo= [-1*map2d.azimuthal,map2d.radial,str(map2d.azimuthal_unit),str(map2d.radial_unit)]   

    PYFAI_QI[0]=totaloutqi
    PYFAI_QI[1]=totaloutcounts
    CAKE[0] = totalcake
    CAKE[1]=totalcount
    QPQPMAP[0]=totalqpqpmap
    QPQPMAP[1]=totalqpqpcounts
    return SHARED_PYFAI_QI_NAME, SHARED_CAKE_NAME , SHARED_QPQPMAP_NAME, cakeaxisinfo, mapaxisinfo

def _on_exit(shared_mem: SharedMemory) -> None:
    """
    Can be used with the atexit module. Makes sure that the shared memory is
    cleaned when called.

    Args:
        shared_mem:
            The shared memory to close/unlink.
    """
    try:
        shared_mem.close()
        shared_mem.unlink()
    except FileNotFoundError:
        # The file has already been unlinked; do nothing.
        pass


def chunk(lst, num_chunks):
    """
    Split lst into num_chunks almost evenly sized chunks. Algorithm lifted from
    stackoverflow almost without change (but isn't everything, in the end...)

    Args:
        lst:
            The iterable to split up.
        num_chunks:
            The number of almost-evenly-sized chunks to split lst into. Note
            that the last chunk can be decently smaller.

    Yields:
        the chunks. Note that this yields as a generator, it does not return.
    """
    chunk_size = int(len(lst)/num_chunks)
    if chunk_size * num_chunks < len(lst):
        chunk_size += 1
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def _chunk_indices(array: np.ndarray, num_chunks: int) -> tuple:
    """
    Yield num_chunks (N) tuples of incides (a, b) such that array[a0:b0],
    array[a1:b1], ..., array[aN:bN] spans the entire array.
    """
    chunk_size = int(len(array)/num_chunks)
    if chunk_size * num_chunks < len(array):
        chunk_size += 1
    for i in range(0, len(array), chunk_size):
        yield i, i+chunk_size


def _bin_one_map(start: np.ndarray,
                 stop: np.ndarray,
                 step: np.ndarray,
                 min_intensity: float,
                 idx: int,
                 processing_steps: list,
                 oop: str,
                 map_each_image: bool = False,
                 previous_images: int = 0
                 ) -> np.ndarray:
    """
    Calculates and bins the reciprocal space map with index idx. Saves the
    result to the shared memory buffer.
    """
    if map_each_image:
        rsm_before = np.copy(RSM)
        count_before = np.copy(COUNT)

    image = Image(METADATA, idx)
    image._processing_steps = processing_steps

    # Do the mapping for this image; bin the mapping.
    q_vectors = image.q_vectors(FRAME, oop=oop)
    weighted_bin_3d(q_vectors,
                    image.data,
                    RSM,
                    COUNT,
                    start,
                    stop,
                    step,
                    min_intensity)

    if map_each_image:
        # If the user also wants us to map each image, rerun the map for just
        # this image.
        rsm = RSM - rsm_before
        count = COUNT - count_before

        # Normalise it.
        normalised_map = rsm/(count.astype(np.float32))

        # Get the unique id for this image & pad with zeros for paraview.
        image_id = str(previous_images + idx).zfill(6)

        # Now we just need to save this map; work out its unique name.
        volume_path = str(OUTPUT_FILE_NAME) + '_' + str(image_id)

        # Save the vtk, as well as a .npy and a bounds file.
        linear_bin_to_vtk(normalised_map, volume_path, start, stop, step)
        np.save(volume_path, normalised_map)
        np.savetxt(str(volume_path) + "_bounds.txt",
                   np.array((start, stop, step)).transpose(),
                   header="start stop step")

        q_vec_path = volume_path + "_q"
        intensities_path = volume_path + "_uncorrected_intensities"
        corrected_intensity_path = volume_path + "_corrected_intensities"

        # Re-calculate q-vectors. Don't apply corrections (if users want
        # per-image data, they likely want control over corrections. This is
        # especially true if people want to use this data to project to 1D,
        # where the application of custom corrections is particularly easy).
        fresh_image = Image(METADATA, idx)
        q_vectors = fresh_image.q_vectors(
            FRAME,
            oop=oop,
            lorentz_correction=False,
            pol_correction=False)

        # Also, just to provide complete information on a per-image basis, save
        # every single *exact* q-vector for this scan.
        # These should both be saved.
        np.save(q_vec_path, q_vectors.ravel())
        np.save(intensities_path, fresh_image.data.ravel())

        # Also save the corrected intensities.
        np.save(corrected_intensity_path, image.data.ravel())


def bin_maps_with_indices(indices: List[int],
                          start: np.ndarray,
                          stop: np.ndarray,
                          step: np.ndarray,
                          min_intensity: float,
                          motors: Dict[str, np.ndarray],
                          metadata: dict,
                          processing_steps: list,
                          skip_images: List[int],
                          oop: str,
                          map_each_image: bool = False,
                          previous_images: int = 0
                          ) -> None:
    """
    Bins all of the maps with indices in indices. The purpose of this
    intermediate function call is to decrease the amount of context switching/
    serialization that the interpreter has to do.
    """
    METADATA.update_i07_nx(motors, metadata)

    # We need to catch all exceptions and explicitly print them in worker
    # threads.
    # pylint: disable=broad-except.
    try:
        # Do the binning, adding each binned dataset to binned_q.
        for idx in indices:
            # Skip this if we've been asked to.
            if idx in skip_images:
                continue

            # print(f"Processing image {idx}. ", end='')
            _bin_one_map(start, stop, step, min_intensity, idx,
                         processing_steps, oop, map_each_image, previous_images)
    except Exception as exception:
        print("Exception thrown in bin_one_map:")
        print(traceback.format_exc())
        raise exception

    # In place of returning very large arrays when they might not be wanted, we
    # just return the names of the shared memory blocks where the RSM/count data
    # is stored. The caller can then choose to access the shared memory blocks
    # directly without needlessly serializing enormous arrays.
    return SHARED_RSM_NAME, SHARED_COUNT_NAME


class Scan:
    """
    This class stores all of the data and metadata relating to a reciprocal
    space map.

    Attrs:
        metadata:
            Scan metadata.
        skip_images:
            Image indices to be skipped in a reciprocal space map.
        load_image:
            A Callable that takes an index as an argument and returns an
            instance of Image with that corresponding index.
    """

    def __init__(self, metadata: RSMMetadata, skip_images: List[int] = None):
        self.metadata = metadata

        if isinstance(skip_images, int):
            skip_images = [skip_images]
        self.skip_images = [] if skip_images is None else skip_images

        self._processing_steps = []

    @property
    def processing_steps(self):
        """Left as a property to give the option to deepcopy in the future."""
        return self._processing_steps

    def add_processing_step(self, function) -> None:
        """
        Adds the processing step to the processing pipeline.

        Args:
            function:
                A function that takes a numpy array as an argument, and returns
                a numpy array.
        """
        self._processing_steps.append(function)

    def load_image(self, idx: int, load_data=True):
        """
        Convenience method for loading a single image. This is unpicklable.
        """
        return Image(self.metadata, idx, load_data)

    def q_bounds(self, frame: Frame, oop: str = 'y') -> Tuple[np.ndarray]:
        """
        Works out the region of reciprocal space sampled by this scan.

        Args:
            frame:
                The frame of reference in which we want to calculate the bounds.

        Returns:
            (start, stop), where start and stop are numpy arrays with shape (3,)
        """
        top_left = (0, 0)
        top_right = (0, -1)
        bottom_left = (-1, 0)
        bottom_right = (-1, -1)
        poni = self.metadata.beam_centre
        extremal_q_points = np.array(
            [top_left, top_right, bottom_left, bottom_right, poni])
        extremal_q_points = (extremal_q_points[:, 0], extremal_q_points[:, 1])

        # Get some sort of starting value.
        img = self.load_image(0, load_data=False)
        q_vec = img.q_vectors(frame, poni, oop)


        start, stop = q_vec, q_vec

        # Iterate over every image in the scan.
        for i in range(self.metadata.data_file.scan_length):
            # Instantiate an image without loading its data.
            img = self.load_image(i, load_data=False)

            # Work out all the extreme q values for this image.
            q_vecs = img.q_vectors(frame, extremal_q_points, oop)

            # Get the min/max of each component.
            min_q = np.array([np.amin(q_vecs[:, i]) for i in range(3)])
            max_q = np.array([np.amax(q_vecs[:, i]) for i in range(3)])

            # Update start/stop accordingly.
            start = [min_q[x] if min_q[x] < start[x] else start[x]
                     for x in range(3)]
            stop = [max_q[x] if max_q[x] > stop[x] else stop[x]
                    for x in range(3)]

        # Give a bit of wiggle room. For now, I'm using 5% padding, but this was
        # chosen arbitrarily.
        start, stop = np.array(start), np.array(stop)
        side_lengths = stop - start
        padding = side_lengths/20
        start -= padding
        stop += padding

        return start, stop

    @classmethod
    def from_i10(cls,
                 path_to_nx: Union[str, Path],
                 beam_centre: Tuple[int],
                 detector_distance: float,
                 sample_oop: Vector3,
                 path_to_data: str = ''):
        """
        Instantiates a Scan from the path to an I10 nexus file, a beam centre
        coordinate, a detector distance (this isn't stored in i10 nexus files)
        and a sample out-of-plane vector.

        Args:
            path_to_nx:
                Path to the nexus file containing the scan metadata.
            beam_centre:
                A (y, x) tuple of the beam centre, measured in the usual image
                coordinate system, in units of pixels.
            detector_distance:
                The distance between the sample and the detector, which cant
                be stored in i10 nexus files so needs to be given by the user.
            sample_oop:
                An instance of a diffraction_utils Vector3 which descrbes the
                sample out of plane vector.
            path_to_data:
                Path to the directory in which the images are stored. Defaults
                to '', in which case a bunch of reasonable directories will be
                searched for the images.
        """
        # Load the nexus file.
        i10_nexus = I10Nexus(path_to_nx, path_to_data, detector_distance)

        # Load the state of the RASOR diffractometer; prepare the metadata.
        diff = I10RasorDiffractometer(i10_nexus, sample_oop, 'area')
        meta = RSMMetadata(diff, beam_centre)

        # Make sure the sample_oop vector's frame's diffractometer is correct.
        sample_oop.frame.diffractometer = diff

        return cls(meta)

    @staticmethod
    def from_i07(path_to_nx: Union[str, Path],
                 beam_centre: Tuple[int],
                 detector_distance: float,
                 setup: str,
                 path_to_data: str = ''):
        """
        Aliases to io.from_i07.

        Args:
            path_to_nx:
                Path to the nexus file containing the scan metadata.
            beam_centre:
                A (y, x) tuple of the beam centre, measured in the usual image
                coordinate system, in units of pixels.
            detector_distance:
                The distance between the sample and the detector.
            setup:
                What was the experimental setup? Can be "vertical", "horizontal"
                or "DCD".
            path_to_data:
                Path to the directory in which the images are stored. Defaults
                to '', in which case a bunch of reasonable directories will be
                searched for the images.

        Returns:
            Corresponding instance of Scan.
        """
        return io.from_i07(path_to_nx, beam_centre, detector_distance,
                           setup, path_to_data)
