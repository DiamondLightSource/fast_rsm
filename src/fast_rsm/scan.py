"""
This module contains the scan class, that is used to store all of the
information relating to a reciprocal space scan.
"""

# pylint: disable=protected-access
# pylint: disable=global-statement



import copy
import traceback
from multiprocessing import current_process
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Lock
from pathlib import Path
from typing import Union, Tuple, List, Dict
import logging
from pyFAI.multi_geometry import MultiGeometry
import pyFAI
from pyFAI import units

import numpy as np

from diffraction_utils import  Frame #I10Nexus, Vector3,
#from diffraction_utils.diffractometers import I10RasorDiffractometer

import fast_rsm.io as io
from fast_rsm.binning import weighted_bin_3d
from fast_rsm.image import Image
from fast_rsm.rsm_metadata import RSMMetadata
from fast_rsm.writing import linear_bin_to_vtk

logger = logging.getLogger("fastrsm")


lock=None
RSM_ARRAY=None
COUNT_ARRAY=None
SHM_RSM=None
SHM_COUNT=None
METADATA=None
NUM_THREADS=None
FRAME=None
OUTPUT_FILE_NAME=None
SHM_INTENSITY=None
INTENSITY_ARRAY=None

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
    chunk_size = int(len(lst) / num_chunks)
    if chunk_size * num_chunks < len(lst):
        chunk_size += 1
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def _chunk_indices(array: np.ndarray, num_chunks: int) -> tuple:
    """
    Yield num_chunks (N) tuples of incides (a, b) such that array[a0:b0],
    array[a1:b1], ..., array[aN:bN] spans the entire array.
    """
    chunk_size = int(len(array) / num_chunks)
    if chunk_size * num_chunks < len(array):
        chunk_size += 1
    for i in range(0, len(array), chunk_size):
        yield i, i + chunk_size

# ==========testing functions========


def bin_maps_with_indices_smm(indices: List[int],
                              start: np.ndarray,
                              stop: np.ndarray,
                              step: np.ndarray,
                              min_intensity: float,
                              processing_steps: list,
                              skip_images: List[int],
                              oop: str,
                              spherical_bragg_vec: np.array,
                              map_each_image: bool = False,
                              previous_images: int = 0,
                              ) -> None:
    """
    Bins all of the maps with indices in indices. The purpose of this
    intermediate function call is to decrease the amount of context switching/
    serialization that the interpreter has to do.
    """

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
            _bin_one_map_smm(start, stop, step, min_intensity, idx,
                processing_steps, oop, spherical_bragg_vec, map_each_image, previous_images)
    except Exception as exception:
        print("Exception thrown in bin_one_map:")
        print(traceback.format_exc())
        raise exception

    # In place of returning very large arrays when they might not be wanted, we
    # just return the names of the shared memory blocks where the RSM/count data
    # is stored. The caller can then choose to access the shared memory blocks
    # directly without needlessly serializing enormous arrays.
    # return SHARED_RSM_NAME, SHARED_COUNT_NAME


def _bin_one_map_smm(start: np.ndarray,
                     stop: np.ndarray,
                     step: np.ndarray,
                     min_intensity: float,
                     idx: int,
                     processing_steps: list,
                     oop: str,
                     spherical_bragg_vec: np.array,
                     map_each_image: bool = False,
                     previous_images: int = 0,

                     ) -> np.ndarray:
    """
    Calculates and bins the reciprocal space map with index idx. Saves the
    result to the shared memory buffer.
    """

    if map_each_image:
        rsm_before = np.copy(RSM_ARRAY)
        count_before = np.copy(COUNT_ARRAY)

    image = Image(METADATA, idx)
    image._processing_steps = processing_steps

    # Do the mapping for this image; bin the mapping.
    q_vectors = image.q_vectors(FRAME, spherical_bragg_vec, oop=oop)
    weighted_bin_3d(q_vectors,
                    image.data,
                    RSM_ARRAY,
                    COUNT_ARRAY,
                    start,
                    stop,
                    step,
                    min_intensity)

    if map_each_image:
        # If the user also wants us to map each image, rerun the map for just
        # this image.
        rsm = RSM_ARRAY - rsm_before
        count = COUNT_ARRAY - count_before

        # Normalise it.
        normalised_map = rsm / (count.astype(np.float32))

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
            spherical_bragg_vec,
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


def rsm_init_worker(l, shm_rsm_name: str, shm_counts_name: str, shmshape: np.ndarray,\
                    metadata: RSMMetadata,newmetadata: dict,\
                    motors: Dict[str, np.ndarray], num_threads: int, \
                    frame: Frame, output_file_name: str = None):
    """
    initialiser for reciprocal space mapping
    """

    global lock
    global RSM_ARRAY
    global COUNT_ARRAY
    global SHM_RSM
    global SHM_COUNT
    global METADATA

    global NUM_THREADS
    global FRAME
    global OUTPUT_FILE_NAME

    OUTPUT_FILE_NAME = output_file_name
    NUM_THREADS = num_threads
    FRAME = frame

    METADATA = metadata
    METADATA.update_i07_nx(motors, newmetadata)

    lock = l
    SHM_RSM = SharedMemory(name=shm_rsm_name)
    RSM_ARRAY = np.ndarray(shmshape, dtype=np.float32, buffer=SHM_RSM.buf)
    SHM_COUNT = SharedMemory(name=shm_counts_name)
    COUNT_ARRAY = np.ndarray(shmshape, dtype=np.uint32, buffer=SHM_COUNT.buf)

# ============================

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

    def q_bounds(self, frame: Frame, spherical_bragg_vec: np.array,
                 oop: str = 'y') -> Tuple[np.ndarray]:
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
        q_vec = img.q_vectors(frame, spherical_bragg_vec,
                              indices=poni, oop=oop)

        start, stop = q_vec, q_vec

        # Iterate over every image in the scan.
        for i in range(self.metadata.data_file.scan_length):
            # Instantiate an image without loading its data.
            img = self.load_image(i, load_data=False)

            # Work out all the extreme q values for this image.
            q_vecs = img.q_vectors(
                frame, spherical_bragg_vec, indices=extremal_q_points, oop=oop)

            # Get the min/max of each component.
            min_q = np.array([np.amin(q_vecs[:, i]) for i in range(3)])
            max_q = np.array([np.amax(q_vecs[:, i]) for i in range(3)])

            # Update start/stop accordingly.
            start = [min_q[x] if min_q[x] < start[x] else start[x]
                     for x in range(3)]
            stop = [max_q[x] if max_q[x] > stop[x] else stop[x]
                    for x in range(3)]
        start, stop = np.array(start), np.array(stop)
        # adjust start,stop,step if frame is in spherical polar co-ordinates
        if frame.coordinates == Frame.sphericalpolar:
            # calculate vector radius
            maxradius = np.max(np.array(
                [np.linalg.norm(q_vecs[i, :]) for i in range(len(extremal_q_points))]))
            start = [0, 0, -np.pi]
            stop = [maxradius, np.pi, np.pi]
            return start, stop
        side_lengths = stop - start
        padding = side_lengths / 20
        start -= padding
        stop += padding
        return start, stop

    # def from_i10(cls,
    #              path_to_nx: Union[str, Path],
    #              beam_centre: Tuple[int],
    #              detector_distance: float,
    #              sample_oop: Vector3,
    #              path_to_data: str = ''):
    #     """
    #     Instantiates a Scan from the path to an I10 nexus file, a beam centre
    #     coordinate, a detector distance (this isn't stored in i10 nexus files)
    #     and a sample out-of-plane vector.

    #     Args:
    #         path_to_nx:
    #             Path to the nexus file containing the scan metadata.
    #         beam_centre:
    #             A (y, x) tuple of the beam centre, measured in the usual image
    #             coordinate system, in units of pixels.
    #         detector_distance:
    #             The distance between the sample and the detector, which cant
    #             be stored in i10 nexus files so needs to be given by the user.
    #         sample_oop:
    #             An instance of a diffraction_utils Vector3 which descrbes the
    #             sample out of plane vector.
    #         path_to_data:
    #             Path to the directory in which the images are stored. Defaults
    #             to '', in which case a bunch of reasonable directories will be
    #             searched for the images.
    #     """
    #     # Load the nexus file.
    #     i10_nexus = I10Nexus(path_to_nx, path_to_data, detector_distance)

    #     # Load the state of the RASOR diffractometer; prepare the metadata.
    #     diff = I10RasorDiffractometer(i10_nexus, sample_oop, 'area')
    #     meta = RSMMetadata(diff, beam_centre)

    #     # Make sure the sample_oop vector's frame's diffractometer is correct.
    #     sample_oop.frame.diffractometer = diff

    #     return cls(meta)

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
