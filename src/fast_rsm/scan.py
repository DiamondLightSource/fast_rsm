"""
This module contains the scan class, that is used to store all of the
information relating to a reciprocal space scan.
"""

# pylint: disable=protected-access

import time
from multiprocessing.pool import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Lock
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np

from diffraction_utils import I07Nexus, I10Nexus, Vector3, Frame
from diffraction_utils.diffractometers import \
    I10RasorDiffractometer, I07Diffractometer

from .binning import finite_diff_shape, weighted_bin_3d
from .image import Image
from .rsm_metadata import RSMMetadata


def init_process_pool(lock: Lock):
    """
    Initializes a processing pool to have a global shared lock.
    """
    # pylint: disable=global-variable-undefined.

    # Make a global lock for the shared memory block used in parallel code.
    global LOCK
    LOCK = lock


def _on_exit(shared_mem: SharedMemory):
    """
    Cane be used with the atexit module. Makes sure that the shared memory is
    cleaned when called.
    """
    try:
        shared_mem.close()
        shared_mem.unlink()
        print("Had to emergency unlink shared memory.")
    except FileNotFoundError:
        # The file has already been unlinked; do nothing.
        pass


def _chunks(lst, num_chunks):
    """Split lst into num_chunks almost evenly sized chunks."""
    chunk_size = int(len(lst)/num_chunks)
    if chunk_size * num_chunks < len(lst):
        chunk_size += 1
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def _bin_one_map(frame: Frame,
                 start: np.ndarray,
                 stop: np.ndarray,
                 step: np.ndarray,
                 idx: int,
                 metadata: RSMMetadata,
                 processing_steps: list
                 ) -> np.ndarray:
    """
    Calculates and bins the reciprocal space map with index idx. Saves the
    result to the shared memory buffer.
    """

    image = Image(metadata, idx)
    image._processing_steps = processing_steps
    # Do the mapping for this image; bin the mapping.
    q_vectors = image.q_vectors(frame)
    binned_q = weighted_bin_3d(q_vectors,
                               image.data,
                               start,
                               stop,
                               step)
    return binned_q


def _bin_maps_with_indices(indices: List[int],
                           frame: Frame,
                           start: np.ndarray,
                           stop: np.ndarray,
                           step: np.ndarray,
                           metadata: RSMMetadata,
                           processing_steps: list
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
        binned_q = np.zeros(shape=finite_diff_shape(start, stop, step))

        # Do the binning, adding each binned dataset to binned_q.
        for idx in indices:
            binned_q += _bin_one_map(frame, start, stop, step, idx, metadata,
                                     processing_steps)

        # Now we've finished binning, add this to the final shared data array.
        shared_mem = SharedMemory(name='arr')
        shape = finite_diff_shape(start, stop, step)
        final_data = np.ndarray(shape, dtype=np.float64, buffer=shared_mem.buf)

        # Update the final data in a thread safe way.
        with LOCK:
            final_data += binned_q

        # Always do your chores.
        shared_mem.close()

    except Exception as exception:
        print(f"Exception thrown in bin_one_map: \n{exception}")


class Scan:
    """
    This class stores all of the data and metadata relating to a reciprocal
    space map.

    Attrs:
        metadata:
            Scan metadata.
        load_image:
            A Callable that takes an index as an argument and returns an
            instance of Image with that corresponding index.
    """

    def __init__(self, metadata: RSMMetadata):
        self.metadata = metadata
        self._processing_steps = []

    def add_processing_step(self, function) -> None:
        """
        Adds the processing step to the processing pipeline.

        Args:
            function:
                A function that takes a numpy array as an argument, and returns
                a numpy array.
        """
        self._processing_steps.append(function)

    def binned_reciprocal_space_map(
        self,
        frame: Frame,  # The frame in which we'll do the mapping.
        start: np.ndarray,  # Bin start.
        stop: np.ndarray,  # Bin stop.
        step: np.ndarray,  # Bin step.
        num_threads: int = 1  # How many threads to use for this map.
    ) -> np.ndarray:
        """
        Runs a reciprocal space map, but bins image by image. All of start,
        stop and step are numpy arrays with shape (3) for [xstart, ystart,
        zstart] etc.

        Args:
            frame:
                The frame in which we want to carry out the map.
            start:
                Where to start our finite differences binning grid. This should
                be an array-like object [startx, starty, startz].
            stop:
                Where to stop our finite differences binning grid. This should
                be an array-like object [stopx, stopy, stopz].
            step:
                Step size for our finite differences binning grid. This should
                be an array-like object [stepx, stepy, stepz].
            num_threads:
                How many threads to use for this calculation. Defaults to 1.
        """
        # Lets prevent complicated errors from showing up somewhere deeper.
        start, stop, step = np.array(start), np.array(stop), np.array(step)

        # Also, make sure that we start our scan at scan index 0.
        frame.scan_index = 0
        # Make sure that our scan has the correct diffractometer associated.
        frame.diffractometer = self.metadata.diffractometer

        # Prepare an array with the same shape as our final binned data array.
        shape = finite_diff_shape(start, stop, step)
        # Note that this doesn't initialise the array; arr is nonsense.
        arr = np.ndarray(shape=shape)

        # Make sure we never leak this memory more than once.
        try:
            shm = SharedMemory('arr', size=100)
            shm.close()
            shm.unlink()
            print("Had to unlink leaked shared memory before starting...")
        except FileNotFoundError:
            # Don't do anything if we couldn't open 'arr'.
            pass

        # Make a shared memory block for final_data; initialize it.
        shared_mem = SharedMemory('arr', create=True, size=arr.nbytes)

        # Now hook final_data up to the shared_mem buffer that we just made.
        final_data = np.ndarray(shape, dtype=arr.dtype,
                                buffer=shared_mem.buf)
        # Set the array to be full of zeros.
        final_data.fill(0)

        # A pool-less single threaded approach.
        if num_threads == 1:
            print("Using single threaded routine.")
            final_data = np.zeros(shape)
            for i in range(self.metadata.data_file.scan_length):
                print(f"Processing image {i}...")
                img = self.load_image(i)
                img._processing_steps = self._processing_steps
                time_1 = time.time()
                q_vectors = img.q_vectors(frame)
                time_taken = time.time() - time_1
                print(f"Mapping time: {time_taken}")

                time_1 = time.time()
                final_data += weighted_bin_3d(
                    q_vectors,
                    img.data,
                    start, stop, step)
                time_taken = time.time() - time_1
                print(f"Binning, img.data & final_data+= time: {time_taken}")
            final_data = np.copy(final_data)
            shared_mem.close()
            shared_mem.unlink()
            return final_data

        # If execution reaches here, we want a multithreaded binned RSM using
        # a thread pool. For this, we'll need a lock to share between processes.
        lock = Lock()

        # The high performance approach.
        with Pool(
            processes=num_threads,  # The size of our pool.
            initializer=init_process_pool,  # Our pool's initializer.
            initargs=(lock,)  # The initializer makes this lock global.
        ) as pool:
            # Submit all of the image processing functions to the pool as jobs.
            async_results = []
            for indices in _chunks(list(range(
                    self.metadata.data_file.scan_length)), num_threads):
                async_results.append(pool.apply_async(
                    _bin_maps_with_indices,
                    (indices, frame, start, stop, step,
                     self.metadata, self._processing_steps)))

            # Wait for all the work to complete.
            for result in async_results:
                result.wait()
                if not result.successful():
                    raise ValueError(
                        "Could not carry out map for an unknown reason.")

        # Close the shared memory pool; return the final data.
        final_data = np.copy(final_data)
        shared_mem.close()
        shared_mem.unlink()
        _on_exit(shared_mem)
        return final_data

    def reciprocal_space_map(self, frame: Frame, num_threads: int = 1):
        """
        Don't use this unless you understand what you're doing. Use the binned
        reciprocal space map method, your computer will thank you.

        Calculates the scan's reciprocal space map, without binning. I hope you
        have a *LOT* of RAM.

        Args:
            frame:
                The frame of reference in which we want to carry out the
                reciprocal space map.
        """
        if num_threads != 1:
            raise NotImplementedError(
                "Reciprocal space maps are currently single threaded only.")

        q_vectors = []
        # Load images one by one.
        for idx in range(self.metadata.data_file.scan_length):
            image = Image(self.metadata, idx)
            # Do the mapping for this image in correct frame.
            q_vectors.append(image.q_vectors(frame))

        return q_vectors

    def load_image(self, idx: int):
        """
        Convenience method for loading a single image. This is unpicklable.
        """
        return Image(self.metadata, idx)

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

    @classmethod
    def from_i07(cls,
                 path_to_nx: Union[str, Path],
                 beam_centre: Tuple[int],
                 detector_distance: float,
                 setup: str,
                 sample_oop: Union[Vector3, np.ndarray, List[float]],
                 path_to_data: str = ''):
        """
        Instantiates a Scan from the path to an I07 nexus file, a beam centre
        coordinate tuple, a detector distance and a sample out-of-plane vector.

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
            sample_oop:
                An instance of a diffraction_utils Vector3 which descrbes the
                sample out of plane vector.
            path_to_data:
                Path to the directory in which the images are stored. Defaults
                to '', in which case a bunch of reasonable directories will be
                searched for the images.
        """
        # Load the nexus file.
        i07_nexus = I07Nexus(path_to_nx, path_to_data,
                             detector_distance, setup)

        if not isinstance(sample_oop, Frame):
            frame = Frame(Frame.sample_holder, None, None)
            sample_oop = Vector3(sample_oop, frame)

        # Load the state of the diffractometer; prepare the RSM metadata.
        diff = I07Diffractometer(i07_nexus, sample_oop, setup)
        metadata = RSMMetadata(diff, beam_centre)

        # Make sure that the sample_oop vector's frame's diffractometer is good.
        sample_oop.frame.diffractometer = diff

        return cls(metadata)
