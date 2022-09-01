"""
This module contains the scan class, that is used to store all of the
information relating to a reciprocal space scan.
"""

# pylint: disable=protected-access

import time
import traceback
from multiprocessing.pool import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Lock
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np

from diffraction_utils import I07Nexus, I10Nexus, Vector3, Frame
from diffraction_utils.diffractometers import \
    I10RasorDiffractometer, I07Diffractometer

from . import io
from .binning import finite_diff_shape, weighted_bin_3d
from .image import Image
from .rsm_metadata import RSMMetadata


def check_shared_memory(shared_mem_name: str):
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


def init_process_pool(locks: List[Lock], num_threads: int):
    """
    Initializes a processing pool to have a global shared lock.
    """
    # pylint: disable=global-variable-undefined.

    # Make a global lock for the shared memory block used in parallel code.
    global LOCKS
    global NUM_THREADS

    LOCKS = locks
    NUM_THREADS = num_threads


def _on_exit(shared_mem: SharedMemory):
    """
    Can be used with the atexit module. Makes sure that the shared memory is
    cleaned when called.
    """
    try:
        shared_mem.close()
        shared_mem.unlink()
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


def _bin_one_map(frame: Frame,
                 start: np.ndarray,
                 stop: np.ndarray,
                 step: np.ndarray,
                 min_intensity: float,
                 idx: int,
                 metadata: RSMMetadata,
                 processing_steps: list,
                 out: np.ndarray,
                 count: np.ndarray
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
                               out,
                               count,
                               start,
                               stop,
                               step,
                               min_intensity)
    return binned_q


def _bin_maps_with_indices(indices: List[int],
                           frame: Frame,
                           start: np.ndarray,
                           stop: np.ndarray,
                           step: np.ndarray,
                           min_intensity: float,
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

    rsm_shape = finite_diff_shape(start, stop, step)
    # Allocate a new numpy array on the python end. Originally, I malloced a
    # big array on the C end, but the numpy C api documentation wasn't super
    # clear on 1) how to cast this to a np.ndarray, or 2) how to prevent memory
    # leaks on the manually malloced array.
    # np.zeros is a fancy function; it is blazingly fast. So, allocate the large
    # array using np.zeros (as opposed to manual initialization to zeros on the
    # C end).
    binned_q = np.zeros(rsm_shape, np.float32)

    # We need a second array of the same size to store the number of times we
    # add to each voxel in the out array. This prevents overcounting errors.
    count = np.zeros(rsm_shape, np.uint32)
    try:
        # Do the binning, adding each binned dataset to binned_q.
        for idx in indices:
            print(f"Processing image {idx}.\r", end='')
            _bin_one_map(frame, start, stop, step, min_intensity, idx, metadata,
                         processing_steps, binned_q, count)

        # Now we've finished binning, add this to the final shared data array.
        shared_mem = SharedMemory(name='arr')
        shared_count = SharedMemory(name='count')
        shape = finite_diff_shape(start, stop, step)
        final_data = np.ndarray(
            shape, dtype=np.float32, buffer=shared_mem.buf)
        final_cnt = np.ndarray(
            shape, dtype=np.uint32, buffer=shared_count.buf)

        # Update the final data in a thread safe but parallel way. Here we're
        # ensuring that no two threads are ever in the same bit of the final
        # array.
        for i, chunk_idx in enumerate(_chunk_indices(final_data, NUM_THREADS)):
            with LOCKS[i]:
                start, stop = chunk_idx
                final_data[start:stop] += binned_q[start:stop]
                final_cnt[start:stop] += count[start:stop]

        # Always do your chores.
        shared_mem.close()
        shared_count.close()

    except Exception as exception:
        print("Exception thrown in bin_one_map:")
        print(traceback.format_exc())
        raise exception


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
        min_intensity: float = None,  # Cutoff intensity for pixels.
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
            min_intensity:
                The intensity value of a pixel below which the pixel will be
                completely ignored (the algorithm will act as though that pixel
                was never measured). This is used for masking.
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
        arr = np.ndarray(shape=shape, dtype=np.float32)

        check_shared_memory('arr')
        check_shared_memory('count')

        # Make a shared memory block for final_data; initialize it.
        shared_mem = SharedMemory('arr', create=True, size=arr.nbytes)
        # Do the same for the count.
        shared_count = SharedMemory('count', create=True, size=arr.nbytes)

        # Now hook final_data up to the shared_mem buffer that we just made.
        final_data = np.ndarray(shape, dtype=arr.dtype,
                                buffer=shared_mem.buf)
        # Set the array to be full of zeros.
        final_data.fill(0)

        # Also zero the count array.
        counts = np.ndarray(shape, dtype=np.uint32, buffer=shared_count.buf)

        # A pool-less single threaded approach. This comes with a little extra
        # performance related information.
        if num_threads == 1:
            print("Using single threaded routine.")
            counts = np.zeros_like(counts)

            for i in range(self.metadata.data_file.scan_length):
                print(f"Processing image {i}...\r", end='')
                img = self.load_image(i)
                img._processing_steps = self._processing_steps
                time_1 = time.time()
                q_vectors = img.q_vectors(frame)
                time_taken = time.time() - time_1
                print(f"Mapping time: {time_taken}")

                time_1 = time.time()
                weighted_bin_3d(q_vectors,
                                img.data,
                                final_data,
                                counts,
                                start,
                                stop,
                                step,
                                min_intensity)
                time_taken = time.time() - time_1
                print(f"Binning, img.data & final_data+= time: {time_taken}")
            return_arr = np.copy(final_data)
            _on_exit(shared_mem)
            _on_exit(shared_count)
            return return_arr, counts

        # If execution reaches here, we want a multithreaded binned RSM using
        # a thread pool. For this, we'll need a lock to share between processes.
        locks = [Lock() for _ in range(num_threads)]

        # The high performance approach.
        with Pool(
            processes=num_threads,  # The size of our pool.
            initializer=init_process_pool,  # Our pool's initializer.
            initargs=(locks,  # The initializer makes this lock global.
                      num_threads)  # Initializer also makes num_threads global.
        ) as pool:
            # Submit all of the image processing functions to the pool as jobs.
            async_results = []
            for indices in _chunks(list(range(
                    self.metadata.data_file.scan_length)), num_threads):
                async_results.append(pool.apply_async(
                    _bin_maps_with_indices,
                    (indices, frame, start, stop, step, min_intensity,
                     self.metadata, self._processing_steps)))

            # Wait for all the work to complete.
            for result in async_results:
                result.wait()
                if not result.successful():
                    raise ValueError(
                        "Could not carry out map for an unknown reason. "
                        "Probably one of the threads segfaulted, or something.")

        # Close the shared memory pool; return the final data.
        final_data = np.copy(final_data)
        counts = np.copy(counts)
        _on_exit(shared_mem)
        _on_exit(shared_count)
        return final_data, counts

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

    def load_image(self, idx: int, load_data=True):
        """
        Convenience method for loading a single image. This is unpicklable.
        """
        return Image(self.metadata, idx, load_data)

    def q_bounds(self, frame: Frame):
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
        q_vec = img.q_vectors(frame, poni)

        start, stop = q_vec, q_vec

        # Iterate over every image in the scan.
        for i in range(self.metadata.data_file.scan_length):
            # Instantiate an image without loading its data.
            img = self.load_image(i, load_data=False)

            # Work out all the extreme q values for this image.
            q_vecs = img.q_vectors(frame, extremal_q_points)

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
