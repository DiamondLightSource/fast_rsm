"""
This module contains the scan class, that is used to store all of the
information relating to a reciprocal space scan.

TODO: Better exception handling.
"""

from multiprocessing.pool import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Lock
from pathlib import Path
from typing import Union, Tuple, List

import numpy as np

from diffraction_utils import I10Nexus, Vector3, Frame
from diffraction_utils.diffractometers import I10RasorDiffractometer

from .binning import linear_bin, finite_diff_shape
from .image import Image
from .rsm_metadata import RSMMetadata


# Make a global lock for the shared memory block used in parallel code.
LOCK = Lock()


def _load_image(image_paths: List[str],
                metadata: RSMMetadata,
                img_idx: int) -> Image:
    """A global (and therefore picklable) image loader."""
    return Image.from_image_paths(image_paths, metadata, img_idx)


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
                 image_paths: List[str],
                 idx: int,
                 metadata: RSMMetadata
                 ) -> None:
    """
        Calculates and bins the reciprocal space map with index idx. Saves the
        result to the shared memory buffer.
        """
    shared_mem = SharedMemory(name='arr')
    shape = finite_diff_shape(start, stop, step)
    final_data = np.ndarray(shape, dtype=np.float64, buffer=shared_mem.buf)
    image = _load_image(image_paths, metadata, idx)
    # Do the mapping for this image; bin the mapping.
    delta_q = image.delta_q(frame)
    binned_q = linear_bin(delta_q,
                          image.data,
                          start,
                          stop,
                          step)

    # Update the final data array in a thread safe way.
    with LOCK:
        final_data += binned_q

    # logging.info("idx %i", idx)
    # Do some tidying up.
    shared_mem.close()


def _bin_maps_with_indices(indices: List[int],
                           frame: Frame,
                           start: np.ndarray,
                           stop: np.ndarray,
                           step: np.ndarray,
                           image_paths: List[str],
                           metadata: RSMMetadata
                           ) -> None:
    """
    Bins all of the maps with indices in indices. The purpose of this
    intermediate function call is to decrease the amount of context switching/
    serialization that the interpreter has to do.
    """
    try:
        for idx in indices:
            _bin_one_map(frame, start, stop, step, image_paths, idx, metadata)
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

    def __init__(self,
                 metadata: RSMMetadata,
                 image_paths: List[str]):
        self.metadata = metadata
        self.image_paths = image_paths

        self._rsm = None
        self._rsm_frame = None

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

        # Prepare an array with the same shape as our final binned data array.
        shape = finite_diff_shape(start, stop, step)
        # Note that this doesn't initialise the array; arr is nonsense.
        arr = np.ndarray(shape=shape)

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
                final_data += linear_bin(
                    img.delta_q(frame),
                    img.data,
                    start, stop, step)
            return final_data

        # The high performance approach.
        with Pool(processes=num_threads) as pool:
            # Submit all of the image processing functions to the pool as jobs.
            async_results = []
            for indices in _chunks(list(range(
                    self.metadata.data_file.scan_length)), num_threads):
                async_results.append(pool.apply_async(
                    _bin_maps_with_indices,
                    (indices, frame, start, stop, step, self.image_paths,
                     self.metadata,)))

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

        delta_qs = []
        # Load images one by one.
        for idx in range(self.metadata.data_file.scan_length):
            image = _load_image(self.image_paths, self.metadata, idx)
            # Do the mapping for this image in correct frame.
            delta_qs.append(image.delta_q(frame))

        return delta_qs

    def load_image(self, idx: int):
        """
        Convenience method for loading a single image. This is unpicklable.
        """
        return _load_image(self.image_paths, self.metadata, idx)

    @classmethod
    def from_i10(cls,
                 path_to_nx: Union[str, Path],
                 beam_centre: Tuple[int],
                 detector_distance: float,
                 sample_oop: Vector3,
                 path_to_tiffs: str = ''):
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
            path_to_tiffs:
                Path to the directory in which the images are stored. Defaults
                to '', in which case a bunch of reasonable directories will be
                searched for the images.
        """
        # Load the nexus file.
        i10_nexus = I10Nexus(path_to_nx, detector_distance)

        # Load the state of the RASOR diffractometer; prepare the metadata.
        diff = I10RasorDiffractometer(i10_nexus, sample_oop, 'area')
        meta = RSMMetadata(diff, beam_centre)

        # Make sure the sample_oop vector's frame's diffractometer is correct.
        sample_oop.frame.diffractometer = diff

        image_paths = meta.data_file.get_local_image_paths(path_to_tiffs)
        return cls(meta, image_paths)
