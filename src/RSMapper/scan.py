"""
This module contains the scan class, that will be used to store all of the
information relating to a reciprocal space scan.
"""

import logging
from pathlib import Path
from typing import Union, Tuple, Callable

import numpy as np

from diffraction_utils import I10Nexus, Vector3, Frame
from diffraction_utils.diffractometers import I10RasorDiffractometer

from .binning import linear_bin, finite_diff_shape
from .image import Image
from .rsm_metadata import RSMMetadata


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

    def __init__(self, metadata: RSMMetadata,
                 image_loader: Callable[[int], Image]):
        self.metadata = metadata
        self.load_image = image_loader

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
        start, stop, step = np.array(start), np.array(stop), np.array(step)
        if num_threads != 1:
            raise NotImplementedError(
                "Reciprocal space maps are currently single threaded only.")

        # Prepare the final binned data array.
        final_data = np.zeros(finite_diff_shape(start, stop, step))

        # Load images one by one.
        for idx in range(self.metadata.data_file.scan_length):
            logging.debug("Mapping image number %i.", idx)
            image = self.load_image(idx)
            # Do the mapping for this image in correct frame; bin the mapping.
            delta_q = image.delta_q(frame)
            binned_dq = linear_bin(delta_q,
                                   image.data,
                                   start,
                                   stop,
                                   step)

            # Add this freshly binned data to our overall reciprocal space map.
            final_data += binned_dq

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
            image = self.load_image(idx)
            # Do the mapping for this image in correct frame.
            delta_qs.append(image.delta_q(frame))

        return delta_qs

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

        def image_loader(img_idx: int):
            """The image loader for I10 images."""
            return Image(i10_nexus.load_image_array(img_idx, path_to_tiffs),
                         metadata=meta, index=img_idx)

        return cls(meta, image_loader)
