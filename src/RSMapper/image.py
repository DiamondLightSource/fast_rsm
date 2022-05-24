"""
This module contains the class that is used to store images.
"""

from typing import List

import numpy as np
from PIL import Image as PILImage
from diffraction_utils import Frame

from .rsm_metadata import RSMMetadata


class Image:
    """
    The class used to store raw image data. Internally, this data is stored as
    a numpy array.

    Attrs:
        data:
            A numpy array storing the image data. This is a property.
        metadata:
            An instance of RSMMetadata containing the scan's metadata.
        diffractometer:
            A reference to the RSMMetadata's diffractometer.
        index:
            The index of the image in the scan.
    """

    def __init__(self, raw_data: np.ndarray, metadata: RSMMetadata, index: int):
        self._raw_data = raw_data
        self.metadata = metadata
        self.diffractometer = self.metadata.diffractometer
        self.index = index

        self._delta_q = None

        # Carry out transposes etc. if necessary:
        # We want self.data[0, 0] to be the top left pixel.
        # We want self.data[-1, 0] to be th top right pixel.
        # self.metadata should contain enough information for us to do this.
        self._correct_img_axes()

        # Storage for all of the processing steps to be applied to the _raw_data
        # prior to mapping.
        self._processing_steps = []

    def _correct_img_axes(self):
        """
        Correct the image axes so that the image is the right way around, taking
        transposes if necessary. This method can use the metadata to work out
        where the data came from so that different transposes/inversions can be
        carried out depending on where the data was acquired. Information should
        be scraped from self.metadata to work out if any corrections are
        necessary at this point.
        """

    def add_processing_step(self, function) -> None:
        """
        Adds the processing step to the processing pipeline.

        Args:
            function:
                A function that takes a numpy array as an argument, and returns
                a numpy array.
        """
        self._processing_steps.append(function)

    @property
    def data(self):
        """
        Returns the normalized, processed data. If you want to apply some
        pre-processing to the data before mapping (e.g. by applying some
        thresholding, masking, clustering, or any arbitrary algorithm) then
        """
        arr = self._raw_data
        for step in self._processing_steps:
            arr = step(arr)

        return arr/self.metadata.solid_angles

    def pixel_polar_angle(self, frame: Frame) -> np.ndarray:
        """
        Returns the polar angle at each pixel in the specified frame.

        Args:
            frame (Frame):
                The frame of reference in which we want the pixel's polar angle.

        returns:
            The polar angle at each pixel in the requested frame.
        """
        # Make sure that the frame's index is correct.
        frame.scan_index = self.index
        # Grab the detector vector in our frame of interest.
        detector_vector = self.diffractometer.get_detector_vector(frame)
        # Now return the polar angle at each pixel.
        return self.metadata.relative_polar + detector_vector.polar_angle

    def pixel_azimuthal_angle(self, frame: Frame):
        """
        Returns the azimuthal angle at each pixel in the specified frame.

        Args:
            frame (Frame):
                The frame of reference in which we want the pixel's azimuthal#
                angle.

        Returns:
            The azimuthal angle at each pixel in the requested frame.
        """
        # Make sure that the frame's index is correct.
        frame.scan_index = self.index
        # Grab the detector vector in our frame of interest.
        detector_vector = self.diffractometer.get_detector_vector(frame)
        # Now return the azimuthal angle at each pixel.
        return self.metadata.relative_azimuth + detector_vector.azimuthal_angle

    @property
    def q_out(self) -> np.ndarray:
        """
        Returns the q vectors of the light after scattering to each pixel on the
        detector.
        """
        q_z = np.zeros_like(self.delta_q)
        q_z[:, :, 2] = self.metadata.q_incident_lenth
        return self.delta_q + q_z

    def delta_q(self, frame: Frame) -> None:
        """
        Calculates the wavevector through which light had to scatter to reach
        every pixel on the detector in a given frame of reference.

        This is the most performance critical part of the code. As a result,
        some of the maths is written out in weird ways. Trig identities are used
        to save time wherever possible.
        """
        # Make sure that our frame of reference has the correct index.
        frame.scan_index = self.index

        # We need num_x_pixels, num_y_pixels, 3 to be our shape.
        # Note that we need the extra "3" to store qx, qy, qz (3d vector).
        desired_shape = tuple(list(self._raw_data.shape) + [3])
        delta_q = np.zeros(desired_shape)

        # Optimized trig calculations.
        cos_azimuth = np.cos(self.pixel_azimuthal_angle(frame))
        sin_azimuth = np.sqrt(1 - cos_azimuth**2)
        cos_polar = np.cos(self.pixel_polar_angle(frame))
        sin_polar = np.sqrt(1 - cos_polar**2)
        # Now set the elements of the delta q matrix element.
        # First set all the delta_q_x values, then delta_q_y, then delta_q_z.
        # Note that these are just q_out for now.
        delta_q[:, :, 0] = sin_polar * sin_azimuth
        delta_q[:, :, 1] = cos_polar
        delta_q[:, :, 2] = sin_polar * cos_azimuth

        # delta_q = q_out - q_in; finally, give it the correct length.
        delta_q -= self.diffractometer.get_incident_beam(frame).array
        delta_q *= self.metadata.q_incident_lenth

        print("Top left: ", delta_q[0, 0, :])
        print("Bottom right: ", delta_q[-1, -1, :])

        return delta_q

    @classmethod
    def from_image_paths(cls, paths: List[str], metadata, img_idx):
        """
        Loads an image from a list of paths and an image index.
        """
        return cls(np.array(PILImage.open(paths[img_idx])), metadata, img_idx)
