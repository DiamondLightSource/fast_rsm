"""
This module contains the class that is used to store images.
"""

import copy
import numpy as np
from PIL import Image as PILImage

from .metadata import Metadata
from .motors import Motors


class Image:
    """
    The class used to store raw image data. Internally, this data is stored as
    a numpy array.

    Attrs:
        data:
            A numpy array storing the image data. This is a property.
    """

    def __init__(self,
                 raw_data: np.ndarray,
                 motors: Motors,
                 metadata: Metadata):
        self._raw_data = raw_data
        self.motors = motors
        self.metadata = metadata

        self._delta_q = None

        # Carry out transposes etc. if necessary:
        # We want self.data[0, 0] to be the top left pixel.
        # We want self.data[-1, 0] to be th top right pixel.
        # self.metadata should contain enough information for us to do this.
        self._correct_img_axes()

    def _correct_img_axes(self):
        """
        Correct the image axes so that the image is the right way around, taking
        transposes if necessary. This method can use the metadata to work out
        where the data came from so that different transposes/inversions can be
        carried out depending on where the data was acquired. Information should
        be scraped from self.metadata to work out if any corrections are
        necessary at this point.
        """

    @property
    def data(self):
        """
        Returns the normalized data.
        """
        return self._raw_data/self.metadata.solid_angles

    @property
    def pixel_polar_angle_sample_frame(self):
        """
        Returns the polar angle at each pixel, where the coordinate system is
        tied to the sample.
        """
        return self.metadata.relative_polar + self.motors.detector_polar

    @property
    def pixel_azimuthal_angle_sample_frame(self):
        """
        Returns the azimuthal angle at each pixel, where the coordinate system
        is the typical synchrotron coordinate system, but tied to the sample
        (so with omega ≠ 0 or chi ≠ 0, the frame rotates with the sample).
        """
        return self.metadata.relative_azimuth + self.motors.detector_azimuth

    @property
    def q_out(self) -> np.ndarray:
        """
        Returns the q vectors of the light after scattering to each pixel on the
        detector.
        """
        q_z = np.zeros_like(self.delta_q)
        q_z[:, :, 2] = self.metadata.q_incident_lenth
        return self.delta_q + q_z

    @property
    def delta_q(self) -> np.ndarray:
        """
        Returns the q vectors through which light had to scatter to reach each
        pixel.
        """
        if self._delta_q is None:
            self._init_delta_q()
        return self._delta_q

    def _init_delta_q(self) -> None:
        """
        Sets the self._delta_q attribute for this instance of Image.
        """
        # We need num_x_pixels, num_y_pixels, 3 to be our shape.
        # Note that we need the extra "3" to store qx, qy, qz (3d vector).
        desired_shape = tuple(list(self._raw_data.shape) + [3])
        delta_q = np.zeros(desired_shape)

        # Optimized trig calculations. One day, these should be done in C via
        # lookup tables for maximum speed.
        cos_azimuth = np.cos(self.pixel_azimuthal_angle_sample_frame)
        sin_azimuth = np.sqrt(1 - cos_azimuth**2)
        cos_polar = np.cos(self.pixel_polar_angle_sample_frame)
        sin_polar = np.sqrt(1 - cos_polar**2)
        # Now set the elements of the delta q matrix element.
        # First set all the delta_q_x values, then delta_q_y, then delta_q_z.
        # Note that these are just q_out for now.
        delta_q[:, :, 0] = sin_polar * sin_azimuth
        delta_q[:, :, 1] = cos_polar
        delta_q[:, :, 2] = sin_polar * cos_azimuth

        # delta_q = q_out - q_in; finally, give it the correct length.
        delta_q -= self.motors.incident_beam
        delta_q *= self.metadata.q_incident_lenth

        self._delta_q = delta_q

    @classmethod
    def from_file(cls, path: str, motors: Motors, metadata: Metadata):
        """
        Loads the image from a file.
        """
        with PILImage.open(path) as img:
            img_arr = copy.deepcopy(np.array(img))

        return cls(img_arr, motors, metadata)
