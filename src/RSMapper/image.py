"""
This module contains the class that is used to store images.
"""

import numpy as np
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

    def __init__(self, metadata: RSMMetadata, index: int):
        self._raw_data = metadata.data_file.get_image(index)
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

    def q_vectors(self, frame: Frame) -> None:
        """
        Calculates the wavevector through which light had to scatter to reach
        every pixel on the detector in a given frame of reference.

        First, we calculate the wavevector of the outgoing light at each pixel.
        This is done using basic vector algebra.
        """
        # Make sure that our frame of reference has the correct index.
        frame.scan_index = self.index

        # We need num_x_pixels, num_y_pixels, 3 to be our shape.
        # Note that we need the extra "3" to store qx, qy, qz (3d vector).
        desired_shape = tuple(list(self._raw_data.shape) + [3])
        k_out_array = np.zeros(desired_shape)

        # Get a unit vector pointing towards the detector.
        det_displacement = self.diffractometer.get_detector_vector(frame)

        # Make a unit vector that points upwards along the slow axis of the
        # detector in this frame of reference.
        det_vertical = self.diffractometer.get_detector_vertical(frame)

        # Make a unit vector that points horizontally along the fast axis of the
        # detector in this frame of reference.
        det_horizontal = self.diffractometer.get_detector_horizontal(frame)

        # Now we have an orthonormal basis: det_displacement points from the
        # sample to the detector, det_vertical points vertically up the detector
        # and det_horizontal points horizontally along the detector. To get a
        # vector parallel to k_out, we can just work out the displacement from
        # each pixel to the sample.
        # This calculation is done component-by-component to match array shapes.
        detector_distance = self.metadata.data_file.detector_distance
        k_out_array[:, :, 0] = (
            det_displacement.array[0]*detector_distance +
            det_vertical.array[0]*self.metadata.vertical_pixel_distances +
            det_horizontal.array[0]*self.metadata.horizontal_pixel_distances)
        k_out_array[:, :, 1] = (
            det_displacement.array[1]*detector_distance +
            det_vertical.array[1]*self.metadata.vertical_pixel_distances +
            det_horizontal.array[1]*self.metadata.horizontal_pixel_distances)
        k_out_array[:, :, 2] = (
            det_displacement.array[2]*detector_distance +
            det_vertical.array[2]*self.metadata.vertical_pixel_distances +
            det_horizontal.array[2]*self.metadata.horizontal_pixel_distances)

        # We're going to need to normalize; this function bottlenecks.
        norms_calculated = np.linalg.norm(k_out_array, axis=-1)

        # Now we have to do some broadcasting magic. This is fastest when done
        # explicitly. This part of the code is bottlenecking, so here goes.
        norms = np.zeros_like(k_out_array)
        # Scuffed, but the fastest way to get an array of the right shape.
        norms[:, :, 0] = norms_calculated
        norms[:, :, 1] = norms_calculated
        norms[:, :, 2] = norms_calculated

        # Right now, k_out_array[a, b] has units of meters for all a, b. We want
        # k_out_array[a, b] to be normalized (elastic scattering). This can be
        # done now that norms has been created because it has the right shape.
        k_out_array /= norms

        # Now simply subtract and rescale to get the q_vectors!
        k_out_array -= self.diffractometer.get_incident_beam(frame).array
        k_out_array *= self.metadata.q_incident_length
        return k_out_array
