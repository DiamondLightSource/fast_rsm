"""
This module contains the class that is used to store images.
"""


import numpy as np
from diffraction_utils import Frame
from scipy.spatial.transform import Rotation
import mapper_c_utils
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
        # Store intensities as 32 bit floats.
        self._raw_data = metadata.data_file.get_image(index).astype(np.float32)
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
        use the add_processing_step method.
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

    def q_vectors(self, frame: Frame) -> np.ndarray:
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
        # Don't bother initializing this.
        k_out_array = np.ndarray(desired_shape, np.float32)

        # Get a unit vector pointing towards the detector.
        det_displacement = self.diffractometer.get_detector_vector(frame)
        det_displacement.array = det_displacement.array.astype(np.float32)

        # Make a unit vector that points upwards along the slow axis of the
        # detector in this frame of reference.
        det_vertical = self.diffractometer.get_detector_vertical(frame)
        det_vertical.array = det_vertical.array.astype(np.float32)

        # Make a unit vector that points horizontally along the fast axis of the
        # detector in this frame of reference.
        det_horizontal = self.diffractometer.get_detector_horizontal(frame)
        det_horizontal.array = det_horizontal.array.astype(np.float32)

        # Now we have an orthonormal basis: det_displacement points from the
        # sample to the detector, det_vertical points vertically up the detector
        # and det_horizontal points horizontally along the detector. To get a
        # vector parallel to k_out, we can just work out the displacement from
        # each pixel to the sample.
        # This calculation is done component-by-component to match array shapes.
        # This routine has been benchmarked to be ~4x faster than using an
        # outer product and reshaping it.
        detector_distance = self.metadata.data_file.detector_distance
        detector_distance = np.array(detector_distance, np.float32)

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

        # We're going to need to normalize; this function bottlenecks if not
        # done exactly like this!
        # This weird manual norm calculation is an order of magnitude faster
        # than using np.linalg.norm(k_out_array, axis=-1), and the manual
        # addition is an order of magnitude faster than using sum(.., axis=-1)
        k_out_squares = np.square(k_out_array)
        norms = (
            k_out_squares[:, :, 0] +
            k_out_squares[:, :, 1] +
            k_out_squares[:, :, 2])
        norms = np.sqrt(norms)
        # Right now, k_out_array[a, b] has units of meters for all a, b. We want
        # k_out_array[a, b] to be normalized (elastic scattering). This can be
        # done now that norms has been created because it has the right shape.
        k_out_array[:, :, 0] /= norms
        k_out_array[:, :, 1] /= norms
        k_out_array[:, :, 2] /= norms

        # Note that for performance reasons these should also be float32.
        incident_beam_arr = self.diffractometer.get_incident_beam(frame).array
        incident_beam_arr = incident_beam_arr.astype(np.float32)
        q_incident = np.array(self.metadata.q_incident_length, np.float32)

        # Now simply subtract and rescale to get the q_vectors!
        # Note that this is an order of magnitude faster than:
        # k_out_array -= incident_beam_arr
        k_out_array[:, :, 0] -= incident_beam_arr[0]
        k_out_array[:, :, 1] -= incident_beam_arr[1]
        k_out_array[:, :, 2] -= incident_beam_arr[2]
        k_out_array *= q_incident

        # Finally, if a user has specified that they want their results output
        # in hkl-space, multiply each of these vectors by the inverse of UB.
        # Note that this is not an intelligent solution! A more optimal
        # calculation would be carried out in hkl coordinates to begin with.
        # It's more the case that I'm lazy, this calculation is cleaner and the
        # performance difference is pretty small. And, I mean, doing the whole
        # calculation in a non-orthogonal basis sounds gross.
        if frame.frame_name == Frame.hkl:
            # pylint: disable=c-extension-no-member
            ub_mat = self.metadata.data_file.ub_matrix.astype(np.float32)
            ub_mat = np.linalg.inv(ub_mat)

            # OKAY, so for ...reasons... this ub matrix's z-axis is my y-axis.
            # Also, its y is my -z.
            # If you don't like linear algebra, shut your eyes real quick.
            basis_change = Rotation.from_rotvec([np.pi/2, 0, 0])

            ub_mat = np.matmul(ub_mat, basis_change.as_matrix())
            ub_mat = np.matmul(basis_change.inv().as_matrix(), ub_mat)

            # It turns out that diffcalc assumes that k has an extra factor of
            # 2π. I would never in my life have realised this had it not been
            # for an offhand comment by my colleague Dean. Thanks, Dean.
            ub_mat *= 2*np.pi

            # Final fixes to make the orientation of reciprocal space match
            # diffcalc's orientation. This is really optional and just a
            # definition. This is equivalent to (but faster than) rotating the
            # coordinate system by π about the k(y)-axis later.
            ub_mat[0] = -ub_mat[0]
            ub_mat[2] = -ub_mat[2]

            # The custom, high performance linear_map expects float32's.
            ub_mat = ub_mat.astype(np.float32)

            k_out_array = k_out_array.reshape(
                (desired_shape[0]*desired_shape[1], 3))

            mapper_c_utils.linear_map(k_out_array, ub_mat)

            # Reshape the k_out_array to have the same shape as the raw image.
            k_out_array = k_out_array.reshape(desired_shape)

        return k_out_array
