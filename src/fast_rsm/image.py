"""
This module contains the class that is used to store images.
"""


from typing import Union

import numpy as np
from diffraction_utils import Frame, I07Nexus, Polarisation

import mapper_c_utils
from fast_rsm.rsm_metadata import RSMMetadata

import fast_rsm.corrections as corrections
import logging

logger = logging.getLogger(__name__)

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
        load_image:
            Should we actually load the image? It can be useful to access an
            Image object without incurring the latency of an image load to e.g.
            quickly calculate a few q_vectors.
    """

    def __init__(self, metadata: RSMMetadata, index: int, load_image=True):
        # Store intensities as 32 bit floats.
        if load_image:
            self._raw_data = metadata.data_file.get_image(
                index).astype(np.float32)
        else:
            # Allocate, but don't initialize.
            self._raw_data = np.ndarray(metadata.data_file.image_shape)

        self.metadata = metadata
        self.diffractometer = self.metadata.diffractometer
        self.index = index

        self._delta_q = None#

        # Carry out transposes etc. if necessary:
        # We want self.data[0, 0] to be the top left pixel.
        # We want self.data[-1, 0] to be th top right pixel.
        # self.metadata should contain enough information for us to do this.
        #need to make this conditional on loading an image, as the predefined imageshapes
        #already have account for rotation. 
        if load_image:
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
        if isinstance(self.metadata.data_file, I07Nexus):
            if (self.metadata.data_file.is_rotated):
                # The detector has been rotated in the experiment!
                # NOTE: this is slow. If you flip your detector and run fscans,
                # f#!@ you.
                self._raw_data = self._raw_data.transpose()
                self._raw_data = np.flip(self._raw_data, axis=0)
                

    def generate_mask(self, min_intensity: Union[float, int]) -> np.ndarray:
        """
        Generates a mask from every pixel whose intensity is below a certain
        value. This mask uses the intensities recorded in the _raw_data, not
        the corrected data returned by the Image.data property. While this
        might seem a bit dodgy, it does mean that the mask is consistent with
        what a user would see as a raw detector output.

        Note that the condition is written as
            self._raw_data >= min_intensity
        *not*
            self._raw_data > min_intensity

        Args:
            min_intensity:
                If the raw intensity value of a pixel is below this cutoff, that
                pixel will be masked.

        Returns:
            A numpy array of ones and zeroes, where a zeroes represent masked
            pixels. The dtype of the array is dtype('bool'), but under the hood
            it's just ones and zeros.
        """
        return self._raw_data >= min_intensity

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

        # Solid angle corrections are not optional.
        arr /= self.metadata.solid_angles

        # If our data file has transmission data, divide by it.
        # This is not optional!
        try:
            if isinstance(self.metadata.data_file.transmission, np.ndarray):
                transmissionlist=self.metadata.data_file.transmission.flatten()
                arr /= transmissionlist[self.index]
            else:
                arr /= self.metadata.data_file.transmission
        except AttributeError:
            pass
        
        #normalise image data to count time
        scan_entry=self.metadata.diffractometer.data_file.nxfile
        try:
            if isinstance(scan_entry.entry.attenuation.count_time.nxdata,np.ndarray):
                arr /= scan_entry.entry.attenuation.count_time.nxdata[self.index]
            else:
                arr /= scan_entry.entry.attenuation.count_time.nxdata
        except AttributeError:
            pass
        #if there is an edf mask file loaded, apply mask
        if self.metadata.edfmask is not None:
            arr[self.metadata.edfmask.astype(bool)]=np.nan
        
        # If there are pixels to mask, mask them.
        if self.metadata.mask_pixels is not None:
            arr[self.metadata.mask_pixels] = np.nan

        # If there are regions to mask, mask them too.
        if self.metadata.mask_regions is not None:
            for region in self.metadata.mask_regions:
                arr[region.slice] = np.nan

        return arr

    def q_vectors(self,
                  frame: Frame,
                  spherical_bragg_vec: np.array,
                  oop='y',
                  indices: tuple = None,
                  lorentz_correction: bool = False,
                  pol_correction: bool = True,
                   ) -> np.ndarray:
        """
        Calculates the wavevector through which light had to scatter to reach
        every pixel on the detector in a given frame of reference.

        First, we calculate the wavevector of the outgoing light at each pixel.
        This is done using basic vector algebra.

        Args:
            frame:
                The frame of reference in which we want to calculate the
                q_vectors.
            spherical_bragg_vec:
                XYZ vector for shifting centre of volume when interested in plotting
                spherical polar plots. defaults to [0,0,0] when unspecified. 
            indices:
                The indices that we want to carry out the map for. Defaults to
                None, in which case the entire image is mapped. E.g. passing
                indices=(-1, -1) will only calculate the scattering vector for
                the bottom right pixel.

        Returns:
            If your image has a shape of (a, b), the output of q_vectors has
            shape (a, b, 3), i.e. you get one q_vector for each pixel. If
            you specify an index, then your output will have shape (1, 1, 3).
            Probably.
        """


        if indices is None:
            i = slice(None)
            j = slice(None)
        else:
            i = indices[0]
            j = indices[1]
            # #may need additional swap of axis if image is rotated
            # if self.metadata.data_file.is_rotated:
            #     i=indices[1]
            #     j=indices[0]
 
        # Make sure that our frame of reference has the correct index and
        # diffractometer.
        frame.scan_index = self.index
        frame.diffractometer = self.metadata.diffractometer

        # We need num_x_pixels, num_y_pixels, 3 to be our shape.
        # Note that we need the extra "3" to store qx, qy, qz (3d vector).
        desired_shape = tuple(list(self._raw_data.shape) + [3])
        # if self.metadata.data_file.is_rotated:
        #     desired_shape= tuple(list([self._raw_data.shape[1],self._raw_data.shape[0]]) + [3])

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
        detector_distance = self.metadata.get_detector_distance(self.index)
        detector_distance = np.array(detector_distance, np.float32)
        vertical = self.metadata.get_vertical_pixel_distances(self.index)
        horizontal = self.metadata.get_horizontal_pixel_distances(self.index)

        k_out_array[i, j, 0] = (
            det_displacement.array[0]*detector_distance +
            det_vertical.array[0]*vertical[i, j] +
            det_horizontal.array[0]*horizontal[i, j])
        k_out_array[i, j, 1] = (
            det_displacement.array[1]*detector_distance +
            det_vertical.array[1]*vertical[i, j] +
            det_horizontal.array[1]*horizontal[i, j])
        k_out_array[i, j, 2] = (
            det_displacement.array[2]*detector_distance +
            det_vertical.array[2]*vertical[i, j] +
            det_horizontal.array[2]*horizontal[i, j])

        # We're going to need to normalize; this function bottlenecks if not
        # done exactly like this!
        # This weird manual norm calculation is an order of magnitude faster
        # than using np.linalg.norm(k_out_array, axis=-1), and the manual
        # addition is an order of magnitude faster than using sum(.., axis=-1)
        k_out_squares = np.square(k_out_array[i, j, :])

        # k_out_sqares' shape depends on what i and j are. Handle all 3 cases.
        if len(k_out_squares.shape) == 1:
            norms = np.sum(k_out_squares)
        elif len(k_out_squares.shape) == 2:
            norms = (k_out_squares[:, 0] +
                     k_out_squares[:, 1] +
                     k_out_squares[:, 2])
        elif len(k_out_squares.shape) == 3:
            norms = (k_out_squares[:, :, 0] +
                     k_out_squares[:, :, 1] +
                     k_out_squares[:, :, 2])
        norms = np.sqrt(norms)

        # Right now, k_out_array[a, b] has units of meters for all a, b. We want
        # k_out_array[a, b] to be normalized (elastic scattering). This can be
        # done now that norms has been created because it has the right shape.
        k_out_array[i, j, 0] /= norms
        k_out_array[i, j, 1] /= norms
        k_out_array[i, j, 2] /= norms

        # For performance reasons these should also be float32.
        incident_beam_arr = self.diffractometer.get_incident_beam(frame).array
        incident_beam_arr = incident_beam_arr.astype(np.float32)
        k_incident_len = np.array(self.metadata.k_incident_length, np.float32)

        # At exactly this point, while k_in and k_out are normalised "for
        # free", Lorentz/polarisation corrections should be applied. Only do
        # this if we're mapping the entire image (i.e. indices is None).
     # this if we're mapping the entire image (i.e. indices is None).
        if indices is None:
            if lorentz_correction:
                corrections.lorentz(
                    self._raw_data, incident_beam_arr, k_out_array)
            # The kind of polarisation correction that we want to apply of
            # depends, rather obviously, on the polarisation of the beam!
            polarisation = self.metadata.data_file.polarisation
            if polarisation.kind != Polarisation.linear:
                raise NotImplementedError(
                    "Only linear polarisation corrections have been "
                    "implemented.")
            if polarisation.kind == Polarisation.linear:
                pol_vec = polarisation.vector
                pol_vec.to_frame(frame)
                if pol_correction:
                    corrections.linear_polarisation(
                        self._raw_data, k_out_array, pol_vec.array)
        # Note that this is an order of magnitude faster than:
        # k_out_array -= incident_beam_arr
        k_out_array[i, j, 0] -= incident_beam_arr[0]
        k_out_array[i, j, 1] -= incident_beam_arr[1]
        k_out_array[i, j, 2] -= incident_beam_arr[2]

        # Richard:  It turns out that diffcalc assumes that k has an extra factor of
        # 2π. I would never in my life have realised this had it not been
        # for an offhand comment by my colleague Dean. Thanks, Dean.
        k_out_array[i, j, :] *= k_incident_len * 2*np.pi

        # If a user has specified that they want their results output
        # in hkl-space, multiply each of these vectors by the inverse of UB.
        # Note that this is not an intelligent solution! A more optimal
        # calculation would be carried out in hkl coordinates to begin with.
        # It's more the case that I'm lazy, this calculation is cleaner and the
        # performance difference is pretty small. And, I mean, doing the whole
        # calculation in a non-orthogonal basis sounds gross.
        if frame.frame_name == Frame.hkl:
            ub_mat = self.metadata.data_file.ub_matrix.astype(np.float32)
            ub_mat = np.linalg.inv(ub_mat)
        else:
            ub_mat = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        # Finally, we make it so that (001) will end up OOP.
        if oop == 'y':
            coord_change_mat = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
        elif oop == 'x':
            coord_change_mat = np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ])
        elif oop == 'z':
            coord_change_mat = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])

        ub_mat = np.matmul(ub_mat, coord_change_mat)

        # #ADD IN HERE INVERSE OF OMEGA AND ALPHA ROTATIONS, WHICH ARE NOT INCLUDED IN THE UB MATRIX. Currently only have kout-kin which is Hlab. For hkl in vertical mode we need 
        # # B^(-1)  U^(-1) (Ω^(-1)  A^(-1)  H_lab)
        # #or for horizontal 
        # #B^(-1)  U^(-1) (Θ^(-1)  χ^(-1)  H_lab)
        # #incorrectly labelled U matrix, is actually the necessary omega+alpha or theta-chi- rotations
        # samplerotations=self.diffractometer.get_u_matrix(frame.scan_index)
        # invSampRot=np.linalg.inv(samplerotations)
        # ub_mat=np.matmul(ub_mat,invSampRot)

        # The custom, high performance linear_map expects float32's.
        ub_mat = ub_mat.astype(np.float32)
        # pylint: disable=c-extension-no-member
        if indices is not None:
            to_map = np.ascontiguousarray(k_out_array[i, j, :])
            mapper_c_utils.linear_map(to_map, ub_mat)
            k_out_array[i, j, :] = to_map
        else:
            k_out_array = k_out_array.reshape(
                (desired_shape[0]*desired_shape[1], 3))
            # This takes CPU time: mapping every vector.
            mapper_c_utils.linear_map(k_out_array, ub_mat)
            # Reshape the k_out_array to have the same shape as the image.
            k_out_array = k_out_array.reshape(desired_shape)

        # If the user asked for polars then change to those coords.

        if frame.coordinates == Frame.sphericalpolar:

            k_out_array[i, j, 0] -= spherical_bragg_vec[0]
            k_out_array[i, j, 1] -= spherical_bragg_vec[1]
            k_out_array[i, j, 2] -= spherical_bragg_vec[2]
            # pylint: disable=c-extension-no-member
            if indices is not None:
                to_polar = np.ascontiguousarray(k_out_array[i, j, :])
                #mapper_c_utils.cylindrical_polar(to_polar)
                mapper_c_utils.spherical_polar(to_polar)
                k_out_array[i, j, :] = to_polar
            else:
                k_out_array = k_out_array.reshape(
                    (desired_shape[0]*desired_shape[1], 3))
                mapper_c_utils.spherical_polar(k_out_array)
                k_out_array = k_out_array.reshape(desired_shape)

        # Only return the indices that we worked on.
        return k_out_array[i, j, :]

    def q_vector_array(self,
                       frame: Frame,
                       spherical_bragg_vec: np.array,
                       oop='y',
                       lorentz_correction: bool = False,
                       pol_correction: bool = True) -> np.ndarray:
        """
        Returns a numpy array of q_vectors whose shape is (N,3).
        """
        q_vectors = self.q_vectors(
            frame,
            spherical_bragg_vec,
            oop=oop,
            lorentz_correction=lorentz_correction,
            pol_correction=pol_correction).reshape()
        num_q_vectors = q_vectors.shape[0]*q_vectors.shape[1]
        return q_vectors.reshape((num_q_vectors, 3))
