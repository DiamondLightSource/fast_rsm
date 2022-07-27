"""
This module contains the metadata class, which provides a python interface
for the .nxs file written for the scan.
"""

from typing import Tuple

import numpy as np
from scipy.constants import physical_constants

from diffraction_utils import DiffractometerBase, I07Nexus


class RSMMetadata:
    """
    This class contains a diffraction_utils DiffractometerBase implementation,
    which contains all of the information that it's possible to extract about
    the experiment automatically. Generally, the poni (here called beam_centre)
    has to be given by a user at run-time since this is usually experiment
    specific, so we take it as an explicit argument here.

    This class contains some reciprocal space map specific stuff that doesn't
    belong in diffraction_utils, such as relative polar coordinate angle
    calculations, solid angle calculations and the like.

    Attrs:
        diffractometer:
            An implementation of DiffractometerBase that can be used to change
            reference frames, map vectors etc. The diffractometer also has a
            data_file attribute that can be used to access additional metadata
            from the .nxs file, such as detector_distance, probe_energy etc.
        data_file:
            A reference to the diffractometer's data file.
        beam_centre:
            The centre of the beam.
    """

    def __init__(self,
                 diffractometer: DiffractometerBase,
                 beam_centre: Tuple[int, int]):  # In number of pixels.
        self.diffractometer = diffractometer
        self.data_file = diffractometer.data_file  # A handy reference.
        self.beam_centre = beam_centre

        # Correct the beam centre in case of diffractometer specific weirdness.
        self._correct_beam_centre()

        self._relative_polar = None
        self._relative_azimuth = None
        self._solid_angles = None
        self._vertical_pixel_offsets = None
        self._horizontal_pixel_offsets = None
        self._vertical_pixel_distances = None
        self._horizontal_pixel_distances = None

    def _correct_beam_centre(self):
        """
        Correct the beam centre, if necessary. We can use the metadata to work
        out where the data was acquired. Some data sources might report the
        beam_centre using a different convention; this is where we correct for
        that to make sure that that data[0, 0] is the top left pixel,
        data[-1, 0] is the bottom left pixel.

        This means that, if the beam was centered at the top left pixel, the
        user should enter that the beam_centre=[0, 0]. This seems unlikely; this
        would probably be given as [0, -1] in typical software!
        """
        if isinstance(self.data_file, I07Nexus):
            # I07 beam centres are given (x, y) (the wrong way around).
            self.beam_centre = (self.beam_centre[1], self.beam_centre[0])

            if self.data_file.is_rotated:
                self.beam_centre = (
                    self.data_file.image_shape[0] - self.beam_centre[0],
                    self.beam_centre[1])

        # Finally, make sure that the beam_centre can lie within the image.
        test_arr = np.ndarray(self.data_file.image_shape)
        try:
            test_arr[self.beam_centre[0], self.beam_centre[1]]
        except IndexError as error:
            print(f"beam_centre {self.beam_centre} out of bounds. Your image "
                  f"has shape {self.data_file.image_shape} (slow_axis, "
                  f"fast_axis).")
            raise error

    @property
    def solid_angles(self):
        """
        Returns a 2D array whose elements are proportional to the solid angle
        covered by each pixel in the camera. Divide by this to correct for
        variation in solid angle.
        """
        if self._solid_angles is None:
            self._init_solid_angles()
        return self._solid_angles

    @property
    def vertical_pixel_offsets(self):
        """
        The offsets between each pixel and the central pixel.
        """
        if self._vertical_pixel_offsets is None:
            self._init_vertical_pixel_offsets()
        return self._vertical_pixel_offsets

    @property
    def horizontal_pixel_offsets(self):
        """
        The offsets between each pixel and the central pixel.
        """
        if self._horizontal_pixel_offsets is None:
            self._init_horizontal_pixel_offsets()
        return self._horizontal_pixel_offsets

    @property
    def vertical_pixel_distances(self):
        """
        The vertical distances (in m) between each pixel and the central
        vertical pixel.
        """
        if self._vertical_pixel_distances is None:
            self._vertical_pixel_distances = \
                self.vertical_pixel_offsets * self.data_file.pixel_size
        return self._vertical_pixel_distances

    @property
    def horizontal_pixel_distances(self):
        """
        The horizontal pixel distances (in m) between each pixel and the central
        horizontal pixel.
        """
        if self._horizontal_pixel_distances is None:
            self._horizontal_pixel_distances = \
                self.horizontal_pixel_offsets * self.data_file.pixel_size
        return self._horizontal_pixel_distances

    @property
    def relative_polar(self):
        """
        This property makes accessing the detector's relative theta from the
        beam centre for each pixel, when all detector rotation motors have been
        zeroed.
        """
        if self._relative_polar is None:
            self._init_relative_polar()
        return self._relative_polar

    @property
    def relative_azimuth(self):
        """
        This property makes accessing the detector's relative phi from the
        beam centre for each pixel, when all detector rotation motors have been
        zeroed.
        """
        if self._relative_azimuth is None:
            self._init_relative_azimuth()
        return self._relative_azimuth

    @property
    def incident_wavelength(self):
        """
        Returns the wavelength of the incident light in Å.
        """
        return (physical_constants["Planck constant in eV s"][0] *
                physical_constants["speed of light in vacuum"][0] /
                self.diffractometer.data_file.probe_energy *
                1e10)  # To convert to Å.

    @property
    def k_incident_length(self):
        """
        Returns the wavevector of the incident beam in Å^-1.
        """
        return 1/self.incident_wavelength

    def _init_solid_angles(self):
        """
        Initialize the solid angles property. Note that this is not an exact
        calculation, but all the values should be almost correct (since small
        angle approximations are likely to hold nicely).

        The following are low priority because the solid angle correction is
        pretty damn minor/unimportant.
        TODO: This implementation is wrong by half a pixel. Fix this.
        TODO: This implementation is approximate and should be replaced by an
            exact treatment.
        """
        # We're going to need to inc the data shape to hack this.
        data_shape = self.data_file.image_shape
        self._init_relative_polar((data_shape[0]+1, data_shape[1]))
        theta_diffs = np.copy(self.relative_polar)
        theta_diffs = -np.diff(theta_diffs, axis=0)  # Remember the minus sign!

        self._init_relative_azimuth((data_shape[0], data_shape[1]+1))
        phi_diffs = np.copy(self._relative_azimuth)
        phi_diffs = -np.diff(phi_diffs, axis=1)

        # Now return the relative polar/azimuth arrays to normal.
        self._init_relative_polar()
        self._init_relative_azimuth()

        # And finally, do what we came here to do: a scuffed calculation.
        self._solid_angles = phi_diffs*theta_diffs

        # To prevent numbers from getting too silly, normalise this.
        self._solid_angles /= np.max(self._solid_angles)

        # Finally, store as a single precision float.
        self._solid_angles = self._solid_angles.astype(np.float32)

    def _init_vertical_pixel_offsets(self, image_shape: int = None):
        """
        Initializes the array of relative pixel offsets.
        """
        if image_shape is None:
            image_shape = self.data_file.image_shape

        num_y_pixels = image_shape[0]
        # Imagine num_y_pixels = 11.
        # pixel_offsets = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        pixel_offsets = np.arange(num_y_pixels-1, -1, -1)
        # Imagine y_beam_centre = 2
        y_beam_centre = self.beam_centre[0]

        # Now pixel_offsets -= ((11-1) - 2)
        # => pixel_offsets = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
        # This is good! The top of the detector is above the centre in y.
        pixel_offsets -= ((num_y_pixels-1) - y_beam_centre)

        # Save this value to an array with the same shape as the images.
        self._vertical_pixel_offsets = np.zeros(image_shape, np.float32)
        for i, pixel_offset in enumerate(pixel_offsets):
            self._vertical_pixel_offsets[i, :] = pixel_offset

    def _init_horizontal_pixel_offsets(self, image_shape: int = None):
        """
        Initializes the array of relative horizontal pixel offsets, i.e. the
        distance (in units of pixels) between each pixel and the horizontal
        beam centre.
        """
        if image_shape is None:
            image_shape = self.data_file.image_shape

        # Follow the recipe from above.
        # The azimuthal angle is larger towards the left of the image.
        num_x_pixels = image_shape[1]
        pixel_offsets = np.arange(num_x_pixels-1, -1, -1)
        x_beam_centre = self.beam_centre[1]
        pixel_offsets -= ((num_x_pixels-1) - x_beam_centre)

        # Save this value to an array with the same shape as the images.
        self._horizontal_pixel_offsets = np.zeros(image_shape, np.float32)
        for i, pixel_offset in enumerate(pixel_offsets):
            self._horizontal_pixel_offsets[:, i] = pixel_offset

    def _init_relative_polar(self, image_shape: int = None):
        """
        Initializes the relative_polar array.
        """
        if image_shape is None:
            image_shape = self.data_file.image_shape
        # First we want to calculate pixel offsets.
        num_y_pixels = image_shape[0]
        # Imagine num_y_pixels = 11.
        # pixel_offsets = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        pixel_offsets = np.arange(num_y_pixels-1, -1, -1)
        # Imagine y_beam_centre = 2
        y_beam_centre = self.beam_centre[0]

        # Now pixel_offsets -= ((11-1) - 2)
        # => pixel_offsets = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
        # This is good! The top of the detector is above the centre in theta.
        pixel_offsets -= ((num_y_pixels-1) - y_beam_centre)

        # Now convert pixel number to distances
        # Imagine self.pixel_size = 0.1m
        # Now distance_offsets = [0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5...]
        distance_offsets = pixel_offsets*self.data_file.pixel_size

        # Now do a trig
        #   |
        #   |
        #   |
        #   |¬                        \ <- Theta measured from here.
        #   ---------------------------
        #                ^ This distance is detector distance.
        theta_offsets = np.arctan2(distance_offsets,
                                   self.data_file.detector_distance)

        # Now use these offsets to initialize the relative_polar array.
        self._relative_polar = np.zeros(image_shape)
        for i, theta_offset in enumerate(theta_offsets):
            self._relative_polar[i, :] = -theta_offset

    def _init_relative_azimuth(self, image_shape: int = None):
        """
        Initializes the relative_azimuth array.
        """
        if image_shape is None:
            image_shape = self.data_file.image_shape

        # Follow the recipe from above.
        # The azimuthal angle is larger towards the left of the image.
        num_x_pixels = image_shape[1]
        pixel_offsets = np.arange(num_x_pixels-1, -1, -1)
        x_beam_centre = self.beam_centre[1]
        pixel_offsets -= ((num_x_pixels-1) - x_beam_centre)

        # Now convert from pixels to distances to angles.
        distance_offsets = pixel_offsets*self.data_file.pixel_size
        phi_offsets = np.arctan2(distance_offsets,
                                 self.data_file.detector_distance)

        # Now use these offsets to initialize the relative_azimuth array
        self._relative_azimuth = np.zeros(image_shape)
        for i, column in enumerate(phi_offsets):
            self._relative_azimuth[:, i] = column
