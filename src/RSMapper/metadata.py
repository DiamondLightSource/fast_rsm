"""
This module contains the metadata class, which provides a python interface
for the .nxs file written for the scan.
"""


from typing import Tuple

import numpy as np
from scipy.constants import physical_constants


class Metadata:
    """
    This class contains all of the information stored in the .nxs file. It also
    contains some convenience methods/properties for the manipulation of .nxs
    files.
    """

    def __init__(self, detector_distance: float, pixel_size: float,
                 data_shape: Tuple[int, int], beam_centre: Tuple[int, int]):
        self.detector_distance = detector_distance
        self.pixel_size = pixel_size
        self.beam_centre = beam_centre
        self.data_shape = data_shape

        self._relative_theta = None
        self._relative_chi = None
        self._solid_angles = None

    def _correct_beam_centre(self):
        """
        Correct the beam centre, if necessary. We can use the metadata to work
        out where the data was acquired. Some data sources might report the
        beam_centre using a different convention; this is where we correct for
        that to make sure that that data[0, 0] is the top left pixel,
        data[-1, 0] is the top right pixel.

        This means that, if the beam was centered at the top left pixel, the
        user should enter that the beam_centre=[0, 0]. This seems unlikely; this
        would probably be given as [1, 1] in typical software!
        """
        raise NotImplementedError()

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
    def relative_theta(self):
        """
        This property makes accessing the detector's relative theta from the
        beam centre for each pixel, when all detector rotation motors have been
        zeroed.
        """
        if self._relative_theta is None:
            self._init_relative_theta()
        return self._relative_theta

    @property
    def relative_chi(self):
        """
        This property makes accessing the detector's relative chi from the
        beam centre for each pixel, when all detector rotation motors have been
        zeroed.
        """
        if self._relative_chi is None:
            self._init_relative_chi()
        return self._relative_chi

    @property
    def energy(self):
        """
        Parses the nexus file; returns the incident beam energy.
        """
        raise NotImplementedError()

    @property
    def q_incident_lenth(self):
        """
        Returns the momentum of the incident beam.
        """
        return self.energy/physical_constants["reduced Planck constant"]

    def _init_solid_angles(self):
        """
        Initialize the solid angles property. Note that this is not an exact
        calculation, but all the values should be correct to within a few %.

        Also note that this implementation is an awful hack.
        """
        # We're going to need to inc the data shape to hack this.
        self.data_shape = self.data_shape[0], self.data_shape[1]+1
        self._init_relative_theta()
        theta_diffs = np.copy(self.relative_theta)
        theta_diffs = np.diff(theta_diffs, axis=1)

        self.data_shape = self.data_shape[0]+1, self.data_shape[1]-1

        self._init_relative_chi()
        chi_diffs = np.copy(self._relative_chi)
        chi_diffs = np.diff(chi_diffs, axis=0)

        # Now return the shape back to normal.
        self.data_shape = self.data_shape[0]-1, self.data_shape[1]
        self._init_relative_theta()
        self._init_relative_chi()

        # And finally, do what we came here to do.
        self._solid_angles = chi_diffs*theta_diffs

    def _init_relative_theta(self):
        """
        Initializes the relative_theta array.
        """
        # First we want to calculate pixel offsets.
        num_y_pixels = self.data_shape[1]
        # Imagine num_y_pixels = 11.
        # pixel_offsets = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        pixel_offsets = np.arange(num_y_pixels-1, -1, -1)
        # Imagine y_beam_centre = 2
        y_beam_centre = self.beam_centre[1]

        # Now pixel_offsets -= ((11-1) - 2)
        # => pixel_offsets = [2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]
        # This is good! The top of the detector is above the centre in theta.
        pixel_offsets -= ((num_y_pixels-1) - y_beam_centre)

        # Now convert pixel number to distances
        # Imagine self.pixel_size = 0.1m
        # Now distance_offsets = [0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.4, -0.5...]
        distance_offsets = pixel_offsets*self.pixel_size

        # Now do a trig
        #   |
        #   |
        #   |
        #   |¬                        \ <- Theta measured from here.
        #   ---------------------------
        #                ^ This distance is detector distance.
        theta_offsets = np.arctan(distance_offsets/self.detector_distance)

        # Now use these offsets to initialize the relative_theta array.
        self._relative_theta = np.zeros(self.data_shape)
        for i, row in enumerate(theta_offsets):
            self._relative_theta[:, i] = row

    def _init_relative_chi(self):
        """
        Initializes the relative_chi array.
        """
        # Follow the recipe from above.
        # Because we don't need to invert any axes, this is easier to follow.
        num_x_pixels = self.data_shape[1]
        pixel_offsets = np.arange(0, num_x_pixels)
        x_beam_centre = self.beam_centre[0]
        pixel_offsets -= x_beam_centre

        # Now convert from pixels to distances to angles.
        distance_offsets = pixel_offsets*self.pixel_size
        chi_offsets = np.arctan(distance_offsets/self.detector_distance)

        # Now use these offsets to initialize the relative_chi array
        self._relative_chi = np.zeros(self.data_shape)
        for i, column in enumerate(chi_offsets):
            self._relative_chi[i, :] = column
