"""
This module stores the Vector class, which is a light wrapper around a numpy
array. It contains some convenience methods/attributes for dealing with
coordinate system changes.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation


if TYPE_CHECKING:
    from .frame_of_reference import Frame
    from .diffractometer_base import DiffractometerBase


class Vector3:
    """
    This class is a light wrapper around a numpy array, with some convenience
    methods/attributes for dealing with coordinate system changes.
    """

    def __init__(self, array: np.ndarray, frame: 'Frame'):
        self.array = np.array(array)
        self.frame = frame

    @property
    def azimuthal_angle(self):
        """
        Returns this vector's azimuthal angle in its current reference frame.
        """
        return np.arctan2(self.array[0], self.array[2])

    @property
    def polar_angle(self):
        """
        Returns this vector's polar angle in its current reference frame.
        """
        return np.arccos(self.unit[1])

    @property
    def unit(self):
        """
        Returns the unit vector parallel to this Vector3.
        """
        return self.array/np.linalg.norm(self.array)

    def to_frame(self, frame: 'Frame',
                 diffractometer: 'DiffractometerBase' = None):
        """
        Transforms to a frame with name `frame`.

        Args:
            frame:
                The name of the frame of reference to transform to.
            diffractometer (optional):
                The diffractometer we should use to carry out the
                transformation. If this is None, frame.diffractometer is used.
        """
        if diffractometer is None:
            diffractometer = frame.diffractometer
        diffractometer.rotate_vector_to_frame(self, frame)

    @classmethod
    def from_angles(cls,
                    azimuth: float,
                    polar: float,
                    frame: 'Frame',
                    length=1.0):
        """
        Constructs a new Vector3 from an azimuthal angle, a polar angle and a
        frame of reference.

        Args:
            azimuth:
                The azimuthal angle of the vector to create.
            polar:
                The polar angle of the vector to create.
            frame:
                The frame of reference our new vector will be in.
            length:
                The length of the new vector. Defaults to 1.0.
        """
        array = length * np.array([
            np.sin(polar)*np.sin(azimuth),
            np.cos(polar),
            np.sin(polar)*np.cos(azimuth)
        ])
        return cls(array, frame)


def rot_from_a_to_b(vector_a: Vector3, vector_b: Vector3):
    """
    Generates a rotation that will rotate arrays parallel to vector_a so that
    they're parallel to vector_b.

    TODO: This should handle instance of Vector3 with non-orthonormal basis
        vectors.

    Args:
        vector_a:
            The vector we want to rotate from.
        vector_b:
            The vector we want to rotate to.

    Raises:
        ValueError if vector_a and vector_b aren't in the same frame.

    Returns:
        An instance of scipy.spatial.transform.Rotation that will rotate vectors
        parallel to vector_a so that they're parallel to vector_b.
    """
    if vector_a.frame != vector_b.frame:
        raise ValueError(
            "Vectors must be in the same frame to calculate a rotation "
            "between them.")
    return _rot_arr_from_a_to_b(vector_a.array, vector_b.array)


def _rot_arr_from_a_to_b(array_a: np.ndarray, array_b: np.ndarray):
    """
    Generates a rotation that will rotate arrays parallel to array_a so that
    they're parallel to array_b.

    Args:
        array_a:
            The array we want to rotate from.
        array_b:
            The array we want to rotate to.
    """
    vec_a_unit = np.array(array_a)/np.linalg.norm(np.array(array_a))
    vec_b_unit = np.array(array_b)/np.linalg.norm(np.array(array_b))
    cross = np.cross(vec_a_unit, vec_b_unit)

    # The sine of the angle between vector_a and vector_b.
    sin_angle_ab = np.linalg.norm(cross)
    angle_ab = np.arcsin(sin_angle_ab)  # And the corresponding angle.

    # Don't divide by zero!
    if angle_ab == 0:
        return Rotation.from_rotvec([0, 0, 0])

    # Now calculate the rotvec.
    rot_axis_normal = cross/np.linalg.norm(cross)
    rotvec = rot_axis_normal*angle_ab

    # Return the rotvec's corresponding rotation.
    return Rotation.from_rotvec(rotvec)
