"""
This module contains convenience functions for writing reciprocal space maps.
"""


import numpy as np
from pyevtk.hl import gridToVTK


def linear_bin_to_vtk(binned_data: np.ndarray,
                      file_path: str,
                      start: np.ndarray,
                      stop: np.ndarray,
                      step: np.ndarray) -> None:
    """
    Takes binned data and saves it to a .vtk file.
    """
    # Coordinates
    x_range = np.arange(start[0], stop[0] + step[0],
                        step[0], dtype="float64")
    y_range = np.arange(start[1], stop[1] + step[1],
                        step[1], dtype="float64")
    z_range = np.arange(start[2], stop[2] + step[2],
                        step[2], dtype="float64")

    gridToVTK(file_path, x_range, y_range, z_range,
              cellData={"Intensity": binned_data})
