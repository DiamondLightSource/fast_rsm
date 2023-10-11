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
    # This is needed by the pyevtk library.
    file_path = str(file_path)

    # Coordinates

    x_range = np.arange(start[0], stop[0], step[0], dtype="float32")
    x_range = list(x_range)
    x_range.append(stop[0])
    x_range = np.array(x_range, dtype=np.float32)

    y_range = np.arange(start[1], stop[1], step[1], dtype="float32")
    y_range = list(y_range)
    y_range.append(stop[1])
    y_range = np.array(y_range, dtype=np.float32)

    z_range = np.arange(start[2], stop[2], step[2], dtype="float32")
    z_range = list(z_range)
    z_range.append(stop[2])
    z_range = np.array(z_range, dtype=np.float32)

    gridToVTK(file_path, x_range, y_range, z_range,
              cellData={"Intensity": binned_data})
