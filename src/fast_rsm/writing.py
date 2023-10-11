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
    x_range, y_range , z_range = tuple(
        np.linspace(start[i], stop[i]+step[i], num=binned_data.shape[i],
                    dtype=np.float32) for i in range(3)
    )

    gridToVTK(file_path, x_range, y_range, z_range,
              cellData={"Intensity": binned_data})
