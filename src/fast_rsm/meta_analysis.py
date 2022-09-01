"""
The purpose of this module is to provide routines for running some
pre-processing analysis on the data. This was motivated by the need to work out
e.g. reciprocal space bounds on scans in a performant way. The decision was made
to move this into the core of the module.
"""

import numpy as np


def get_step_from_filesize(start: np.ndarray,
                           stop: np.ndarray,
                           file_size: float = 100) -> np.ndarray:
    """
    Takes a requested file size, a start and a stop. Works out the step that
    will give you a resultant file of your requested filesize. To calculate
    start and stop you can use Scan.q_bounds, or calculate it yourself.

    Args:
        start:
            array-like [qx_min qy_min qz_min]
        stop:
            array-like [qx_max qy_max qz_max]
        file_size:
            Requested output filesize in MB. Defaults to 100 MB.

    Returns:
        A numpy array of [step step step] (since step_x=step_y=step_z).
    """
    side_lengths = stop-start
    volume = side_lengths[0]*side_lengths[1]*side_lengths[2]

    # Note that there are 2^20 bytes in a megabyte.
    return np.array([np.cbrt(4*volume/(file_size*(2**20)))]*3)
