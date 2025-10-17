"""
The purpose of this module is to provide routines for running some
pre-processing analysis on the data. This was motivated by the need to work out
e.g. reciprocal space bounds on scans in a performant way. The decision was made
to move this into the core of the module.
"""

from typing import TYPE_CHECKING
import numpy as np


from fast_rsm.scan import Scan
if TYPE_CHECKING:
    from fast_rsm.experiment import Experiment


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
    side_lengths = stop - start
    volume = side_lengths[0] * side_lengths[1] * side_lengths[2]

    # Note that there are 2^20 bytes in a megabyte.
    return np.array([np.cbrt(4 * volume / (file_size * (2**20)))] * 3)


def _find_exc_broken_frames(scan: Scan):
    """
    Takes a scan recorded using i07's excalibur detector, which is a bit broken.
    Attempts to work out which frames are broken. Returns a list of image
    indices corresponding to the frames that we think are broken.

    For now this simple test relies on the broken frames being much brighter
    than its preceding frames.
    """
    # I think it's pretty clean to use the _raw_data directly in this case.
    # pylint: disable=protected-access

    # Don't do anything if this wasn't an i07 excalibur scan.
    try:
        if not scan.metadata.data_file.is_excalibur:
            return []
    except AttributeError:
        return []

    broken_frames = []

    # An intensity jump is suspicious if it's a factor big_intensity_jump more.
    big_intensity_jump = 20
    previous_mean = np.mean(scan.load_image(0)._raw_data)
    for i in range(scan.metadata.data_file.scan_length):
        current_mean = np.mean(scan.load_image(i)._raw_data)

        if current_mean < previous_mean * big_intensity_jump:
            # This is not a suspicious jump in counts.
            previous_mean = current_mean
        else:
            # This is suspicious.
            broken_frames.append(i)
            print(f"Image with index {i} has suspiciously high counts. "
                  "It is likely broken and will be ignored.")

    return broken_frames


def skip_i07_exc_broken_frames(experiment: 'Experiment'):
    """
    This is the public facing function that can be used to skip all dodgy
    frames captured by i07's excalibur detector.
    """
    for scan in experiment.scans:
        scan.skip_images.extend(_find_exc_broken_frames(scan))
