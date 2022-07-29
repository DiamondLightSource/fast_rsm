"""
The purpose of this module is to provide routines for running some
pre-processing analysis on the data. This was motivated by the need to work out
e.g. reciprocal space bounds on scans in a performant way. The decision was made
to move this into the core of the module.
"""

import numpy as np

from diffraction_utils import Frame

from .scan import Scan


def find_q_bounds(scan: Scan, frame: Frame):
    """
    Takes an instance of fast_rsm.scan.Scan, and works out the region of
    reciprocal space sampled by this scan.

    Args:
        scan:
            The scan whose q_bounds we're interested in.
        frame:
            The frame of reference in which we want to calculate the bounds.

    Returns:
        (start, stop), where start and stop are numpy arrays with shape (3,)
    """

    top_left = (0, 0)
    top_right = (0, -1)
    bottom_left = (-1, 0)
    bottom_right = (-1, -1)
    poni = scan.metadata.beam_centre
    extremal_q_points = np.array(
        [top_left, top_right, bottom_left, bottom_right, poni])
    extremal_q_points = (extremal_q_points[:, 0], extremal_q_points[:, 1])

    # Get some sort of starting value.
    img = scan.load_image(0, load_data=False)
    q_vec = img.q_vectors(frame, poni)

    start, stop = q_vec, q_vec

    # Iterate over every image in the scan.
    for i in range(scan.metadata.data_file.scan_length):
        # Instantiate an image without loading its data.
        img = scan.load_image(i, load_data=False)

        # Work out all the extreme q values for this image.
        q_vecs = img.q_vectors(frame, extremal_q_points)

        # Get the min/max of each component.
        min_q = np.array([np.amin(q_vecs[:, i]) for i in range(3)])
        max_q = np.array([np.amax(q_vecs[:, i]) for i in range(3)])

        # Update start/stop accordingly.
        start = [min_q[x] if min_q[x] < start[x] else start[x]
                 for x in range(3)]
        stop = [max_q[x] if max_q[x] > stop[x] else stop[x] for x in range(3)]

    # Give a bit of wiggle room. For now, I'm using 5% padding, but this was
    # chosen arbitrarily.
    start, stop = np.array(start), np.array(stop)
    side_lengths = stop - start
    padding = side_lengths/20
    start -= padding
    stop += padding

    return start, stop


def get_step_from_filesize(start: np.ndarray,
                           stop: np.ndarray,
                           file_size: float = 100) -> np.ndarray:
    """
    Takes a requested file size, a start and a stop. Works out the step that
    will give you a resultant file of your requested filesize. To calculate
    start and stop you can use fast_rsm.meta_analysis.find_q_bounds, or
    calculate it yourself.

    Args:
        start:
            array-like [qx_min qy_min qz_min]
        stop:
            array-like [qx_max qy_max qz_max]
        file_size:
            Requested output filesize in MB. Defaults to 100 MB.

    Returns:
        A numpy array of [step step step].
    """
    side_lengths = stop-start
    volume = side_lengths[0]*side_lengths[1]*side_lengths[2]

    return np.array([np.cbrt(4*volume/(file_size*1e6))]*3)
