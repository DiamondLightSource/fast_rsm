"""
Tests the "map_to_q" CLI.
"""

# pylint: disable=invalid-name

import os
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose


def test_map_to_q_i07_421595(path_to_resources: str, path_to_i07_cli: str,
                             tmp_path: Path):
    """
    Make sure that map_to_q is giving exactly the correct output for i07 scan
    number 421595. At one point in time (30/05/2022) it was known with relative
    certainty that map_to_q was outputting correct .dat files. The point of this
    test is to make sure that, in the future, changes made to the core don't
    affect the output of map_to_q.
    """
    # Set up various essential paths.
    tmp_path_str = str(tmp_path.absolute())
    data_path = "/Users/richard/Data/i07/rsm_soller_test/421595/"
    min_thresh = 0
    max_thresh = 2e50
    det_dist = 0.5026

    # Make sure that our temporary path is empty.
    assert len(os.listdir(tmp_path)) == 0

    # Run the CLI.
    os.system(
        f"python {path_to_i07_cli}/map_to_q.py -x 739 -y 1329 -D {det_dist} "
        f"-o {tmp_path_str} -d {data_path} -N 421595 "
        f"--min_thresh {min_thresh} --max_thresh {max_thresh}")

    # Grab the newly created file (our tmp path should otherwise be empty).
    new_file = os.listdir(tmp_path)[0]
    new_data = np.loadtxt(tmp_path / new_file)

    # Grab the previously calculated data (the stuff we know to be correct).
    old_file = path_to_resources + "421595_mapped.dat"
    old_data = np.loadtxt(old_file)

    # Make sure nothing has changed in the way we calculate the data.
    assert_allclose(new_data, old_data)
