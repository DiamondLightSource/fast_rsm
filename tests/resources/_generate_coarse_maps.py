"""
This module contains functions that can be used to generate the coarse map
resources that are used in some of the system tests.
"""

from time import time

import numpy as np

from diffraction_utils import Frame, Vector3

from RSMapper.writing import linear_bin_to_vtk
from RSMapper.scan import Scan


def generate_i10_693862_coarse_map() -> None:
    """
    Generates one of the coarse maps needed for the tests. This will only work
    when "RSMapper/tests/resources/" is the current working directory.
    """
    # The necessary setup.
    sample_oop = Vector3([0, 1, 0], Frame(Frame.sample_holder))
    i10_scan = Scan.from_i10(
        "i10-693862.nxs", (998, 1016), 0.1363, sample_oop, "")
    frame = Frame(Frame.hkl, i10_scan.metadata.diffractometer)
    start = np.array([-0.0015, 0.11, -0.005])
    stop = np.array([0.002, 0.115, 0.003])
    step = np.array([0.0002, 0.0002, 0.0002])

    # Do the reciprocal space map. Time it, just for fun.
    time1 = time()
    rsm = i10_scan.binned_reciprocal_space_map(frame, start, stop, step, 10)
    time2 = time()
    time_taken = time2 - time1

    num_images = i10_scan.metadata.data_file.scan_length
    time_per_image = time_taken/num_images
    print(f"Time taken to generate map per image: {time_per_image} s.")

    # The file we'll need for the tests.
    np.save("i10_693862_coarse_map", rsm)
    # Generate a .vtr too (always worth visually double checking on paraview).
    linear_bin_to_vtk(rsm, "i10_693862_coarse_map", start, stop, step)


def generate_i07_421595_coarse_map() -> None:
    """
    Generates one of the coarse maps needed for the tests. This will only work
    when "RSMapper/tests/resources/" is the current working directory.
    """
    # The necessary setup.
    beam_centre = (739, 1329)
    detector_distance = 502.6e-3
    setup = 'horizontal'
    sample_oop = [0, 1, 0]

    # Where I happen to have the data stored on my computer.
    path_to_data = \
        "/Users/richard/Data/i07/rsm_soller_test/421595/i07-421595.nxs"

    # Make the scan object.
    scan = Scan.from_i07(path_to_data, beam_centre, detector_distance, setup,
                         sample_oop, path_to_data)

    # Prepare to carry out the map in the lab frame.
    lab_frame = Frame(Frame.lab)
    start = np.array([0, -0.8, -0.9])
    stop = np.array([1.5, 0.1, 0])
    step = np.array([0.005, 0.005, 0.005])

    # Do the map. Time it, just for fun.
    time1 = time()
    rsmap = scan.binned_reciprocal_space_map(lab_frame, start, stop, step, 10)
    time2 = time()
    time_taken = time2 - time1
    num_images = scan.metadata.data_file.scan_length
    time_per_image = time_taken/num_images
    print(f"Time taken to generate map per image: {time_per_image} s.")

    # Do what we came here to do.
    np.save("i07_421595_coarse_map", rsmap)
    # Also save the


if __name__ == "__main__":
    # Uncomment whichever maps you dont want to generate.
    generate_i10_693862_coarse_map()
    generate_i07_421595_coarse_map()
