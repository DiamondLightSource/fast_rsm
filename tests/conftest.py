"""
The conftest file contains fixtures for RSMappers tests.
"""

import os

from pytest import fixture

from RSMapper.metadata import Metadata


@fixture
def metadata_01():
    """
    Returns an instance of metadata with fairly arbitrarily chosen parameters.
    """
    return Metadata(
        None,  # The metadata file could really be anything, so leave it.
        instrument="my_instrument",  # The name of the instrument.
        detector_distance=0.2,  # 20 cm detector distance.
        pixel_size=1e-5,  # 10 µm pixel size (easy number, not unrealistic).
        energy=10e3,  # 10 KeV photons, easy & realistic number.
        data_shape=(2000, 2000),  # A 2kx2k camera.
        beam_centre=(20, 80)  # Beam's poni is towards the top left.
    )


@fixture
def path_to_i07_nx():
    """
    Returns the path to the i07 nexus file. This is a fixture for future
    proofing reasons (i.e. maybe one day some parsing has to be done to locate
    the file).
    """
    # Make it work if we're in the base dir or the test dir.
    if os.path.exists("tests/resources/i07-418550.nxs"):
        return "tests/resources/i07-418550.nxs"
    return "resources/i07-418550.nxs"
