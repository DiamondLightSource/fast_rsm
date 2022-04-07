"""
The conftest file contains fixtures for RSMappers tests.
"""

# Basically always needed in conftest files.
# pylint: disable=redefined-outer-name

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
def path_to_resources():
    """
    Returns the path to the test resources folder.
    """
    if os.path.exists("tests/resources/i07-418550.nxs"):
        return "tests/resources/"
    return "resources/"


@fixture
def path_to_i07_nx_01(path_to_resources):
    """
    Returns the path to the i07 nexus file. This is a fixture for future
    proofing reasons (i.e. maybe one day some parsing has to be done to locate
    the file).
    """
    return path_to_resources + "i07-418550.nxs"


@fixture
def i07_beam_centre_01():
    """
    Returns the position of the beam centre that was recorded during the
    experiment in which the above nexus file was written.
    """
    return 731, 1329


@fixture
def i07_detector_distance_01():
    """
    The distance between the sample and the detector in the experiment in which
    the above nexus file was written. Note that diff1detdist in the nexus file
    is a red herring.
    """
    return 0.5026


@fixture
def i10_nx_01(path_to_resources):
    """
    Returns the path to an i10 nexus file.
    """
    return path_to_resources + "i10-693862.nxs"


@fixture
def i10_beam_centre_01():
    """
    Beam centre for the above nexus file.
    """
    return 1000, 1000


@fixture
def i10_pimte_detector_distance():
    """
    Returns the distance between a sample and the pimte camera in RASOR.
    """
    return 0.1363
