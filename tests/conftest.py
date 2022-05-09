"""
The conftest file contains fixtures for RSMappers tests.
"""

# Basically always needed in conftest files.
# pylint: disable=redefined-outer-name

import os

from pytest import fixture

from diffraction_utils import I10Nexus
from diffraction_utils.diffractometers import I10RasorDiffractometer

from RSMapper.rsm_metadata import RSMMetadata


@fixture
def i10_nxs_path(path_to_resources: str) -> str:
    """
    Returns a path to a .nxs file acquired at beamline i10 in 2022.
    """
    return path_to_resources + "i10-693862.nxs"


@fixture
def i10_nxs_parser(i10_nxs_path: str) -> I10Nexus:
    """
    Returns an instance of I10Nexus.
    """
    return I10Nexus(i10_nxs_path,
                    detector_distance=0.1363)  # Distance to pimte cam in RASOR.


@fixture
def rasor(i10_nxs_parser: I10Nexus) -> I10RasorDiffractometer:
    """
    Returns an instance of I10RasorDiffractometer.
    """
    return I10RasorDiffractometer(i10_nxs_parser, [0, 1, 0],
                                  I10RasorDiffractometer.area_detector)


@fixture
def i10_metadata(rasor) -> RSMMetadata:
    """
    Returns an instance of RSMMetadata corresponding to the above i10 .nxs
    fixtures.
    """
    return RSMMetadata(rasor, (998, 1016))


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


@fixture(scope='session')
def i10_nx_01(path_to_resources):
    """
    Returns the path to an i10 nexus file.
    """
    return path_to_resources + "i10-693862.nxs"


@fixture(scope='session')
def i10_beam_centre_01():
    """
    Beam centre for the above nexus file.
    """
    return 1000, 1000
