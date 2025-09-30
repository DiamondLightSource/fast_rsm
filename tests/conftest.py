"""
The conftest file contains fixtures for fast_rsms tests.
"""

# Basically always needed in conftest files.
# pylint: disable=redefined-outer-name

import os

from pytest import fixture

from diffraction_utils import I10Nexus, Vector3, Frame
from diffraction_utils.diffractometers import I10RasorDiffractometer

from fast_rsm.rsm_metadata import RSMMetadata
from fast_rsm.scan import Scan
from fast_rsm.config_loader import check_config_schema,experiment_config

@fixture
def test_default_config():
    """
    returns a default_config dict with scan_numbers=[1234]    
    """
    defaultconfig=experiment_config([1234])
    defaultconfig['full_path']=__file__
    return defaultconfig

@fixture
def path_to_frsm_example_data():
    """
    Returns a path to example fast_rsm data stored in dls/science
    """
    return "/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data/"


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


@fixture
def i10_scan(i10_nxs_path, path_to_resources) -> Scan:
    """
    Returns a full scan object over 141 images. Because of memory optimizations,
    this doesn't cost anything until a call to .get_image is made.
    """
    sample_oop = Vector3([0, 1, 0], Frame(Frame.sample_holder))
    return Scan.from_i10(i10_nxs_path, (998, 1016), 0.1363, sample_oop,
                         path_to_resources)


@fixture
def path_to_i07_cli(path_to_resources: str) -> str:
    """
    Returns the path to the i07 CLI programs.
    """
    return path_to_resources + "../../CLI/i07/"
