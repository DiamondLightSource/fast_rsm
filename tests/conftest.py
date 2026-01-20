"""
The conftest file contains fixtures for fast_rsms tests.
"""

# Basically always needed in conftest files.
# pylint: disable=redefined-outer-name

import os

from pytest import fixture
from pathlib import Path


from fast_rsm.config_loader import experiment_config

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
def path_to_resources():
    """
    Returns the path to the test resources folder.
    """
    return Path(__file__).parent / "resources"


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
def path_to_i07_cli(path_to_resources: str) -> str:
    """
    Returns the path to the i07 CLI programs.
    """
    return path_to_resources + "../../CLI/i07/"
