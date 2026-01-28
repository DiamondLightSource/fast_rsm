"""
The conftest file contains fixtures for fast_rsms tests.
"""

# Basically always needed in conftest files.
# pylint: disable=redefined-outer-name

import os

from pytest import fixture
from pathlib import Path
from fast_rsm.rsm_metadata import RSMMetadata
from types import SimpleNamespace
from pytest import fixture

from fast_rsm.config_loader import experiment_config

@fixture
def testRSM():
    test_diff=SimpleNamespace(\
        data_file=SimpleNamespace(image_shape=(100,200),\
                                  is_rotated=True))
    beam_centre=(20,40)
    return RSMMetadata(test_diff,beam_centre) 

@fixture
def test_default_config():
    """
    returns a default_config dict with scan_numbers=[1234]    
    """
    defaultconfig=experiment_config([1234])
    defaultconfig['full_path']=__file__
    return defaultconfig

@fixture
def path_to_resources():
    """
    Returns the path to the test resources folder.
    """
    return Path(__file__).parent / "resources"