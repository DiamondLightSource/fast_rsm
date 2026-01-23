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
def path_to_resources():
    """
    Returns the path to the test resources folder.
    """
    return Path(__file__).parent / "resources"