"""
This module contains unit tests for the RSMapper.image.Image class.

As of 07/04/2022, I'm lazily grabbing images from scans. This could be improved
by manually loading instances of Image, which would just require a slightly
clever fixture.
"""

# pylint: disable=protected-access

from copy import deepcopy
from typing import Tuple, List


from RSMapper.image import Image
from RSMapper.metadata import Metadata
from RSMapper.scan import Scan


def test_data(i10_parser_output_01: Tuple[List[Image], Metadata]):
    """
    Make sure that image_instance.data is properly normalized.
    """
    images, _ = i10_parser_output_01
    image = deepcopy(images[0])
