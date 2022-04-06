"""
This module contains tools used for parsing data/metadata files.
"""

from typing import Union, Tuple, TYPE_CHECKING, List
from pathlib import Path

if TYPE_CHECKING:
    from .image import Image
    from .metadata import Metadata


def i07_nexus_parser(path_to_nx: Union[str, Path]) -> \
        Tuple[List['Image'], 'Metadata']:
    """
    Parses an I07 nexus file. Returns everything required to instantiate a Scan.

    Args:
        path_to_nx:
            Path to the nexus file to parse.

    Returns:
        A tuple taking the form (list_of_images, metadata). This can be used to
        instantiate a Scan.
    """
    # Just use some hard-coded paths to grab the data.
    # It doesn't need to be pretty; it needs to work.
