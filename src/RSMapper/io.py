"""
This module contains tools used for parsing data/metadata files.
"""

# Because of the dumb way that values are stored in the nexusformat package.
# pylint: disable=protected-access

from typing import Union, Tuple, List
from pathlib import Path

import nexusformat.nexus as nx

from .image import Image
from .metadata import Metadata
from .motors import Motors


def i07_nexus_parser(path_to_nx: Union[str, Path], beam_centre: Tuple[int]) -> \
        Tuple[List['Image'], 'Metadata']:
    """
    Parses an I07 nexus file. Returns everything required to instantiate a Scan.

    Args:
        path_to_nx:
            Path to the nexus file to parse.
        beam_centre:
            The beam centre when all axes are zeroed.

    Returns:
        A tuple taking the form (list_of_images, metadata). This can be used to
        instantiate a Scan.
    """
    nx_file = nx.nxload(path_to_nx)
    # Just use some hard-coded paths to grab the data.
    # It doesn't need to be pretty; it needs to work.
    detector_distance = nx_file["/entry/instrument/diff1detdist/value"]._value
    energy = nx_file["/entry/instrument/dcm1energy/value"]._value

    default = str(nx_file["/entry/"].get_default())
    image_path = nx_file[
        f"/entry/instrument/{default}/data_file/file_name"]._value

    # Now we need to do some detector specific stuff.
    if 'pil' in default:
        # It's the pilatus detector.
        pixel_size = 172e-6
        data_shape = [1475, 1679]
        metadata = Metadata(nx_file, "i07", detector_distance, pixel_size,
                            energy, data_shape, beam_centre)
        motors = Motors(metadata)
        images = [Image.from_file(x, motors, metadata) for x in image_path]
    else:
        # It's the excalibur detector.
        raise NotImplementedError()

    return images, Metadata(nx_file, "i07", detector_distance, pixel_size,
                            energy, data_shape, beam_centre)
