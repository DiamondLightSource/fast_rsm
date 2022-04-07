"""
This module contains tests for the parsing functions in the RSMapper.io module.
"""

# Obviously we want to also test protected members.
# pylint: disable=protected-access

import nexusformat.nexus as nx
import numpy as np
from PIL import Image as PILImage

from RSMapper.io import i07_nexus_parser


def test_i07_nexus_parser_metadata(path_to_i07_nx_01: str,
                                   i07_beam_centre_01: tuple,
                                   i07_detector_distance_01: float):
    """
    Make sure that the I07 nexus parser can return sensible metadata.
    """
    _, metadata = i07_nexus_parser(path_to_i07_nx_01,
                                   i07_beam_centre_01,
                                   i07_detector_distance_01)

    assert metadata.beam_centre == (731, 1329)
    assert metadata.data_shape == (1475, 1679)
    assert metadata.detector_distance == 0.5026
    assert metadata.energy == 20e3
    assert metadata.instrument == "i07"
    assert metadata.pixel_size == 172e-6
    # If you're wondering why this test is slow, this is why:
    assert metadata.metadata_file.tree == nx.nxload(path_to_i07_nx_01).tree


def test_i07_nexus_parser_images(path_to_resources: str,
                                 path_to_i07_nx_01: str,
                                 i07_beam_centre_01: tuple,
                                 i07_detector_distance_01: float):
    """
    Make sure that the I07 nexus parser returns proper Image objects.
    """
    images, _ = i07_nexus_parser(path_to_i07_nx_01,
                                 i07_beam_centre_01,
                                 i07_detector_distance_01)

    image_names = [
        "p2mImage_418550_770618.tif",
        "p2mImage_418550_770619.tif",
        "p2mImage_418550_770620.tif",
        "p2mImage_418550_770621.tif"
    ]
    image_paths = [path_to_resources + x for x in image_names]

    for i, image in enumerate(images):
        with PILImage.open(image_paths[i]) as open_image:
            true_img_array = np.array(open_image)
            assert (true_img_array == image._raw_data).all()
