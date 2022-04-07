"""
This module contains tests for the parsing functions in the RSMapper.io module.
"""

# Obviously we want to also test protected members.
# pylint: disable=protected-access

import nexusformat.nexus as nx
import numpy as np
from PIL import Image as PILImage

from RSMapper.io import i07_nexus_parser, i10_nxs_parser


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


def test_i07_nexus_parser_img_metadata(path_to_i07_nx_01: str,
                                       i07_beam_centre_01: tuple,
                                       i07_detector_distance_01: float):
    """
    Make sure that images are loaded with the proper metadata.
    """
    images, metadata = i07_nexus_parser(path_to_i07_nx_01,
                                        i07_beam_centre_01,
                                        i07_detector_distance_01)

    # These are lazy checks, but they do make sure that no metadata has gone
    # missing when i07_nexus_parser constructs an image.
    for image in images:
        assert image.motors.metadata.metadata_file.tree == \
            metadata.metadata_file.tree
        assert image.metadata.metadata_file.tree == \
            metadata.metadata_file.tree


def test_i10_nexus_parser_metadata(i10_nx_01: str,
                                   i10_beam_centre_01: tuple,
                                   i10_pimte_detector_distance: float):
    """
    Make sure that our parser makes a valid metadata file.
    """
    _, metadata = i10_nxs_parser(i10_nx_01, i10_beam_centre_01,
                                 i10_pimte_detector_distance)

    assert metadata.beam_centre == (1000, 1000)
    assert metadata.data_shape == (2000, 2000)
    assert metadata.detector_distance == 0.1363
    assert metadata.energy == 931.7725
    assert metadata.instrument == "i10"
    assert metadata.metadata_file.tree == nx.nxload(i10_nx_01).tree
    assert metadata.pixel_size == 13.5e-6


def test_i10_nexus_parser_images(path_to_resources: str,
                                 i10_nx_01: str,
                                 i10_beam_centre_01: tuple,
                                 i10_pimte_detector_distance: float):
    """
    Check that the image data is preserved. Only check one image (loading each
    image twice is redundant and this already takes long enough).
    """
    images, _ = i10_nxs_parser(i10_nx_01,
                               i10_beam_centre_01,
                               i10_pimte_detector_distance)

    path_to_img = path_to_resources + "693862-pimte-files/pimte-00009.tiff"
    with PILImage.open(path_to_img) as open_img:
        correct_array = np.array(open_img)
        assert (images[9]._raw_data == correct_array).all()
