"""
The conftest file contains fixtures for RSMappers tests.
"""

from pytest import fixture

from RSMapper.metadata import Metadata


@fixture
def metadata_01():
    """
    Returns an instance of metadata with fairly arbitrarily chosen parameters.
    """
    return Metadata(
        detector_distance=0.2,  # 20 cm detector distance.
        pixel_size=1e-5,  # 10 µm pixel size (easy number, not unrealistic).
        energy=10e3,  # 10 KeV photons, easy & realistic number.
        data_shape=(2000, 2000),  # A 2kx2k camera.
        beam_centre=(20, 80)  # Beam's poni is towards the top left.
    )
