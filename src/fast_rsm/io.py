"""
This module contains parsers for different instruments that return Scan objects.
"""

from pathlib import Path
from typing import Union, Tuple

from diffraction_utils import I07Nexus, Frame, Vector3
from diffraction_utils.diffractometers import I07Diffractometer

from .rsm_metadata import RSMMetadata
from .scan import Scan


def from_i07(path_to_nx: Union[str, Path],
             beam_centre: Tuple[int],
             detector_distance: float,
             setup: str,
             path_to_data: str = ''):
    """
    Instantiates a Scan from the path to an I07 nexus file, a beam centre
    coordinate tuple, a detector distance and a sample out-of-plane vector.

    Args:
        path_to_nx:
            Path to the nexus file containing the scan metadata.
        beam_centre:
            A (y, x) tuple of the beam centre, measured in the usual image
            coordinate system, in units of pixels.
        detector_distance:
            The distance between the sample and the detector.
        setup:
            What was the experimental setup? Can be "vertical", "horizontal"
            or "DCD".
        path_to_data:
            Path to the directory in which the images are stored. Defaults
            to '', in which case a bunch of reasonable directories will be
            searched for the images. This is useful in case you store the small
            .nxs file in a different place to the potentially very large image
            data (e.g. .nxs files on a local disc, .h5 files on portable hard
            drive).

    Returns:
        Corresponding instance of fast_rsm.scan.Scan
    """
    # Load the nexus file.
    i07_nexus = I07Nexus(path_to_nx, path_to_data,
                         detector_distance, setup)

    # Not used at the moment, but not deleted in case full UB matrix
    # calculations become important in the future (in which case we'll also
    # need to supply a second value).
    sample_oop = Vector3([0, 1, 0], Frame(Frame.sample_holder, None, None))

    # Load the state of the diffractometer; prepare the RSM metadata.
    diff = I07Diffractometer(i07_nexus, sample_oop, setup)
    metadata = RSMMetadata(diff, beam_centre)

    # Make sure that the sample_oop vector's frame's diffractometer is good.
    sample_oop.frame.diffractometer = diff

    return Scan(metadata)
