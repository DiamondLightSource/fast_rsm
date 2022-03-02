"""
This module contains the metadata class, which provides a python interface
for the .nxs file written for the scan.
"""


class Metadata:
    """
    This class contains all of the information stored in the .nxs file. It also
    contains some convenience methods/properties for the manipulation of .nxs
    files.
    """

    def __init__(self, detector_distance: float, pixel_size: float) -> None:
        self.detector_distance = detector_distance
        self.pixel_size = pixel_size
