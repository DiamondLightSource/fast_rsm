"""
This module contains the scan class, that will be used to store all of the
information relating to a reciprocal space scan.
"""

from pathlib import Path
from typing import List, Callable, Union, Tuple

from .image import Image
from .metadata import Metadata


class Scan:
    """
    This class stores all of the data and metadata relating to a reciprocal
    space map.

    Attrs:
        Images:
            The images collected in the scan.
        Metadata:
            Scan metadata.
    """

    def __init__(self, images: List[Image], metadata: Metadata):
        self.images = images
        self.metadata = metadata

        self._rsm = None

    @property
    def reciprocal_space_map(self):
        """
        Returns a full RSM from this scan's constutuent images.
        """
        if self._rsm is None:
            self._init_rsm()

        return self._rsm

    def _init_rsm(self):
        """
        Initializes the scan's reciprocal space map.
        """
        raise NotImplementedError()

    @classmethod
    def from_file(cls,
                  file_path: Union[str, Path],
                  parser: Callable,
                  beam_centre: Tuple[int] = None,
                  detector_distance: float = None):
        """
        Returns an instance of Scan from the path to a data file and a parser
        that can be used to parse the data file. Parser functions can be found
        in RSMapper.io.

        Args:
            file_path:
                Path to the file to load.
            parser:
                The parser that we'll use to parse the file. These can be found
                in the RSMapper.io module.
            beam_centre:
                The central pixel.
            detector_distance:
                The distance between the sample and the detector.

        Returns:
            An instance of Scan.
        """
        # Use the parser to grab this scan's images and metadata; call __init__.
        if beam_centre is None:
            imgs, metadata = parser(file_path)
        else:
            imgs, metadata = parser(file_path, beam_centre, detector_distance)
        return cls(imgs, metadata)
