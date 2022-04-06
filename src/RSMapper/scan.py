"""
This module contains the scan class, that will be used to store all of the
information relating to a reciprocal space scan.
"""

from pathlib import Path
from typing import List, Callable, Union

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
    def from_file(cls, file_path: Union[str, Path], parser: Callable):
        """
        Returns an instance of Scan from the path to a data file and a parser
        that can be used to parse the data file. Parser functions can be found
        in RSMapper.io.
        """
        # Use the parser to grab this scan's images and metadata; call __init__.
        images, metadata = parser(file_path)
        return cls(images, metadata)
