"""
This module contains the scan class, that will be used to store all of the
information relating to a reciprocal space scan.
"""

from typing import List

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
