"""
This module contains the scan class, that will be used to store all of the
information relating to a reciprocal space scan.
"""

from pathlib import Path
from typing import List, Union, Tuple

import numpy as np

from diffraction_utils import I10Nexus
from diffraction_utils import Vector3
from diffraction_utils.diffractometers import I10RasorDiffractometer

from .image import Image
from .rsm_metadata import RSMMetadata


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

    def __init__(self, images: List[Image], metadata: RSMMetadata):
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

    def _init_rsm(self,
                  lattice_parameters: Union[List, np.ndarray, float] = None,
                  sample_orientation: Tuple[int] = (0, 1, 0)):
        """
        Initializes the scan's reciprocal space map.

        Args:
            lattice_parameters:
                The lattice parameters of the crystal we're mapping. Can be just
                a float in the case of a cubic crystal. In the case of an
                orthorhombic crystal, takes a numpy array/list of the three
                conventional lattice vectors [a, b, c].
            sample_orientation:
                What face is out of plane? Takes miller indices as a Tuple.
                Defaults to (0,1,0).
        """
        raise NotImplementedError()

    @classmethod
    def from_i10(cls,
                 path_to_nx: Union[str, Path],
                 beam_centre: Tuple[int],
                 detector_distance: float,
                 sample_oop: Vector3,
                 path_to_tiffs: str = ''):
        """
        Instantiates a Scan from the path to an I10 nexus file, a beam centre
        coordinate, a detector distance (this isn't stored in i10 nexus files)
        and a sample out-of-plane vector.

        Args:
            path_to_nx:
                Path to the nexus file containing the scan metadata.
            beam_centre:
                A (y, x) tuple of the beam centre, measured in the usual image
                coordinate system, in units of pixels.
            detector_distance:
                The distance between the sample and the detector, which cant
                be stored in i10 nexus files so needs to be given by the user.
            sample_oop:
                An instance of a diffraction_utils Vector3 which descrbes the
                sample out of plane vector.
            path_to_tiffs:
                Path to the directory in which the images are stored. Defaults
                to '', in which case a bunch of reasonable directories will be
                searched for the images.
        """
        # Load the nexus file.
        i10_nexus = I10Nexus(path_to_nx, detector_distance)

        # Load the state of the RASOR diffractometer; prepare the metadata.
        diff = I10RasorDiffractometer(i10_nexus, sample_oop, 'area')
        meta = RSMMetadata(diff, beam_centre)

        # Now load the images.
        images = [Image(data, meta, x) for x, data in
                  enumerate(i10_nexus.load_image_arrays(path_to_tiffs))]

        return cls(images, meta)
