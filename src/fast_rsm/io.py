"""
This module contains parsers for different instruments that return Scan objects.
"""

from pathlib import Path
from typing import Union, Tuple, TYPE_CHECKING


from .diffractometers import I07Diffractometer

from . import scan
from .rsm_metadata import RSMMetadata

if TYPE_CHECKING:
    from .scan import Scan


import json
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Union
from warnings import warn


import nexusformat.nexus.tree as nx
import numpy as np
import pandas as pd
from nexusformat.nexus import nxload

from .data_file import NexusBase
from .frame_of_reference import Frame
from .polarisation import Polarisation
from .region import Region
from .vector import Vector3




BAD_NEXUS_FILE = (
    "Nexus files suck. It turns out your nexus file sucked too. "
    "If you're seeing this message, it means some non-essential data couldn't "
    "be parsed by diffraction_utils.")







