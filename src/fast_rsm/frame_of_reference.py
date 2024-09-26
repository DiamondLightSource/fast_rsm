"""
This module contains the Frame class, which is used to describe the frame of
reference in which vectors live.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .diffractometer_base import DiffractometerBase


class Frame:
    """
    Instances of this class contain enough information to uniquely identify a
    frame of reference. This isn't quite as simple as saying something like
    "sample frame", because the sample frame is generally a function of time
    during a scan. Instead, the frame of reference is completely described by
    a combination of an index and a string identifier.

    It is worth noting that in several special cases, scan_index does not need
    to be provided. For example, consider a crystal glued to a sample holder.
    In the Frame.sample_holder frame, descriptions of the crystal are
    independent of scan index. In that case, scan indices are only needed to
    transform to or from this frame: they are not needed to describe vectors in
    this frame.

    On the flip side, a vector describing a property of the crystal in the lab
    frame will need a scan_index and a diffractometer, since the diffractometer
    is generally moving during a scan.
    """

    sample_holder = 'sample holder'
    hkl = 'hkl'
    lab = 'lab'
    qpar_qperp = 'qpar_qperp'
    cartesian = 'cartesian'
    polar = 'polar'

    def __init__(self,
                 frame_name: str,
                 diffractometer: 'DiffractometerBase' = None,
                 scan_index: int = None,
                 coordinates: str = cartesian):
        self.frame_name = frame_name
        self.diffractometer = diffractometer
        self.scan_index = scan_index
        self.coordinates = coordinates

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, type(self)):
            return False

        frame_name_eq = self.frame_name == __o.frame_name
        scan_idx_eq = self.scan_index == __o.scan_index
        diffractometer_eq = (self.diffractometer.data_file.local_path ==
                             __o.diffractometer.data_file.local_path)

        return frame_name_eq and scan_idx_eq and diffractometer_eq
