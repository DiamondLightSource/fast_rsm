"""
This module contains a simple object that defines the polarisation state of
incident radiation.
"""


from .vector import Vector3


class Polarisation:
    """
    Instances of this class define the polarisation state of a beam.

    Attrs:
        kind:
            A string describing the kind of polarisation. Can be "linear",
            "unpolarised", "right-circular" or "left-circular". More can be
            implemented, just raise an issue on the github repo if you need
            something more specific to be implemented (e.g. elliptical etc.)
        vector:
            The vector describing the polarisation. This is None for unpolarised
            light.
    """

    linear = "linear"
    unpolarised = "unpolarised"

    def __init__(self, kind: str, vector: Vector3 = None) -> None:
        self.kind = kind
        self.vector = vector

        if (self.vector is None) and (self.kind != Polarisation.unpolarised):
            raise ValueError("Polarised beams must be described by a vector.")
