"""
This module contains a convenience class for tracking motor positions.
"""

from dataclasses import dataclass


@dataclass
class Motors:
    """
    Keeps track of motor positions.
    """

    def __init__(self, theta: float, phi: float) -> None:
        self.theta = theta
        self.phi = phi
