"""
This module contains the class that is used to store images.
"""

import numpy as np


class Image:
    """
    The class used to store raw image data. Internally, this data is stored as
    a numpy array.

    Attrs:
        data:
            A numpy array storing the image data.
    """

    def __init__(self, data: np.ndarray):
        self.data = data
