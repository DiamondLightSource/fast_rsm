from time import time

import numpy as np
from numpy.testing import assert_allclose

from diffraction_utils import Frame
from fast_rsm.scan import Scan

import inspect
import pytest
from pathlib import Path


def eval_py_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        exec(content, globals())



def test_do_calculations(path_to_frsm_example_data:str):
    tests_dir = Path(__file__).parent
    calcfile=tests_dir/'CLI/i07/example_calc_setup.py'
    print(calcfile)
    eval_py_file(path_to_frsm_example_data+"/si28599_exp_setup.py")
    scan_numbers=[531484]
    eval_py_file(calcfile)