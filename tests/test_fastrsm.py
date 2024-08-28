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


def run_job_lines(expfile,tests_dir,scanline):
    calcfile=tests_dir/'CLI/i07/example_calc_setup.py'
    #print(calcfile)
    eval_py_file(expfile)
    exec(f"{scanline}",globals())
    eval_py_file(calcfile)
    
    
# def test_GWsingle_large_det_default(path_to_frsm_example_data:str):
#     expfile=f'{path_to_frsm_example_data}/exp_setup_432196.py'
#     scanline="scan_numbers=[432196]"
#     tests_dir=Path(__file__).parent.parent
#     run_job_lines(expfile,tests_dir,scanline)


# #passes but need to double check output
# def test_GWstep_small_exc90_default(path_to_frsm_example_data:str):
#     expfile=f'{path_to_frsm_example_data}/si28599_exp_setup.py'
#     scanline="scan_numbers=[412802]"
#     tests_dir=Path(__file__).parent.parent
#     run_job_lines(expfile,tests_dir,scanline)


# def test_GWstep_small_exc0_default(path_to_frsm_example_data:str):
#     expfile=f'{path_to_frsm_example_data}/si31699_exp_setup.py'
#     scanline="scan_numbers=[448617]"
#     tests_dir=Path(__file__).parent.parent
#     run_job_lines(expfile,tests_dir,scanline)


# def test_GW4d_single_p2d_deafult(path_to_frsm_example_data:str):
#     expfile=f'{path_to_frsm_example_data}/exp_setup_531473.py'
#     scanline="scan_numbers=[531473]"
#     tests_dir=Path(__file__).parent.parent
#     run_job_lines(expfile,tests_dir,scanline)   


#CTR measurements     stepped   dls/i07/data/2023/si34539-1/ana001/	492602 -  FOR scan
def test_ctr_step_FOR_default(path_to_frsm_example_data:str):
    expfile=f'{path_to_frsm_example_data}/si34539_exp_setup.py'
    scanline="scan_numbers=[492602]"
    tests_dir=Path(__file__).parent.parent
    run_job_lines(expfile,tests_dir,scanline)  

#CTR measurements     stepped   dls/i07/data/2023/si34539-1/ana001/	492632 - 639  - omega + fast scan  10L 
#CTR measurements     stepped   dls/i07/data/2023/si34539-1/ana001/	492655 - 492662  omega+ fast scan   13L

#CTR measurements     continuous si36936-1	all rocking curves  513349-513389   10L
#CTR measurements     continuous si36936-1	all rocking curves 513391-513427  22L

#stepped - large area detector (p2m)  '/dls/staging/dls/i07/data/2022/cm31120-1/20220226_Soller_test'	421593