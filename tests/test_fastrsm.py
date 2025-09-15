from time import time

import numpy as np
from numpy.testing import assert_allclose

from diffraction_utils import Frame
from fast_rsm.scan import Scan

import inspect
import pytest
from pathlib import Path
import subprocess
import sys
import pexpect


PROCESS_SCRIPT = '/dls_sw/apps/fast_rsm/testing/fast_rsm/CLI/i07/runconfig.py'
PATH_TO_EXAMPLE_DATA = '/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data'


def run_process(exp_setup_path, scans):
    command = f"python {PROCESS_SCRIPT} -exp {PATH_TO_EXAMPLE_DATA}/{exp_setup_path} -s {scans} -calc /dls/i07/data/2024/cm37245-1/PhilMousley_testing/fast_rsm/CLI/i07/example_calc_setup.py"
    print(f'got command  :   {command}', flush=True)

    child = pexpect.spawn(command, encoding='utf-8', timeout=1200)
    exit_status = 0
    while True:
        try:
            # Wait for the next line of output
            line = child.readline()
            if not line:
                # EOF reached
                break

            # Remove trailing newline and print the line
            line = line.rstrip()
            print(line)

            # Check for specific conditions
            if "Error" in line:
                print("Error detected!")
                child.sendline('d')
                exit_status = 1

            if "PROCESSING FINISHED" in line:
                print("Processing finished, sending exit command...")
                child.sendline('d')  # Send exit command

            if "Starting tail monitoring" in line:
                print("Tail monitoring detected, waiting for completion...")

        except pexpect.EOF:
            print("Process ended")
            break

    # Get the exit status
    child.close()
    return exit_status

    # process = subprocess.Popen(command,stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
    # while True:
    #     output = process.stdout.readline()
    #     if output:
    #         print(output.strip(),flush=True)
    #         sys.stdout.flush()  #
    #     if 'Error' in output:
    #         print('found error',flush=True)
    #         process.stdin.write('d\n')
    #         process.stdin.flush()
    #         break
    #     if 'PROCESSING FINISHED' in output:

    #         process.stdin.write('d\n')
    #         process.stdin.flush()
    #     if output == '' and process.poll() is not None:
    #         break

    # stdout, stderr = process.communicate()
    # returncode = process.returncode
    # return returncode, stdout, stderr
    # return result.returncode, result.stdout, result.stderr


@pytest.mark.parametrize("exp_setup_path,scans", [
    # pytest.param("si31699_exp_setup.py", "448617", id="test_GWstep_small_exc0_default"),
    # pytest.param("exp_setup_432196.py", "432196", id="test_GWsingle_large_det_default"),
    # pytest.param("si28599_exp_setup.py", "412802", id="test_GWstep_small_exc90_default"),
    # pytest.param("si36936_exp_setup.py", " ".join(map(str,np.arange(512422,512434,1))),id="test_inplaneomegaCTR"),
    # pytest.param("exp_setup_531473.py", "531473", id="test_GW4d_single_p2d_deafult"),
    # pytest.param("si34539_exp_setup.py", "492602", id="test_ctr_step_FOR_default"),
    pytest.param("si34983_exp_setup.py", "509441", id="test_GWdcdStep")

    # Add more combinations as needed
])
# 'si31699_exp_setup.py' 448617    test_GWstep_small_exc0_default
# 'exp_setup_432196.py'   432196    test_GWsingle_large_det_default
# 'si28599_exp_setup.py' 412802    test_GWstep_small_exc90_default
# 'exp_setup_531473.py'   531473    test_GW4d_single_p2d_deafult
# 'si34539_exp_setup.py'  492602    test_ctr_step_FOR_default
# si34983         509441       test_dcd_smallstep
# need to download data
# 'si36936       513017 - 513070	     test_inplane_map #extremeely large scans, only look at 513017-018 for testing
def test_process_script(exp_setup_path, scans):
    print(
        f"Running test with setup file: {exp_setup_path} and scans: {scans}", flush=True)
    returncode = run_process(exp_setup_path, scans)

    print(
        f"finished testing  setup file: {exp_setup_path} and scans: {scans}", flush=True)
    # print(f"Process finished with return code {returncode}")
    # Assert that the process ran successfully
    assert returncode == 0, f"Process failed with setup file {exp_setup_path} and scans {scans}."
# CTR measurements     stepped   dls/i07/data/2023/si34539-1/ana001/	492632 - 639  - omega + fast scan  10L
# CTR measurements     stepped   dls/i07/data/2023/si34539-1/ana001/	492655 - 492662  omega+ fast scan   13L

# CTR measurements     continuous si36936-1	all rocking curves  513349-513389   10L
# CTR measurements     continuous si36936-1	all rocking curves 513391-513427  22L

# stepped - large area detector (p2m)  '/dls/staging/dls/i07/data/2022/cm31120-1/20220226_Soller_test'	421593
