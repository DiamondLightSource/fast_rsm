#!/usr/bin/env python3
"""
A simple command line program that takes a single image and maps it to
reciprocal space.
"""

# pylint: disable=invalid-name

import argparse
import os
from datetime import datetime

import numpy as np

from diffraction_utils import I07Nexus, Frame
from fast_rsm.scan import Scan
import argparse
import os
from yaml import load, dump, Loader


if __name__ == "__main__":

    HELP_STR = (
        "Takes in configuration settings from YAML file and uses it\n"
        "to create job, then sends this job to the cluster."
    )
    parser = argparse.ArgumentParser(description=HELP_STR)

    HELP_STR = (
        "Path to the YAML file with configuration settings. "
    )
    parser.add_argument("-y", "--yaml_path", help=HELP_STR)
    HELP_STR = (
        "Scan numbers to be mapped into one reciprocal volume"
    )
    parser.add_argument("-s", "--scan_nums", help=HELP_STR, type=int,
                        default=os.environ.get("NUM_BINS"))


    # Extract the arguments from the parser.
    args = parser.parse_args()

    y_file = open(args.yaml_path, 'r', encoding='utf-8')
    recipe = load(y_file, Loader=Loader)
    y_file.close()

    EXP_N=recipe['visit']['experiment_number']
    YEAR=recipe['visit']['year']#
    SCANS=args.scan_nums

    f=open(r"C:\Users\rpy65944\Documents\testjobtemplate_nocom.py")
    lines=f.readlines()
    f.close()


