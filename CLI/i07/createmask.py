#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:37:13 2024

@author: rpy65944
"""

import argparse
import os
import subprocess

from fast_rsm.diamond_utils import get_im_path

if __name__ == "__main__":
    HELP_STR = "Takes in directory path and scan number, to open up pyfai masking GUI"

    parser = argparse.ArgumentParser(description=HELP_STR)
    HELP_STR = "Path to experiment directory where the nexus file is located "
    parser.add_argument("-dir", "--dir_path", help=HELP_STR)

    HELP_STR = "Scan which you want to create a mask for "
    parser.add_argument("-s", "--scan_number", help=HELP_STR)

    HELP_STR = "Image from chosen scan to create a mask for, defaults to 0 "
    parser.add_argument("-im", "--image_number", help=HELP_STR, default=0)
    args = parser.parse_args()

    impath = get_im_path(args.dir_path, args.scan_number, args.image_number)
    subprocess.run(["pyFAI-drawmask", f"{impath}"])
