#!/usr/bin/env python3
"""
A simple command line program that takes a single image and maps it to
reciprocal space.
"""

# pylint: disable=invalid-name

import argparse

from fast_rsm.diamond_utils import ProcessArgs

version_path = __file__.split("fast_rsm/CLI")[0]
python_version = version_path + "/conda_env/bin/python"

if __name__ == "__main__":
    HELP_STR = (
        "Takes in configuration settings from YAML file and uses it\n"
        "to create job, then sends this job to the cluster."
    )
    parser = argparse.ArgumentParser(description=HELP_STR)

    HELP_STR = "Path to experiment setup options. "
    parser.add_argument("-exp", "--exp_path", help=HELP_STR)

    HELP_STR = "Path to the calcuation options. "
    parser.add_argument(
        "-calc",
        "--calc_path",
        help=HELP_STR,
        default=f"{version_path}/fast_rsm/CLI/i07/example_calc_setup.py",
    )

    HELP_STR = "Separate scan numbers to be mapped into one reciprocal volume without brackets e.g 441124 441128"
    parser.add_argument("-s", "--scan_nums", nargs="+", help=HELP_STR)

    HELP_STR = "Evenly spaced range of scans to be mapped into one reciprocal volume of the format [start,stop,step]"
    parser.add_argument("-sr", "--scan_range", help=HELP_STR, default=0)

    HELP_STR = "Path to the directory for saving output files to. "
    parser.add_argument("-o", "--out_path", help=HELP_STR, default=None)

    HELP_STR = "Use this flag to activate the debug logger"
    parser.add_argument("-dblog", "--debuglogging", help=HELP_STR, default=0)

    HELP_STR = "Set this flag to true to run job locally"
    parser.add_argument("-local", "--local", help=HELP_STR, action="store_true")

    HELP_STR = "Set this flag to true to run job using development python version"
    parser.add_argument("-dev", "--dev", help=HELP_STR, action="store_true")
    # Extract the arguments from the parser.
    args = parser.parse_args()
    args.version_path = version_path
    args.python_version = python_version
    if args.dev:
        args.python_version = "/dls/science/users/rpy65944/python_envs/runpytest/bin/python"  # python_version

    process_args = ProcessArgs(**vars(args))
    process_args.parse_and_reduce()
