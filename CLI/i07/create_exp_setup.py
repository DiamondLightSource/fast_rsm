#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:37:13 2024

@author: rpy65944
"""
import os
import subprocess
from pathlib import Path

def create_exp_setup():
    """
    copies over experimental setup file template to users home/fast_rsm directory
    """
    current_path=Path(__file__)
    current_dir=current_path.parent
    example_file=current_dir / 'example_exp_setup.py'
    homepath=Path.home()
    if not os.path.exists(homepath/'fast_rsm'):
        os.mkdir(homepath/'fast_rsm')

    with open(homepath/'fast_rsm/test_exp_setup.py','w') as f1, open(example_file) as f2:
        outlines=f2.readlines()
        f1.write(f'# copied from {str(example_file)}\n')
        for line in outlines:
            f1.write(line)
    print(subprocess.run(['gedit', f'{homepath}/fast_rsm/test_exp_setup.py']))


if __name__ == "__main__":
    create_exp_setup()

