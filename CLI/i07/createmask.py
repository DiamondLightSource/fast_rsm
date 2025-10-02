#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:37:13 2024

@author: rpy65944
"""
import os
import nexusformat.nexus as nx
from diffraction_utils import I07Nexus
import subprocess
import argparse
import h5py
from PIL import Image


def get_im_path(directorypath, scan_number,image_number):
    files = [file for file in os.listdir(f'{directorypath}') if '.nxs' in file]
    found_file = [file for file in files if str(
        scan_number) + '.nxs' in file][0]
    filepath = f'{directorypath}/{found_file}'
    found_nexus = I07Nexus(filepath, directorypath)
    if found_nexus.has_hdf5_data == True:
        hf = h5py.File(found_nexus.local_hdf5_path)
        imdata = hf[found_nexus.hdf5_internal_path][image_number]
        imout = Image.fromarray(imdata, mode='I')
        fname = found_nexus.local_hdf5_path.split('/')[-1].strip('.h5')
        home_dir = os.path.expanduser('~')
        outname = fr'maskimage_{fname}_0.tiff'
        outpath = os.path.join(home_dir, outname)
        # print(f'outpath before saving ={outpath}')
        imout.save(outpath, "TIFF")

        return outpath
    return found_nexus.local_image_paths[0]


if __name__ == "__main__":

    HELP_STR = (
        "Takes in directory path and scan number, to open up pyfai masking GUI"
    )

    parser = argparse.ArgumentParser(description=HELP_STR)
    HELP_STR = (
        "Path to experiment directory where the nexus file is located "
    )
    parser.add_argument("-dir", "--dir_path", help=HELP_STR)

    HELP_STR = (
        "Scan which you want to create a mask for "
    )
    parser.add_argument("-s", "--scan_number", help=HELP_STR)

    HELP_STR = (
        "Image from chosen scan to create a mask for, defaults to 0 "
    )
    parser.add_argument("-im", "--image_number", help=HELP_STR,default=0)
    args = parser.parse_args()

    impath = get_im_path(args.dir_path, args.scan_number,args.image_number)
    subprocess.run(['pyFAI-drawmask', f'{impath}'])
