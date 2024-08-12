#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:33:28 2024

@author: rpy65944

"""
import subprocess
import argparse
import h5py

def splitandopen(f,out):
    dataset=f['config/joblines']
    exp_file_path=f'{out}/exp_setup_output.txt'
    calc_file_path=f'{out}/calc_setup_output.txt'
    with open(exp_file_path, 'w') as text_file_exp,\
        open(calc_file_path, 'w') as text_file_calc:
        # Iterate over the dataset
        active_textfile=text_file_exp
        for byte_string in dataset:
            # Decode the byte string to a regular string
            decoded_string = byte_string.decode('utf-8').replace('\n','')
            if 'scan_numbers' in decoded_string:
                active_textfile.write('#'+decoded_string+'\n')
                active_textfile=text_file_calc
            else:
                # Write the decoded string to the text file
                active_textfile.write(decoded_string + '\n')
    subprocess.run(['gio', 'open', exp_file_path])
    subprocess.run(['gio', 'open', calc_file_path])



if __name__ == "__main__":
    
    HELP_STR = (
    "Takes in path to hdf5 output file, and shows exp_setup and calc_setup files used for processing. Opens up both setting in separate text files for viewing "
)

    parser = argparse.ArgumentParser(description=HELP_STR)
    HELP_STR = (
        "Path to hdf5 file "
    )
    parser.add_argument("-hf", "--hf_path", help=HELP_STR)
    
    HELP_STR = (
        "Path to output directory to save example exp_setup and calc_setup files, defaults to home/fast_rsm"
    )
    parser.add_argument("-outdir", "--out_dir", help=HELP_STR)
    args = parser.parse_args()

    hf_path=args.hf_path
    outdir=args.out_dir
    spacer='='*25

    f = h5py.File(hf_path,'r')
    if 'config/joblines' in f.keys():
        splitandopen(f,outdir)
        f.close()
    else:
        print(f'{spacer}\nUnable to open up setup files, no joblines found in hdf5 file\n{spacer}')