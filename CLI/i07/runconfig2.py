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
from pathlib import Path
import subprocess
import time
if __name__ == "__main__":

    HELP_STR = (
        "Takes in configuration settings from YAML file and uses it\n"
        "to create job, then sends this job to the cluster."
    )
    parser = argparse.ArgumentParser(description=HELP_STR)

    HELP_STR = (
        "Path to experiment setup options. "
    )
    parser.add_argument("-exp", "--exp_path", help=HELP_STR)

    HELP_STR = (
        "Path to the calcuation options. "
    )
    parser.add_argument("-calc", "--calc_path", help=HELP_STR)
    
    HELP_STR = (
        "Scan numbers to be mapped into one reciprocal volume"
    )
    parser.add_argument("-s", "--scan_nums",nargs="+", help=HELP_STR)

    HELP_STR = (
        "Path to the directory for saving output files to. "
    )
    parser.add_argument("-o", "--out_path", help=HELP_STR)

    # Extract the arguments from the parser.
    args = parser.parse_args()
    f=open(args.exp_path)
    lines1=f.readlines()
    f.close()
    f=open(args.calc_path)
    lines2=f.readlines()
    f.close()




    OUTDIR=args.out_path
    SCANS=args.scan_nums

    i=1

    save_file_name = f"job_scan_{SCANS[0]}_{i}.py"
    save_path = Path(OUTDIR)/Path(save_file_name)
    # Make sure that this name hasn't been used in the past.
    while (os.path.exists(str(save_path))):
        i += 1
        save_file_name = f"job_scan_{SCANS[0]}_{i}.py"
        save_path = Path(OUTDIR)/Path(save_file_name)
        if i > 1e7:
            raise ValueError(
                "naming counter hit limit therefore exiting ")   
    
    #save variables to job file using job template
    f=open(save_path,'x')
    f.write(''.join(lines1))
    f.write(f'scan_numbers= {SCANS}\n')
    f.write(''.join(lines2))
    f.close()
    
    
    #load in template mapscript, new paths
    f=open(f'{Path.home()}/fast_rsm/mapscript_template.sh')
    lines=f.readlines()
    f.close()

    #update mapscript in the /home/fast_rsm  directory using template, and filling in variables
    f=open(f'{Path.home()}/fast_rsm//mapscript.sh','w')
    for line in lines:
        if '$' in line:
            phrase=line[line.find('$'):line.find('}')+1]
            outphrase=phrase.strip('$').strip('{').strip('}')
            outline=line.replace(phrase,str(locals()[f'{outphrase}']))
            #print(outline)
            f.write(outline)
        else:
            f.write(line)
    f.close()
    #get list of slurm out files in home directory
    
    startfiles=os.listdir(f'{Path.home()}/fast_rsm')
    startslurms=[x for x in startfiles if '.out' in x]
    #get latest slurm file  before submitting job
    endfiles=os.listdir(f'{Path.home()}/fast_rsm')
    endslurms=[x for x in endfiles if '.out' in x]
    count=0
    limit=0
    #call subprocess to submit job using wilson
    subprocess.run(["ssh","wilson","cd fast_rsm \nsbatch mapscript.sh"])

    #have check loop to find a new slurm out file
    while endslurms[-1]==startslurms[-1]:
        endfiles=os.listdir(f'{Path.home()}/fast_rsm')
        endslurms=[x for x in endfiles if '.out' in x]
        if count >50:
            limit=1
            break
        print(f'Job submitted, waiting for SLURM output.  Timer={5*count}',end="\r")
        time.sleep(5)
        count+=1
    if limit==1:
        print('Timer limit reached before new slurm ouput file found')
    else:
        print(f'Job finished\nSlurm output file: {Path.home()}/fast_rsm/{endslurms[-1]}')
        print(f'Checking slurm output')
        time.sleep(15)
        f=open(f'{Path.home()}/fast_rsm/{endslurms[-1]}')
        lines=f.readlines()
        f.close()
        if 'PROCESSING FINISHED.\n' in lines:
            print('Processing completed successfully')
        else:
            print("error encountered during processing, view slurm file for details")
            #subprocess.run([f"less {Path.home()}/fast_rsm/{endslurms[-1]}"])
            #os.system(f"less {Path.home()}/fast_rsm/{endslurms[-1]}")



