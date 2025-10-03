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
import re

import sys



version_path=__file__.split('fast_rsm/CLI')[0]
python_version=version_path+'/conda_env/bin/python'

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
    parser.add_argument("-calc", "--calc_path", help=HELP_STR,default=f'{version_path}/fast_rsm/CLI/i07/example_calc_setup.py')
    
    HELP_STR = (
        "Separate scan numbers to be mapped into one reciprocal volume without brackets e.g 441124 441128"
    )
    parser.add_argument("-s", "--scan_nums",nargs="+", help=HELP_STR)

    HELP_STR = (
        "Evenly spaced range of scans to be mapped into one reciprocal volume of the format [start,stop,step]"
    )
    parser.add_argument("-sr", "--scan_range", help=HELP_STR,default=0)

    HELP_STR = (
        "Path to the directory for saving output files to. "
    )
    parser.add_argument("-o", "--out_path", help=HELP_STR,default=None)


    # Extract the arguments from the parser.
    args = parser.parse_args()
    with open(args.exp_path) as f1:
        lines1=f1.readlines()

    with open(args.calc_path) as f2:
        lines2=f2.readlines()

    outline=[line for line in lines1 if 'local_output' in line]
    if len(outline)==0:
        OUTDIR=args.out_path
    else:
        exec(outline[0])
        OUTDIR=local_output_path


    if args.scan_range==0:
        SCANS=args.scan_nums
    else:
        rlist=eval(args.scan_range)
        if len(np.shape(rlist))==1:
            scanrange=rlist
            SCANS=list(range(int(scanrange[0]),int(scanrange[1])+1,int(scanrange[2])))
        else:
            SCANS=[]
            for r in rlist:
                scanrange=r
                SCANS.extend(list(range(int(scanrange[0]),int(scanrange[1])+1,int(scanrange[2]))))


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
    f=open(f'{version_path}fast_rsm/CLI/i07/mapscript_template.sh')
    lines=f.readlines()
    f.close()

    #update mapscript in the /home/fast_rsm  directory using template, and filling in variables
    script_path=f'{Path.home()}/mapscript.sh'
    print(script_path)
    f=open(script_path,'w')
    for line in lines:
        phrase_matches=list(re.finditer(r'\${[^}]+\}',line))
        phrase_positions=[(match.start(),match.end()) for match in phrase_matches]
        outline=line
        for pos in phrase_positions:
            phrase=line[pos[0]:pos[1]]
            outphrase=phrase.strip('$').strip('{').strip('}')
            outline=outline.replace(phrase,str(locals()[f'{outphrase}']))
        f.write(outline)
    f.close()
    #get list of slurm out files in home directory
    startfiles=os.listdir(f'{Path.home()}/fast_rsm')
    startslurms=[x for x in startfiles if '.out' in x]
    startslurms.append(startfiles[0])
    startslurms.sort(key=lambda x: os.path.getmtime(f'{Path.home()}/fast_rsm/{x}'))

    #get latest slurm file  before submitting job
    endfiles=os.listdir(f'{Path.home()}/fast_rsm')
    endslurms=[x for x in endfiles if '.out' in x]
    endslurms.append(endfiles[0])
    endslurms.sort(key=lambda x: os.path.getmtime(f'{Path.home()}/fast_rsm/{x}'))
    count=0
    limit=0
    #call subprocess to submit job using wilson
    subprocess.run(["ssh","wilson",f"cd fast_rsm\nsbatch {script_path}"])

    #have check loop to find a new slurm out file
    while endslurms[-1]==startslurms[-1]:
        endfiles=os.listdir(f'{Path.home()}/fast_rsm')
        endslurms=[x for x in endfiles if '.out' in x]
        endslurms.append(endfiles[0])
        endslurms.sort(key=lambda x: os.path.getmtime(f'{Path.home()}/fast_rsm/{x}'))
        if count >50:
            limit=1
            break
        print(f'Job submitted, waiting for SLURM output.  Timer={5*count}',end="\r")
        time.sleep(5)
        count+=1
    if limit==1:
        print('Timer limit reached before new slurm ouput file found')
    else:
        print(f'Slurm output file: {Path.home()}/fast_rsm//{endslurms[-1]}\n')
        breakerline='*'*35
        monitoring_line=f"\n{breakerline}\n ***STARTING TO MONITOR TAIL END OF FILE, TO EXIT THIS VIEW PRESS ANY LETTER FOLLOWED BY ENTER**** \n{breakerline} \n"
        print(monitoring_line)
        process = subprocess.Popen(["tail","-f",f"{Path.home()}/fast_rsm//{endslurms[-1]}"], stdout=subprocess.PIPE, text=True)
        target_phrase="PROCESSING FINISHED"
        err_msgs=["Errno","error", "Error" ]
        sparse_msg=["Sparse matrix"]
        try:
            for line in process.stdout:
                print(line.strip())  # Print each line of output
                if re.search(target_phrase, line):
                    print(f"Target phrase '{target_phrase}' found. Closing tail.")
                    break
                if any(s in line for s in err_msgs) and ('ForkPoolWorker' not in line) and not any(s in line for s in sparse_msg):
                    print("error found. closing tail")
                    break
        finally:
            process.terminate()
            process.wait()



