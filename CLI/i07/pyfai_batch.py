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
from yaml import load, dump, Loader
from pathlib import Path
import subprocess
import time
import pyFAI, fabio
if __name__ == "__main__":

    HELP_STR = (
        "Takes in configuration settings from YAML file and uses it\n"
        "to create job, then sends this job to the cluster."
    )
    parser = argparse.ArgumentParser(description=HELP_STR)

    HELP_STR = (
        "Path to folder containing experiment images. "
    )
    parser.add_argument("-images", "--imagespath", help=HELP_STR)

    HELP_STR = (
        "Path to mask file. "
    )
    parser.add_argument("-mask", "--maskpath", help=HELP_STR)

    HELP_STR = (
        "Path to PONI calibration file. "
    )
    parser.add_argument("-poni", "--ponipath", help=HELP_STR)
    
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
    parser.add_argument("-o", "--out_path", help=HELP_STR)

    HELP_STR = (
        "number of bins to use for 1D theta integration"
    )
    parser.add_argument("-bins", "--bins", help=HELP_STR,default=1000)

    # Extract the arguments from the parser.
    args = parser.parse_args()

    OUTDIR=args.out_path
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


#print("pyFAI version:", pyFAI.version)

alllist=os.listdir(args.imagespath)
tiflist=[file for file in alllist if file.endswith('.tif')]
    #GIVE PATH TO PONI CALIBRATION FILE
ai = pyFAI.load(args.ponipath)
print("\nIntegrator: \n", ai)
for scan in SCANS:
    fnames=[file for file in tiflist if str(scan) in file]
    fname=fnames[0]
    img = fabio.open(fr'{args.imagespath}/{fname}')
    print("Image:", img)
    

    img_array = img.data
    print("img_array:", type(img_array), img_array.shape, img_array.dtype)
    
    #GIVE PATH TO MASK FILE
    maskimg = fabio.open(args.maskpath)
    mask = maskimg.data
    
    datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    save_as = str(scan) +'_'+ datetime_str + ".dat"
    savepath=f'{OUTDIR}/{save_as}'
    
    
    tth,I = ai.integrate1d_ng(img_array,
                            args.bins,
                            mask=mask,
                            unit="2th_deg",polarization_factor=1)
    Q,I = ai.integrate1d_ng(img_array,
                        args.bins,
                        mask=mask,
                       unit="q_A^-1",polarization_factor=1)
    outdata=pd.DataFrame({'2theta':tth,'Q_angstrom^-1':Q,'Intensity':I})
    print(outdata.head())
    f=open(savepath,'w')
    f.write("#2theta\tQ_angstrom^-1\tIntensity\n")
    f.close()

    outdata.to_csv(savepath,sep='\t',index=False,header=False,mode='a')
    print(f'finished processing scan {scan}')
    print(f'saved 1D profile to {savepath}\n')
    print(f'process scan {scan}\n')
print('Finished processing images')