import os
import nexusformat.nexus as nx
from  diffraction_utils import I07Nexus
import subprocess
import argparse


def get_im_path(directorypath,scan_number):
    files=[file for file in os.listdir(f'{directorypath}') if '.nxs' in file]
    found_file=[file for file in files if str(scan_number)+'.nxs' in file][0]
    filepath=f'{directorypath}/{found_file}'
    found_nexus=I07Nexus(filepath,directorypath)#
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
        "Path to experiment directory where the nexus file is located "
    )
    parser.add_argument("-s", "--scan_number", help=HELP_STR)
    args = parser.parse_args()
    
    directorypath=args.dir_path
    scan=args.scan_number

    impath=get_im_path(directorypath,scan)
    subprocess.run(['pyFAI-drawmask', f'{impath}'])