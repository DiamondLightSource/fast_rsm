import argparse
import os
import re
from argparse import RawTextHelpFormatter

import numpy as np
import pandas as pd
from IPython.display import clear_output
from nexusformat.nexus import *
from PIL import Image

from diffraction_utils import I07Nexus


def parse_scans(scan_range, scan_nums):
    if scan_range is None:
        scans = scan_nums
    else:
        rlist = eval(scan_range)
        if len(np.shape(rlist)) == 1:
            scanrange = rlist
            scans = list(
                range(int(scanrange[0]), int(scanrange[1]) + 1, int(scanrange[2]))
            )
        else:
            scans = []
            for r in rlist:
                scanrange = r
                scans.extend(
                    list(
                        range(
                            int(scanrange[0]),
                            int(scanrange[1]) + 1,
                            int(scanrange[2]),
                        )
                    )
                )
    return scans


def create_dat_file(outdir, filename, i07_nexus, loaded_nexus):
    datfile = rf"{outdir}/{filename}/{filename}.dat"
    with open(rf"{datfile}", "w") as f:
        f.write(f"##created .dat file from {filename}\n")
        for key in loaded_nexus.entry.instrument:
            clear_output(wait=True)
            print(f"\r{key}", end="")
            if "value" in loaded_nexus[f"entry/instrument/{key}"]:
                val = loaded_nexus[f"entry/instrument/{key}/value"]
                f.write(f"{key} = {val}\n")
        f.write(" &END\n")
    outdata = pd.DataFrame()

    outlist = [
        "testMotor1",
        "d5i/d5i",
        "att",
        "transmission",
        "count_time",
        "frameNo",
        "max_val",
        "total",
        "Region_1.max_val",
        "Region_1.total",
        "norm",
    ]

    for key in outlist:
        if key in i07_nexus.nx_instrument:
            outdata[key] = i07_nexus.nx_instrument[key].nxdata

    for key in i07_nexus.nx_entry:
        if "Region" in key:
            outdata[key] = i07_nexus.nx_entry[key]["_".join(key.split("_")[1:])].nxdata
    outdata["d5i"] = i07_nexus.nx_instrument["d5i/d5i"].nxdata

    outdata.to_csv(rf"{datfile}", mode="a", sep="\t", index=False)


def convertnexus(scan, loaddir, outdir):
    filename = f"i07-{scan}"
    nexusfile = rf"{loaddir}/{filename}.nxs"
    print(f"exporting data to {loaddir}/{filename}")
    loaded_nexus = nxload(rf"{nexusfile}")
    i07_nexus = I07Nexus(nexusfile, loaddir)
    detector_name = i07_nexus.detector_info.name
    detector_hdf5 = i07_nexus.nx_detector.data.nxfilename
    a1 = nxload(detector_hdf5)
    if os.path.exists(f"{outdir}/{filename}"):
        print(rf"{outdir}/{filename} already exists so skipping {scan}")
        return
    os.mkdir(rf"{outdir}/{filename}")

    if not os.path.exists(f"{loaddir}/{scan}.dat"):
        create_dat_file(outdir, filename, i07_nexus, loaded_nexus)
    count = 1
    data = a1.entry.data.data

    for n in np.arange(len(data)):
        imdata = data[n, :, :]
        im = Image.fromarray(np.array(imdata))  # float32
        savestring = "{:0>{}}".format(n, 4)
        im.save(rf"{outdir}/{filename}/{scan}_{savestring}.tif", "TIFF")
        count += 1


def convert_directory(loaddir, outdir):
    print(f"converting all .nxs scans found in {loaddir}")
    files = os.listdir(loaddir)
    pattern = r"^i07-.*\.nxs$"
    scanlist = [
        file.split("-")[1].split(".")[0] for file in files if re.search(pattern, file)
    ]
    scanlist.sort()
    convert_scan_list(scanlist, loaddir, outdir)


# filename = "i07-681185"
# dir = "/dls/i07/data/2026/si43482-1/Ye"


# outdir = "/scratch/rpy65944/Downloads/"
def convert_scan_list(scanlist, dir, outdir):
    for scan in scanlist:
        print(f"\n exporting data for scan {scan}")
        try:
            convertnexus(scan, dir, outdir)
        except NeXusError as e:
            print(f"unable to convert {scan}: error message {e}")


def convert_scans(args):

    dir, scan_range, scan_nums, outdir = (
        args.data_directory,
        args.scan_range,
        args.scan_nums,
        args.out_path,
    )
    if args.all:
        convert_directory(dir, outdir)
        return
    scanlist = parse_scans(scan_range, scan_nums)
    convert_scan_list(scanlist, dir, outdir)


if __name__ == "__main__":
    HELP_STR = (
        "Takes in file scan range and then extract data from nexus files into dat and tiff files \n"
        "example useage: \n\t python tiff_dat_exporter.py -dir /dls/i07/data/2026/si43482-1/Ye -sr [[681182,681185,1]] -o /scratch/rpy65944/Downloads/ "
    )
    parser = argparse.ArgumentParser(
        description=HELP_STR, formatter_class=RawTextHelpFormatter
    )

    HELP_STR = (
        "Path to the directory in which the data is stored. If this "
        + "is not specified, your current directory will be used."
    )
    parser.add_argument("-dir", "--data_directory", help=HELP_STR)

    HELP_STR = "Separate scan numbers to be converted to tiff and dat"
    parser.add_argument("-s", "--scan_nums", nargs="+", help=HELP_STR)

    HELP_STR = "Evenly spaced range of scans to be converted to tiff and dat"
    parser.add_argument("-sr", "--scan_range", help=HELP_STR, default=None)

    HELP_STR = "Path to the directory for saving output files to. "
    parser.add_argument("-o", "--out_path", help=HELP_STR, default=None)

    HELP_STR = (
        "Use this flag if you want to convert all .nxs files in the data directory"
    )
    parser.add_argument("-a", "--all", help=HELP_STR, action="store_true")

    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = "/".join(args.data_directory.split("/")[0:6] + ["processing"])
    convert_scans(args)
