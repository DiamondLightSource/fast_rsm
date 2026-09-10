import argparse
import os
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


def convertnexus(scan, loaddir, savedir):
    filename = f"i07-{scan}"
    nexusfile = rf"{loaddir}/{filename}.nxs"
    print(f"exporting data to {loaddir}/{filename}")
    loaded_nexus = nxload(rf"{nexusfile}")
    found_nexus = I07Nexus(nexusfile, loaddir)
    detector_name = found_nexus.detector_info.name
    detector_hdf5 = found_nexus.nx_detector.data.nxfilename
    a1 = nxload(detector_hdf5)
    if not os.path.exists(f"{savedir}/{filename}"):
        os.mkdir(rf"{savedir}/{filename}")

    datfile = rf"{savedir}/{filename}/{filename}.dat"
    f = open(rf"{datfile}", "w")
    f.write(f"##created .dat file from {filename}\n")
    for key in loaded_nexus.entry.instrument.keys():
        clear_output(wait=True)
        print(f"\r{key}", end="")
        if "value" in loaded_nexus[f"entry/instrument/{key}"]:
            val = loaded_nexus[f"entry/instrument/{key}/value"]
            f.write(f"{key} = {val}\n")
    f.write(" &END\n")
    f.close()

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
        if key in found_nexus.nx_instrument.keys():
            outdata[key] = found_nexus.nx_instrument[key].value

    for key in found_nexus.nx_entry.keys():
        if "Region" in key:
            outdata[key] = found_nexus.nx_entry[key][
                "_".join(key.split("_")[1:])
            ].nxdata
    outdata["d5i"] = found_nexus.nx_instrument["d5i/d5i"].nxdata

    outdata.to_csv(rf"{datfile}", mode="a", sep="\t", index=False)

    count = 1
    data = a1.entry.data.data

    for n in np.arange(len(data)):
        imdata = data[n, :, :]
        im = Image.fromarray(np.array(imdata))  # float32
        savestring = "{:0>{}}".format(n, 4)
        im.save(rf"{savedir}/{filename}/{scan}_{savestring}.tif", "TIFF")
        count += 1


# filename = "i07-681185"
# dir = "/dls/i07/data/2026/si43482-1/Ye"

# outdir = "/scratch/rpy65944/Downloads/"


def convert_scan_list(
    dir,
    scan_range,
    scan_nums,
    outdir,
):
    scanlist = parse_scans(scan_range, scan_nums)
    for scan in scanlist:
        print(f"\n exporting data for scan {scan}")
        try:
            convertnexus(scan, dir, outdir)
        except NeXusError as e:
            print(f"unable to convert {scan}: error message {e}")


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

    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = "/".join(args.data_directory.split("/")[0:6] + ["processing"])
    convert_scan_list(
        args.data_directory, args.scan_range, args.scan_nums, args.out_path
    )
