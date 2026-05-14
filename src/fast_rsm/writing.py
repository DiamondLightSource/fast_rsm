"""
This module contains  convenience functions for writing reciprocal space maps.
"""

import os
from dataclasses import dataclass
from time import time

import numpy as np

# from multiprocessing.shared_memory import SharedMemory
import pandas as pd
import tifffile
import yaml
from pyevtk.hl import gridToVTK

# ===================================
# ====data saving functions


@dataclass
class result2d:
    data: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    x_unit: str | None = None
    y_unit: str | None = None


@dataclass
class result1d:
    data: np.ndarray
    data_name: str
    x_axis: np.ndarray
    x_axis_name: str
    x2_axis: np.ndarray | None = None
    x2_axis_name: str | None = None


def write_im_to_tiff(parainfo, perpinfo, hfname, out_filename, imdata):
    metadata = {
        "Description": f"Image data identical to data saved in {hfname}",
        "Xlimits": f"min {parainfo.min()}, max {parainfo.max()}",
        "Ylimits": f"min {perpinfo.min()}, max {perpinfo.max()}",
    }
    tifffile.imwrite(out_filename, imdata, metadata=metadata)


def do_savetiffs(hf, data, axespara, axesperp):
    """
    save separate tiffs for all 2d image data in data
    """
    datashape = np.shape(data)
    extradims = len(datashape) - 2
    outdir = hf.filename.strip(".hdf5")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outname = outdir.split("/")[-1]
    if extradims == 0:
        imdata = data
        parainfo = axespara
        perpinfo = axesperp
        out_filename = f"{outdir}/{outname}.tiff"
        write_im_to_tiff(parainfo, perpinfo, hf.filename, out_filename, imdata)

    if extradims == 1:
        for i1 in np.arange(datashape[0]):
            imdata = data[i1]
            parainfo = axespara[i1]
            perpinfo = axesperp[i1]
            out_filename = f"{outdir}/{outname}_{i1}.tiff"
            write_im_to_tiff(parainfo, perpinfo, hf.filename, out_filename, imdata)

    if extradims == 2:
        for i1 in np.arange(datashape[0]):
            for i2 in np.arange(datashape[1]):
                imdata = data[i1][i2]
                parainfo = axespara[i1][i2]
                perpinfo = axesperp[i1][i2]
                out_filename = f"{outdir}/{outname}_{i1}_{i2}.tiff"
                write_im_to_tiff(parainfo, perpinfo, hf.filename, out_filename, imdata)


def write_qi_to_csv(qvals, intvals, tthetavals, out_filename, metadata):
    outdf = pd.DataFrame(
        {"Q_angstrom^-1": qvals, "Intensity": intvals, "two_theta": tthetavals}
    )
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(metadata)
        outdf.to_csv(f, sep="\t", index=False)


def do_savedats(hf, intdata, qdata, tthdata):
    """
    save all 1d datasets to .dat files
    """
    datashape = np.shape(intdata)
    extradims = len(datashape) - 1
    outdir = hf.filename.strip(".hdf5")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    metadata = f"Intensity data identical to data saved in {hf.filename}\n"
    outname = outdir.split("/")[-1]
    if extradims == 0:
        intvals = intdata
        qvals = qdata
        tthetavals = tthdata
        out_filename = f"{outdir}/{outname}.dat"
        write_qi_to_csv(qvals, intvals, tthetavals, out_filename, metadata)

    if extradims == 1:
        for i1 in np.arange(datashape[0]):
            intvals = intdata[i1]
            qvals = qdata[i1]
            tthetavals = tthdata[i1]
            out_filename = f"{outdir}/{outname}_{i1}.dat"
            write_qi_to_csv(qvals, intvals, tthetavals, out_filename, metadata)

    if extradims == 2:
        for i1 in np.arange(datashape[0]):
            for i2 in np.arange(datashape[1]):
                intvals = intdata[i1][i2]
                qvals = qdata[i1][i2]
                tthetavals = tthdata[i1][i2]
                out_filename = f"{outdir}/{outname}_{i1}_{i2}.dat"
                write_qi_to_csv(qvals, intvals, tthetavals, out_filename, metadata)


def save_1d_integration_static(cfg, hf, outresult: result1d, scan=None):
    """
    save 1d Intensity Vs Q profile to hdf5 file
    """

    dset = hf.create_group("integrations")
    # for k, v in outlist.items():
    #     dset.create_dataset(k, data=v)
    dset.create_dataset(outresult.data_name, data=outresult.data)
    dset.create_dataset(outresult.x_axis_name, data=outresult.x_axis)
    if outresult.x2_axis is not None:
        dset.create_dataset(outresult.x2_axis_name, data=outresult.x2_axis)
    # dset.create_dataset("Intensity", data=outlist[0])
    # dset.create_dataset(f"{outlist[3][0]}", data=outlist[1])
    # dset.create_dataset(f"{outlist[3][1]}", data=outlist[2])

    if (scan is not None) & ("scanfields" not in hf.keys()):
        save_scan_field_values(hf, scan)
    if cfg.savedats is True:
        do_savedats(hf, outresult.data, outresult.x2_axis, outresult.x_axis)
    save_config_variables(hf, cfg)
    hf.close()


def save_1d_integration(
    hf, cfg, ints_final, counts_final, tth_vals_final, q_final, mapaxisinfo
):

    int_array = np.divide(
        ints_final,
        counts_final,
        out=np.copy(ints_final),
        where=counts_final.astype(float) > 0.0,
    )
    # outlist = [int_array, q_final, tth_vals_final, tth_string]

    outlist = {
        "Intensity": int_array,
        f"{mapaxisinfo[0][1]}": mapaxisinfo[0][0],
        "Q_angstrom^-1": q_final,
    }
    outresult = result1d(
        data=int_array,
        data_name="Intensity",
        x_axis=mapaxisinfo[0][0],
        x_axis_name=f"{mapaxisinfo[0][1]}",
        x2_axis=q_final,
        x2_axis_name="Q_angstrom^-1",
    )
    save_1d_integration_static(cfg, hf, outresult)


def save_qperp_qpara(experiment, hf, qperp_qpara_map, scan=0):
    """
    save a qpara vs qperp map to hdf5 file

    """
    dset = hf.create_group("qperp_qpara")
    dset.create_dataset("images", data=qperp_qpara_map[0])
    dset.create_dataset("qpararanges", data=qperp_qpara_map[1])
    dset.create_dataset("qperpranges", data=qperp_qpara_map[2])
    if "scanfields" not in hf.keys():
        save_scan_field_values(hf, scan)

    if experiment.savetiffs is True:
        do_savetiffs(hf, qperp_qpara_map[0], qperp_qpara_map[1], qperp_qpara_map[2])


def save_config_variables(hf, process_config):
    """
    save all variables in the configuration file to the output hdf5 file
    """
    cfg = process_config
    config_group = hf.create_group("i07configuration")
    outdict = vars(cfg)
    with open(cfg.default_config_path, "r", encoding="utf-8") as f:
        default_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    # add in extra to defaults that arent set by user, so that parsing
    # defaults finds it
    default_config_dict["ubinfo"] = 0
    default_config_dict["pythonlocation"] = 0
    default_config_dict["joblines"] = 0
    for key in default_config_dict:
        if key == "ubinfo":
            for i, coll in enumerate(outdict["ubinfo"]):
                ubgroup = config_group.create_group(f"ubinfo_{i + 1}")
                ubgroup.create_dataset(
                    f"lattice_{i + 1}", data=coll["diffcalc_lattice"]
                )
                ubgroup.create_dataset(f"u_{i + 1}", data=coll["diffcalc_u"])
                ubgroup.create_dataset(f"ub_{i + 1}", data=coll["diffcalc_ub"])
            continue
        val = outdict[key]
        if val is None:
            val = "None"
        config_group.create_dataset(f"{key}", data=val)


def save_scan_field_values(hf, scan):
    """
    saves scanfields recorded in nexus file to hdf5 output
    """

    try:
        rank = scan.metadata.data_file.diamond_scan.scan_rank.nxdata
        fields = scan.metadata.data_file.diamond_scan.scan_fields
        scanned = [x.decode("utf-8").split(".")[0] for x in fields[:rank].nxdata]
        scannedvalues = [
            np.unique(scan.metadata.data_file.nx_instrument[field].value)
            for field in scanned
        ]
        scannedvaluesout = [
            scannedvals[~np.isnan(scannedvals)] for scannedvals in scannedvalues
        ]
    except BaseException:
        scanned, scannedvaluesout = None, None

    dset = hf.create_group("scanfields")
    if scan != 0:
        if scanned is not None:
            for i, field in enumerate(scanned):
                dset.create_dataset(f"dim{i}_{field}", data=scannedvaluesout[i])


def save_hf_map_static(hf, cfg, start_time, mapname, mapdata, mapaxisinfo, scan=None):
    end_time = time()
    times = [start_time, end_time]
    dset = hf.create_group(f"{mapname}")
    dset.create_dataset(f"{mapname}_map", data=mapdata)
    dset.create_dataset("map_para", data=mapaxisinfo[1])
    dset.create_dataset("map_para_unit", data=mapaxisinfo[3])
    dset.create_dataset("map_perp", data=mapaxisinfo[0])
    dset.create_dataset("map_perp_unit", data=mapaxisinfo[2])
    dset.create_dataset("map_perp_indices", data=[0, 1, 2])
    dset.create_dataset("map_para_indices", data=[0, 1, 3])

    if (scan is not None) & ("scanfields" not in hf.keys()):
        save_scan_field_values(hf, scan)
    if cfg.savetiffs:
        do_savetiffs(hf, mapdata, mapaxisinfo[1], mapaxisinfo[0])
    save_config_variables(hf, cfg)
    hf.close()
    minutes = (times[1] - times[0]) / 60
    print(f"total calculation took {minutes}  minutes")


def save_hf_map(
    hf,
    mapname,
    sum_array,
    counts_array,
    mapaxisinfo,
    start_time,
    process_config,
):
    cfg = process_config
    norm_array = np.divide(
        sum_array, counts_array, out=np.copy(sum_array), where=counts_array != 0.0
    )
    save_hf_map_static(hf, cfg, start_time, mapname, norm_array, mapaxisinfo)


def save_masks(hf, mask_list):
    dset = hf.create_group("masks")
    dset.create_dataset("total_mask", data=mask_list[0])
    dset.create_dataset("image_mask", data=mask_list[1])
    dset.create_dataset("sector_mask", data=mask_list[2])


def linear_bin_to_vtk(
    binned_data: np.ndarray,
    file_path: str,
    start: np.ndarray,
    stop: np.ndarray,
    step: np.ndarray,
) -> None:
    """
    Takes binned data and saves it to a .vtk file.
    """
    # This is needed by the pyevtk library.
    file_path = str(file_path)

    # Coordinates

    x_range = np.arange(start[0], stop[0], step[0], dtype="float32")
    x_range = list(x_range)
    x_range.append(stop[0])
    x_range = np.array(x_range, dtype=np.float32)

    y_range = np.arange(start[1], stop[1], step[1], dtype="float32")
    y_range = list(y_range)
    y_range.append(stop[1])
    y_range = np.array(y_range, dtype=np.float32)

    z_range = np.arange(start[2], stop[2], step[2], dtype="float32")
    z_range = list(z_range)
    z_range.append(stop[2])
    z_range = np.array(z_range, dtype=np.float32)

    gridToVTK(file_path, x_range, y_range, z_range, cellData={"Intensity": binned_data})
