import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from diffraction_utils.diffractometers.diamond_i07 import I07Diffractometer
from diffraction_utils.io import I07Nexus
from lmfit.models import (
    GaussianModel,
    LinearModel,
    LorentzianModel,
    PseudoVoigtModel,
)
from scipy.signal import find_peaks, savgol_filter

from fast_rsm.rsm_metadata import RSMMetadata

os.environ["QT_API"] = "pyqt5"

# matplotlib.use("QtAgg")


class easyloader:
    def __init__(
        self,
        nexusfile: str,
        datapath: str,
        sample_oop: np.ndarray = np.array([0, 0, 1]),
    ):
        self.nexusfile = nexusfile
        self.datapath = datapath
        self.sample_oop = sample_oop

        self.load_dataset()

    def load_dataset(self):
        self.nex = I07Nexus(self.nexusfile, self.datapath)
        self.diff = I07Diffractometer(self.nex, self.sample_oop)
        self.meta = RSMMetadata(self.diff, (0, 0))

    def get_image(self, ind: int):
        return self.diff.data_file.get_image(ind)


class profpeak:
    def __init__(
        self,
        startpos: float,
        pos_list: list,
        imlist: list,
        range_list: list,
        current_params: dict,
    ):
        self.startpos = startpos
        self.pos_list = pos_list
        self.imlist = imlist
        self.range_list = range_list
        self.current_params = current_params

    def update_lists(self, pos, im, range, params):
        self.pos_list.append(pos)
        self.imlist.append(im)
        self.range_list.append(range)
        self.current_params = params

    def calc_shifts(self):
        self.peak_ends = self.pos_list[1:]
        self.peak_shifts = np.array(
            [endpeak - self.pos_list[i] for i, endpeak in enumerate(self.peak_ends)]
        )


class PeakSet:
    def __init__(self, tracked_peaks: list[profpeak], finished_peaks: list[profpeak]):
        self.track_peaks = tracked_peaks
        self.finish_peaks = finished_peaks

    def add_new_peak(self, peak, index, range, params):
        self.track_peaks.append(
            profpeak(
                startpos=peak,
                pos_list=[peak],
                imlist=[index],
                range_list=[range],
                current_params=params,
            )
        )

    def end_peak(
        self,
    ):
        self.finish_peaks.append(self.track_peaks[0])
        self.track_peaks = self.track_peaks[1:]

    def plot_peaks(self):
        fig, axs = plt.subplots(1, 2)
        axlist = axs.ravel()
        for peak in self.track_peaks:
            axs[0].plot(peak.pos_list)
        for peak in self.finish_peaks:
            axs[1].plot(peak.pos_list)
        plt.show()


def parse_yaml(yaml_file: str):
    with open(yaml_file, "r") as file:
        loaded_settings = yaml.safe_load(file)

    DIR = loaded_settings["dir"]
    OUTDIR = loaded_settings["outdir"]
    SCAN = loaded_settings["scan"]
    CEN_PIX = loaded_settings["cen_pix"]
    IM_START = loaded_settings["im_start"]
    IM_END = loaded_settings["im_end"]
    AXIS = loaded_settings["axis"]
    CLIP = loaded_settings["clip_limits"]

    nexusfile = Path(DIR) / f"i07-{SCAN}.nxs"
    return dict(
        nexusfile=str(nexusfile),
        datapath=DIR,
        central_pixel=CEN_PIX,
        move_axis=AXIS,
        imstart=IM_START,
        imend=IM_END,
        clip_values=CLIP,
        outdir=OUTDIR,
    )


def do_linear_fit(x, y):
    line_mod = LinearModel(prefix="line_")
    mod = line_mod
    params = mod.make_params()
    params["line_intercept"].set(y.min())
    params["line_slope"].set(10)
    result = mod.fit(y, params, x=x)
    # xnew = np.arange(x.min(), x.max(), 0.001)
    # comps = result.eval_components(x=xnew)
    # y_fit = result.best_fit
    return result


def findpeaks(data: np.ndarray, len_limit: int, intensities: np.ndarray):
    ranges = []
    peak_positions = np.array([])
    peakdf = pd.DataFrame({"peak_pos": data, "intensities": intensities})
    peakdf["group"] = peakdf.peak_pos - peakdf.index
    group_stats = peakdf.groupby("group").agg(
        length=("peak_pos", "size"),
        min_peak=("peak_pos", "min"),
        max_peak=("peak_pos", "max"),
        max_int=("intensities", "max"),
    )
    min_peak = 10
    max_peak = len_limit - 10

    valid_groups = group_stats[
        (group_stats["min_peak"] >= min_peak)
        & (group_stats["max_peak"] <= max_peak)
        & (group_stats["length"] >= 5)
        & (group_stats["max_int"] >= 10.0)
    ].index
    filtered_peakdf = peakdf[peakdf["group"].isin(valid_groups)]
    for group_num in np.unique(filtered_peakdf["group"]):
        groupdf = filtered_peakdf[filtered_peakdf["group"] == group_num]
        peakval = groupdf.loc[groupdf["intensities"].idxmax(), "peak_pos"]
        peak_positions = np.append(peak_positions, peakval)
        ranges.append([(groupdf["peak_pos"].min(), groupdf["peak_pos"].max())])

    # for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
    #     group = np.array(list(map(itemgetter(1), g)))
    #     if len(group) <= 2:
    #         continue
    #     if group.min() < 10:
    #         continue
    #     if group.max() > len_limit - 10:
    #         continue
    #     # group_intensities=intensities[]
    #     peakval = data[np.argwhere(intensities == intensities.max())]

    #     peak_positions = np.append(peak_positions, peakval)
    #     ranges.append([(group[0], group[-1])])

    return peak_positions[::-1], ranges[::-1]


def get_line_profiles(
    loader: easyloader,
    selection: np.ndarray,
    central_pixel: int,
    axis: str,
    clip_values: list,
) -> np.ndarray:

    start = central_pixel - 15
    stop = central_pixel + 15
    if axis == "x":
        selectionslice = (slice(start, stop), slice(clip_values[0], clip_values[1]))
        axis_ind = 0
        shape_ind = 1
    else:
        selectionslice = (slice(clip_values[0], clip_values[1]), slice(start, stop))
        axis_ind = 1
        shape_ind = 0

    lineprofiles = np.zeros((len(selection), clip_values[1] - clip_values[0]))

    for line_ind, ind in enumerate(selection):
        section = loader.get_image(ind)[selectionslice]
        lineprofiles[line_ind] = np.sum(section, axis=axis_ind)
    return lineprofiles


def parse_peak_kwargs(peakinfo, prefix):
    outparams = []
    for k, v in peakinfo.items():
        outparams.append([f"{prefix}{k}", v[0], v[1], v[2]])
    return outparams


def fit_peaks(peaklist: list, x: np.ndarray, y: np.ndarray, background=None):

    fit_models = {
        "pvoigt": PseudoVoigtModel,
        "gaussian": GaussianModel,
        "lorentzian": LorentzianModel,
    }
    if background is None:
        background = LinearModel(prefix="bkg_")
    mod = background
    par_settings = []
    peak_weights = np.sqrt(y)
    for ind, peak in enumerate(peaklist):
        peakprefix = f"p{ind + 1}_"
        mod += fit_models[peak["type"]](prefix=peakprefix)
        par_settings += parse_peak_kwargs(peak["settings"], peakprefix)
        if peak["weights"]:
            peak_weights /= peak["weights"]

    pars = mod.make_params()

    pars["bkg_intercept"].set(y.min())
    yslope = (y[-1] - y[0]) / len(y)
    pars["bkg_slope"].set(yslope, min=yslope * 0.15, max=yslope * 1.5)
    for par in par_settings:
        pars[par[0]].set(par[1], min=par[2], max=par[3])
    result = mod.fit(y, pars, x=x)
    comps = result.eval_components(x=x)
    return result, comps


def calc_dist(theta, Opp):
    return Opp / np.tan(np.radians(theta))


def make_peak(outpeak, prof_len, properties):

    p1cen = outpeak

    p1width = prof_len * 0.05
    p1amp = properties["peak_heights"][0]
    new_peak = {
        "type": "pvoigt",
        "settings": {
            "center": [p1cen, p1cen * 0.95, p1cen * 1.05],
            "sigma": [p1width, p1width * 0.1, p1width * 10],
            "height": [p1amp, p1amp * 0.5, p1amp * 1.2],
            "fraction": [0.15, 0, 1],
        },
        "weights": None,
    }
    return new_peak


def overwrite_params(peak, new_params, profile_length):
    peak_tag = list(new_params.keys())[0].split("_")[0]
    outpeak = peak.copy()

    newlist = ["fraction", "sigma", "center"]
    for name in newlist:
        newval = new_params[f"{peak_tag}_{name}"]
        outpeak["settings"][f"{name}"] = [newval, 0.975 * newval, 1.025 * newval]
    outpeak["settings"]["center"][2] = np.max(
        [new_params[f"{peak_tag}_center"] * 1.05, peak["settings"]["center"][2]]
    )
    outpeak["settings"]["center"][1] = new_params[f"{peak_tag}_center"]
    shifts = np.abs(np.arange(0, profile_length) - new_params[f"{peak_tag}_center"])
    # outweights = [2 if val < 10 else 1 for val in shifts]

    # outpeak["weights"] = outweights

    # outpeak["settings"]["sigma"][1] = new_params[f"{peak_tag}_sigma"] * 1.2
    return outpeak


def check_peak_limits(res, outpeaks, prof_len):

    if res.rsquared < 0.8:
        return np.array([])
    outpeaks = [
        res.params[f"p{num + 1}_center"].value for num in np.arange(len(outpeaks))
    ]
    outarr = np.array(outpeaks)
    numvals = np.arange(len(outarr)) + 1
    filter_low = np.array(
        [
            res.params[f"p{num}_center"] - (res.params[f"p{num}_fwhm"]) * 0.8 > 0
            for num in numvals
        ]
    )
    filter_high = np.array(
        [
            res.params[f"p{num}_center"] + (res.params[f"p{num}_fwhm"]) / 2
            < prof_len - 10
            for num in numvals
        ]
    )
    filter_fwhm = np.array([True] * len(outarr))
    if all(res.params[f"p{num}_fwhm"].stderr is not None for num in numvals):
        filter_fwhm = np.array(
            [
                res.params[f"p{num}_fwhm"].stderr / res.params[f"p{num}_fwhm"].value
                < 0.02
                for num in numvals
            ]
        )

    total_filter = [
        all([f1, f2, f3]) for f1, f2, f3 in zip(filter_high, filter_low, filter_fwhm)
    ]
    outarr_fwhm = outarr[total_filter]
    return outarr_fwhm


def plot_fitting(lineprof, smooth_prof, res, comps, outpeaks, peaklist):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axlist = axs.ravel()

    axlist[0].plot(lineprof, color="lightblue")
    axlist[0].plot(smooth_prof, color="orange")
    [axlist[0].axvline(val) for val in outpeaks]
    res.plot_fit(ax=axlist[1])
    axlist[1].set_ylim(0, 25)
    axlist[1].set_xlim(0, 300)
    for pnum in np.arange(len(peaklist)) + 1:
        axlist[1].plot(
            comps[f"p{pnum}_"],
            color="tab:green",
            lw=2,
            label=f"Peak {pnum} (p{pnum}_)",
        )
    axlist[1].plot(
        comps["bkg_"],
        color="tab:blue",
        ls="--",
        lw=2,
        label="Background (bkg_)",
    )

    plt.show()


def make_peaklist(outpeaks, prof_len, properties, peak_set: PeakSet):

    newlist = [make_peak(outpeak, prof_len, properties) for outpeak in outpeaks]
    for i, peak in enumerate(newlist):
        if len(peak_set.track_peaks) < i + 1:
            continue
        newlist[i] = overwrite_params(
            peak, peak_set.track_peaks[i].current_params, prof_len
        )
    return newlist


def smooth_savgol(lineprof):
    smooth_prof = savgol_filter(
        lineprof, window_length=int(np.max([5, len(lineprof) / 20])), polyorder=3
    )  # smoothed_func(xdata)
    # smooth_prof = lineprof
    smooth_prof -= smooth_prof.min()
    return smooth_prof


def check_smoothed(smooth_prof, num_peaks) -> bool:
    check_ends = abs(smooth_prof[-1] - smooth_prof[0]) > 0.5 * smooth_prof.max()
    if (num_peaks == 0) and (check_ends):
        return False
    return True


def check_start_background(smooth_prof: np.ndarray) -> int:
    meancheck = np.array(
        [
            np.mean(smooth_prof[start : start + 50])
            for start in np.arange(len(smooth_prof) - 50)
        ]
    )
    upper = meancheck.mean() * 1.25
    good_start = np.where(meancheck < upper)
    return int(np.min(good_start))


def identify_peaks(lineprofiles: np.ndarray) -> list:
    peak_set = PeakSet(tracked_peaks=[], finished_peaks=[])
    for line_ind, lineprof in enumerate(lineprofiles):
        print(f"DEBUG LINE - index: {line_ind}")
        prof_len = len(lineprof)
        xdata = np.arange(0, len(lineprof))
        # smoothed_func = make_smoothing_spline(xdata,lineprof)
        smooth_prof = smooth_savgol(lineprof)

        background_start = check_start_background(smooth_prof)
        # DEBUGLINE
        if line_ind in (36, 25, 61, 18):
            print("pause line")

        fitpeaks, properties = find_peaks(
            smooth_prof,
            prominence=0.175 * smooth_prof.std(),
            height=smooth_prof.mean() * 1.75,
            distance=10,
        )
        outpeaks = fitpeaks[fitpeaks > background_start + 20][::-1]
        if len(outpeaks) == 0:
            continue
        if not check_smoothed(smooth_prof, len(outpeaks)):
            continue

        peaklist = make_peaklist(outpeaks, prof_len, properties, peak_set)
        res, comps = fit_peaks(peaklist, np.arange(len(smooth_prof)), smooth_prof)
        plot_fitting(lineprof, smooth_prof, res, comps, outpeaks, peaklist)

        plt.close()

        outpeaks = check_peak_limits(res, outpeaks, prof_len)
        # plt.plot(lineprof);plt.plot(xdata,smooth_prof);plt.show()
        # plt.close()
        for i, peak_info in enumerate(outpeaks):
            print("found good peak")
            assign_peak(peak_info, peak_set, res, outpeaks, line_ind, smooth_prof, i)

    return [peak_set.track_peaks, peak_set.finish_peaks]


def assign_peak(peak_info, peak_set, res, outpeaks, line_ind, smooth_prof, ind):
    peakpos = peak_info

    # peak_range = peak_info[1]
    peak_range = (0, 1)
    pkeys = [k for k in res.params.keys() if f"p{ind + 1}" in k]
    fitparams = {k: res.params[k].value for k in pkeys}
    if len(outpeaks) < len(peak_set.track_peaks):
        peak_set.end_peak()

    if len(peak_set.track_peaks) < ind + 1:
        peak_set.add_new_peak(peakpos, line_ind, peak_range, fitparams)
        return

    poslist_i = np.array(peak_set.track_peaks[ind].pos_list)
    posrange_i = np.array(peak_set.track_peaks[ind].range_list)
    if (peakpos > poslist_i[-1]) and (len(poslist_i) == 1):
        peak_set.track_peaks[ind].update_lists(peakpos, line_ind, peak_range, fitparams)
        return

    if (peakpos < posrange_i.min()) or (
        (peakpos - poslist_i[-1]) > 0.75 * len(smooth_prof)
    ):
        peak_set.end_peak()
        peak_set.add_new_peak(peakpos, line_ind, peak_range, fitparams)
        return
    peak_set.track_peaks[ind].update_lists(peakpos, line_ind, peak_range, fitparams)


def fit_linear_models(peaks: list[profpeak]):
    slopes = np.array([])
    slopes_err = np.array([])
    print("fitting linear models")
    for i, peak in enumerate(peaks):
        yvals = np.array(peak.pos_list)
        xvals = np.arange(len(yvals))
        result = do_linear_fit(xvals, yvals)
        # axs.bar(i,result.params['line_slope'].value)
        slopes = np.append(slopes, result.params["line_slope"].value)
        slopes_err = np.append(slopes_err, result.params["line_slope"].stderr)

    sqrs = slopes_err**2
    sum_sqrt = np.sqrt(np.sum(sqrs))
    slope_meanerr = sum_sqrt / len(slopes)

    return slopes, slope_meanerr


def calc_distance(
    nexusfile: str,
    datapath: str,
    central_pixel: int,
    imstart: int,
    imend: int,
    move_axis: str,
    clip_values: list,
    outdir: str,
):

    loader = easyloader(nexusfile=nexusfile, datapath=datapath)
    selection = np.arange(imstart, imend, 1)
    line_profiles = get_line_profiles(
        loader, selection, central_pixel, move_axis, clip_values
    )
    tracked_peaks, finished_peaks = identify_peaks(line_profiles)
    fullpeaks = [
        peak for peak in finished_peaks + tracked_peaks if len(peak.pos_list) > 1
    ]
    print(f"found {len(fullpeaks)} good tracked signals")
    slopes, slopes_meanerr = fit_linear_models(fullpeaks)

    avslope = slopes.mean()
    print(f"average_slope = {avslope} +/- {slopes_meanerr}")

    pixsize = loader.diff.data_file.pixel_size
    selected_angles = loader.diff.data_file.default_axis[selection]
    selected_angles_end = selected_angles[1:]
    ang_shift = np.array(
        [
            end_angle - selected_angles[i]
            for i, end_angle in enumerate(selected_angles_end)
        ]
    )

    d1 = calc_dist(ang_shift.mean(), avslope * pixsize)
    d2 = calc_dist(ang_shift.mean(), (avslope + (slopes_meanerr)) * pixsize)

    print(f"distance (m) = {d1} +/- {d2 - d1}")
