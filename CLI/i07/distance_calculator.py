import argparse
import os
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import matplotlib
import numpy as np
from diffraction_utils.diffractometers.diamond_i07 import I07Diffractometer
from diffraction_utils.io import I07Nexus
from fast_rsm.rsm_metadata import RSMMetadata
from lmfit.models import LinearModel

os.environ["QT_API"] = "pyqt5"

matplotlib.use("QtAgg")


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
    def __init__(self, startpos: float, pos_list: list, imlist: list):
        self.startpos = startpos
        self.pos_list = pos_list
        self.imlist = imlist

    def update_lists(self, pos, im):
        self.pos_list.append(pos[0])
        self.imlist.append(im)

    def calc_shifts(self):
        self.peak_ends = self.pos_list[1:]
        self.peak_shifts = np.array(
            [endpeak - self.pos_list[i] for i, endpeak in enumerate(self.peak_ends)]
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


def findpeaks(data: np.ndarray):
    ranges = np.array([])
    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        if len(group) <= 2:
            continue
        np.append(ranges, (group[0] + group[-1]) / 2)
    return ranges


def get_line_profiles(
    loader: easyloader, selection: np.ndarray, central_pixel: int
) -> np.ndarray:

    lineprofiles = np.zeros(
        (len(selection), loader.diff.data_file.image_shape[1])
    )  # loader.diff.data_file.image_shape[1]
    for line_ind, ind in enumerate(selection):
        start = central_pixel - 10
        stop = central_pixel + 10
        section = loader.get_image(ind)[start:stop, :]
        lineprofiles[line_ind] = np.sum(section, axis=0)
    return lineprofiles


def calc_dist(theta, Opp):
    return Opp / np.tan(np.radians(theta))


def identify_peaks(lineprofiles: np.ndarray) -> list:
    track_peaks: list[profpeak] = []
    finished_peaks: list[profpeak] = []
    for line_ind, lineprof in enumerate(lineprofiles):
        peaks = np.argwhere(lineprof > (lineprof.mean() + 2.5 * lineprof.std()))
        outpeaks = findpeaks(peaks)
        for i, peak in enumerate(outpeaks):
            if len(outpeaks) < len(track_peaks):
                finished_peaks.append(track_peaks[0])
                track_peaks = track_peaks[1:]
            if (len(track_peaks) >= i + 1) and (
                (line_ind - 1) in track_peaks[i].imlist
            ):
                track_peaks[i].update_lists(peak, line_ind)

            else:
                track_peaks.append(
                    profpeak(startpos=peak, pos_list=[peak], imlist=[line_ind])
                )

    return [track_peaks, finished_peaks]


def fit_linear_models(peaks: list[profpeak]):
    slopes = np.array([])
    slopes_err = np.array([])
    print("fitting linear models")
    for i, peak in enumerate(peaks):
        yvals = np.array([pos[0] for pos in peak.pos_list])
        xvals = np.arange(len(yvals))
        result = do_linear_fit(xvals, yvals)
        # axs.bar(i,result.params['line_slope'].value)
        slopes = np.append(slopes, result.params["line_slope"].value)
        slopes_err = np.append(slopes_err, result.params["line_slope"].stderr)

    sqrs = slopes_err**2
    sum_sqrt = np.sqrt(np.sum(sqrs))
    slope_meanerr = sum_sqrt / len(slopes)

    return slopes, slope_meanerr


def calc_distance(nexusfile: str, datapath: str, central_pixel: int, image_inds: tuple):

    loader = easyloader(nexusfile=nexusfile, datapath=datapath)
    selection = np.arange(image_inds[0], image_inds[1], 1)
    line_profiles = get_line_profiles(loader, selection, central_pixel)
    tracked_peaks, finished_peaks = identify_peaks(line_profiles)
    fullpeaks = [peak for peak in finished_peaks if len(peak.pos_list) > 10]
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

    print(f"distance = {d1} +/- {d2 - d1}")


if __name__ == "__main__":
    HELP_STR = (
        "takes in paths to file and central pixel perpendicular to detector movement direction,\n"
        + "finds signals in images and calculates distance based on pixel shifts and angles"
    )
    parser = argparse.ArgumentParser(description=HELP_STR)

    HELP_STR = "Path to data directory"
    parser.add_argument("-dir", "--dir_path", help=HELP_STR)

    HELP_STR = "Separate scan numbers to be mapped into one reciprocal volume without brackets e.g 441124 441128"
    parser.add_argument("-s", "--scan_num", help=HELP_STR)

    HELP_STR = "central pixel in direction perpendicular to detector movement"
    parser.add_argument("-cen", "--cen_pix", help=HELP_STR)

    HELP_STR = "index of image to start tracking peaks"
    parser.add_argument("-imstart", "--start_image", type=int, help=HELP_STR)

    HELP_STR = "index of image to end tracking peaks"
    parser.add_argument("-imend", "--end_image", type=int, help=HELP_STR)

    args = parser.parse_args()
    datapath = Path(args.dir_path)
    infile = datapath / f"i07-{args.scan_num}.nxs"
    im_indices = (args.start_image, args.end_image)
    calc_distance(
        nexusfile=str(infile),
        datapath=str(datapath),
        central_pixel=int(args.cen_pix),
        image_inds=im_indices,
    )
