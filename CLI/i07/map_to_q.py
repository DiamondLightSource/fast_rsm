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


if __name__ == "__main__":
    # First deal with the parsing of the command line arguments using the
    # argparse library.
    HELP_STR = (
        "Maps a single image into Q-space. Plots intensity as a function of Q."
    )
    parser = argparse.ArgumentParser(description=HELP_STR)

    # The most important argument is the path to the data. If this is not
    # provided, we'll assume that we're in the data directory. Note that the
    # default argument is never passed to add_argument because the default
    # behaviour implemented here is too complex in some cases to be replaced by
    # simple hardcoded values. Instead, default values are calculated after
    # parse_args is called.
    HELP_STR = (
        "Path to the directory in which the data is stored. "
        "Defaults to the DATA_PATH environment variable. "
        "If this is not specified, your current directory will be used."
    )
    parser.add_argument("-d", "--data_path", help=HELP_STR,
                        default=os.environ.get('DATA_PATH'))

    HELP_STR = (
        "The scan number of the scan that you want to process. "
        "This must be provided. Duh."
    )
    parser.add_argument("-N", "--scan_number", help=HELP_STR, type=int)

    HELP_STR = (
        "The sample-detector distance. Defaults to the DETECTOR_DISTANCE "
        "environment variable."
    )
    parser.add_argument("-D", "--detector_distance", help=HELP_STR, type=float,
                        default=os.environ.get('DETECTOR_DISTANCE'))

    HELP_STR = (
        "The x pixel hit by the beam when all the diffractometer motors are "
        "zeroed. Defaults to the BEAM_CENTRE_X environment variable. This must "
        "be measured from the left of the image."
    )
    parser.add_argument("-x", "--beam_centre_x", help=HELP_STR, type=int,
                        default=os.environ.get("BEAM_CENTRE_X"))

    HELP_STR = (
        "The y pixel hit by the beam when all the diffractometer motors are "
        "zeroed. Defaults to the BEAM_CENTRE_Y environment variable. This must "
        "be measured **from the top of the image**. As in, a value of 50 means "
        "'50 pixels from the top'. Don't blame me for this convention."
    )
    parser.add_argument("-y", "--beam_centre_y", help=HELP_STR, type=int,
                        default=os.environ.get("BEAM_CENTRE_Y"))

    HELP_STR = (
        "Add --plot if you want your data to be plotted."
    )
    parser.add_argument("--plot", help=HELP_STR, action="store_true")

    HELP_STR = (
        "Add --log if you want your --plot to be on a log scale."
    )
    parser.add_argument("--log", help=HELP_STR, action="store_true")

    HELP_STR = (
        "Add --nodelta to ignore the delta motor."
    )
    parser.add_argument("--nodelta", help=HELP_STR, action="store_true")

    HELP_STR = (
        "NOT ESSENTIAL (defaults sensibly).\n"
        "The number of bins in Q. Defaults to 1000, or the NUM_BINS "
        "environment variable if it has been set."
    )
    parser.add_argument("-b", "--num_bins", help=HELP_STR, type=int,
                        default=os.environ.get("NUM_BINS"))

    HELP_STR = (
        "The minimum value to threshold to. Defaults to MIN_THRESH environment "
        "variable if set. If it isn't set, defaults to 0."
    )
    parser.add_argument("--min_thresh", help=HELP_STR, type=float,
                        default=os.environ.get("MIN_THRESH"))
    HELP_STR = (
        "The maximum value to threshold to. Defaults to MAX_THRESH environment "
        "variable if set. If it isn't set, defaults to infinity."
    )
    parser.add_argument("--max_thresh", help=HELP_STR, type=float,
                        default=os.environ.get("MAX_THRESH"))

    HELP_STR = (
        "NOT ESSENTIAL (defaults sensibly).\n"
        "Specify the directory in which you would like your mapped data to be "
        "stored. Defaults to the OUTPUT environment variable. "
        "If neither are specified, will default to data_path/processing/"
    )
    parser.add_argument("-o", "--output", help=HELP_STR,
                        default=os.environ.get('OUTPUT'))

    # Extract the arguments from the parser.
    args = parser.parse_args()

    # Now we need to generate default values of inputs, where required.
    # Default to local dir.
    if args.data_path is None:
        args.data_path = "./"
    if not args.data_path.endswith("/"):
        args.data_path = args.data_path + "/"

    # Default to data_path/processing/.
    args.processing_path = args.data_path + "processing/"

    if args.output is None:
        args.output = args.processing_path
    if not args.output.endswith("/"):
        args.output = args.output + "/"

    if args.scan_number is None:
        raise ValueError("You forgot to give a scan number...")
    if args.detector_distance is None:
        raise ValueError("You forgot to give a detector distance...")
    if args.beam_centre_x is None:
        raise ValueError("You forgot to give the x beam centre...")
    if args.beam_centre_y is None:
        raise ValueError("You forgot to give the y beam centre...")
    if args.num_bins is None:
        args.num_bins = 1000

    min_thresh = args.min_thresh if args.min_thresh is not None else 0
    max_thresh = args.max_thresh if args.max_thresh is not None else float(
        'inf')

    # We should now have scraped all the data that we need to map an image.
    # First work out where our .nxs file should be stored.
    file_name = "i07-" + str(args.scan_number) + ".nxs"
    path_to_nx = args.data_path + file_name
    beam_centre = (args.beam_centre_x, args.beam_centre_y)
    detector_distance = args.detector_distance
    setup = 'horizontal'
    sample_oop = np.array([0, 1, 0])  # This isn't going to be used.

    # Work out the directory in which stuff will be stored.
    tiff_dir = os.path.dirname(I07Nexus(path_to_nx).raw_image_paths[0])

    # Now we can instantiate a Scan.
    scan = Scan.from_i07(path_to_nx=path_to_nx,
                         beam_centre=beam_centre,
                         detector_distance=detector_distance,
                         setup=setup,
                         sample_oop=sample_oop,
                         path_to_data=tiff_dir)

    def threshold(array):
        array[array >= max_thresh] = max_thresh
        array[array <= min_thresh] = 0
        return array

    # This is designed to be used when there's only 1 image in a scan.
    scan.add_processing_step(threshold)
    image = scan.load_image(0)

    # Ignore the delta motor if asked to.
    if args.nodelta:
        image.metadata.data_file.delta = np.array([0])
        scan.metadata.data_file.delta = np.array([0])

    data = threshold(image.data)

    # Map to reciprocal space.
    print("Loaded image! Beginning map...")
    map_frame = Frame(Frame.sample_holder, scan.metadata.diffractometer, 0)
    q = image.delta_q(map_frame)
    print(f"Map complete! Binning into {args.num_bins} bins.")

    # Now do some binning.
    qs_flattened = q.reshape((scan.metadata.data_file.image_shape[0] *
                              scan.metadata.data_file.image_shape[1], 3))
    data_flattened = data.reshape(
        (scan.metadata.data_file.image_shape[0] *
         scan.metadata.data_file.image_shape[1]))

    q_lengths = np.linalg.norm(qs_flattened, axis=1)
    num_bins = args.num_bins  # Grab from the arg parser.

    # Put the intensities into bins.
    y, _ = np.histogram(q_lengths, num_bins, weights=data_flattened)
    # Normalise the intensity in each bin.
    bincount, _ = np.histogram(q_lengths, num_bins)
    y /= bincount

    x = np.arange(0, np.max(q_lengths), np.max(q_lengths)/num_bins)
    # Correct the sizes, if needed.
    if x.shape[0] == y.shape[0] + 1:
        x = x[:-1]

    # Save the data.
    datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    save_as = str(args.scan_number) + "_mapped_to_Q_" + datetime_str + ".dat"
    save_as = args.output + save_as
    col_data = np.array([x, y]).transpose()
    np.savetxt(save_as, col_data, header="Q(1/Å) I(a.u.)")
    print(f"Data mapped! Saving to {save_as}")

    # Plot the data if the user wanted this.
    if args.plot:
        print("Data saved! Plotting...")
        import plotly.graph_objects as go
        fig = go.Figure().update_layout(title="Raw image",
                                        xaxis_title='x-pixels',
                                        yaxis_title='y-pixels')
        if args.log:
            fig.add_trace(go.Heatmap(z=np.log(data+1), colorscale='Jet'))
        else:
            fig.add_trace(go.Heatmap(z=data, colorscale='Jet'))
        fig.show()

        title = "Intensity vs Q"
        fig = go.Figure().update_layout(title=title,
                                        xaxis_title='Q (1/Å)',
                                        yaxis_title='Intensity (a.u.)')
        fig.add_trace(go.Scatter(x=x, y=y))
        if args.log:
            fig.update_yaxes(type="log")
        fig.show()
