"""
This file contains a suite of utility functions for processing data acquired
specifically at Diamond.
"""

import os
from typing import Tuple

import fast_histogram
import numpy as np
from scipy.constants import physical_constants

import nexusformat.nexus as nx

from .binning import weighted_bin_1d


def load_exact_map(
    q_vector_path: str, intensities_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns exact q vectors and their corresponding intensities, after loading
    them from the provided paths.

    Args:
        q_vector_path:
            Path to exact q vectors.
        intensities_path:
            Path to either corrected or uncorrected intensities.

    Returns:
        (q_vectors, intensities)
        Where, if intensities has a shape (1000,), q_vectors has a shape
        (1000, 3). The intensity measured at q_vectors[i] is intensities[i].
    """
    raw_q = np.load(q_vector_path)
    raw_intensities = np.load(intensities_path)
    raw_q = raw_q.reshape((raw_intensities.shape[0], 3))
    return raw_q, raw_intensities


def intensity_vs_q_exact(
        q_vectors: np.ndarray, intensities: np.ndarray, num_bins=1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This routine is currently only available to data acquired when the
    "map_per_image" option is checked. Note that the "map_per_image" option can
    generate enormous quantities of data (generally producing 5x your input
    data, each time you run a calculation). Q vectors and intensities provided
    need to be obtained using the "load_exact_map" function in this module.

    Args:
        q_vectors:
            An array of q_vectors obtained using the "load_exact_map" function.
        intensities_path:
            An array of intensities loaded using the "load_exact_map" function.
        num_bins:
            Desired length of your output Q and Intensity arrays.

    Returns:
        A (Q, intensity) tuple, where both Q and intensity are represented by
        numpy arrays of length num_bins.
    """
    # These arrays could be affected by the call to weighted_bin_1d. Make sure
    # that callers don't have their objects mangled by making copies here.
    q_vectors, intensities = np.copy(q_vectors), np.copy(intensities)
    q_lengths = np.linalg.norm(q_vectors, axis=1)

    out = np.zeros((num_bins,), np.float32)
    count = np.zeros((num_bins,), np.uint32)

    start = float(np.nanmin(q_lengths))
    stop = float(np.nanmax(q_lengths))
    step = float((stop - start)/num_bins)

    weighted_bin_1d(q_lengths, intensities, out, count, start, stop, step)

    binned_qs = np.linspace(start, stop, num_bins)
    intensities = out/count.astype(np.float32)

    return binned_qs, intensities


def qxy_qz_exact(
    q_vectors: str,
    intensities: str,
    qxy_bins: int = 1000,
    qz_bins: int = 1000,
    qz_axis: int = 2
) -> np.ndarray:
    """
    This routine is currently only available to data acquired when the
    "map_per_image" option is checked. Note that the "map_per_image" option can
    generate enormous quantities of data (generally producing 5x your input
    data, each time you run a calculation). Q vectors and intensities provided
    need to be obtained using the "load_exact_map" function in this module.

    Args:
        q_vectors:
            An array of q_vectors obtained using the "load_exact_map" function.
        intensities_path:
            An array of intensities loaded using the "load_exact_map" function.
        qxy_bins:
            How many qxy steps do you want. Should probably be slightly less
            than the horizontal resolution of your detector.
        qz_bins:
            How many qz steps do you want. Should probably be slightly less than
            the vertical resolution of your detector.
        qz_axis:
            Which axis of the q_vectors array should be taken to be qz? Defaults
            to 2.

    Returns:
        (qxy_qz_intensities, qxy_qz_q_values)
        Numpy arrays where the slow axis (axis=0) is in the qxy direction, and
        the fast axis (axis=1) is in the qz direction.
    """
    # qz is an entirely reasonable name in this context.
    # pylint: disable=invalid-name

    # These arrays could be affected by the call to weighted_bin_1d. Make sure
    # that callers don't have their objects mangled by making copies here.
    q_vectors, intensities = np.copy(q_vectors), np.copy(intensities)

    # Completely ignore all nan values.
    q_vectors = q_vectors[~np.isnan(intensities)]
    intensities = intensities[~np.isnan(intensities)]

    # Deal with variable qz axis.
    i, j, k = (qz_axis+1) % 3, (qz_axis+2) % 3, qz_axis
    # Create an array of qxy, qz q-vectors.
    qxy = np.sqrt(q_vectors[:, i]**2 + q_vectors[:, j]**2)
    qz = q_vectors[:, k]

    # Run a 2D binning (I couldn't be bothered writing a really quick one, so
    # here I'm just defaulting to the reasonable implementation provided by
    # the fast_histogram library).

    # To do this, we need to know min/max of each of qxy and qz.
    min_qxy = np.min(qxy)
    max_qxy = np.max(qxy)
    min_qz = np.min(qz)
    max_qz = np.max(qz)

    # Now run the binning.
    qxy_qz_intensities = fast_histogram.histogram2d(
        x=qxy,
        y=qz,
        bins=(qxy_bins, qz_bins),
        range=[[min_qxy, max_qxy], [min_qz, max_qz]],
        weights=intensities
    )

    # For convenience, also return the corresponding Q values at all pixels.
    binned_qxy = np.linspace(min_qxy, max_qxy, qxy_bins)
    binned_qz = np.linspace(min_qz, max_qz, qz_bins)
    qxy_qz_q_vals = np.meshgrid(binned_qxy, binned_qz, indexing='ij')

    return qxy_qz_intensities, qxy_qz_q_vals


def q_to_theta(q_values: np.ndarray, energy: float) -> np.ndarray:
    """
    Takes a set of q_values IN INVERSE ANGSTROMS and and energy IN ELECTRON
    VOLTS and returns the corresponding theta values (TIMES BY 2 TO GET 2THETA).

    Args:
        q_values:
            The q values in units of inverse angstroms (multiplied by 2pi, for
            those that know the different conventions).
        energy:
            The energy of the incident beam in units of electron volts. Not kev.
            Not joules. Electron volts, please.

    Returns:
        A numpy array of the corresponding theta values (NOT TWO THETA). Just
        multiply by 2 to get two theta.
    """
    # First calculate the wavevector of the incident light.
    planck = physical_constants["Planck constant in eV s"][0]
    speed_of_light = physical_constants["speed of light in vacuum"][0] * 1e10
    # My q_values are angular, so my wavevector needs to be angular too.
    ang_wavevector = 2*np.pi*energy/(planck*speed_of_light)

    # Do some basic geometry.
    theta_values = np.arccos(1 - np.square(q_values)/(2*ang_wavevector**2))

    # Convert from radians to degrees.
    return theta_values*180/np.pi


def get_volume_and_bounds(path_to_npy: str) -> Tuple[np.ndarray]:
    """
    Takes the path to a .npy file. Loads the volume stored in the .npy file, and
    also grabs the definition of the corresponding finite differences volume
    from the bounds file.

    Args:
        path_to_npy:
            Path to the .npy file containing the volume of interest.

    Returns:
        A tuple taking the form (volume, start, stop, step), where each element
        of the tuple is a numpy array of values. The first value returned is the
        full reciprocal space volume, and could be very large (up to ~GB).
    """
    # In case we were given a pathlib.path.
    path_to_npy = str(path_to_npy)
    # Work out the path to the bounds file.
    path_to_bounds = path_to_npy[:-4] + "_bounds.txt"

    # Load the data.
    volume = np.load(path_to_npy)
    q_bounds = np.loadtxt(path_to_bounds)

    # Scrape the start/stop/step values.
    start, stop, step = q_bounds[:, 0], q_bounds[:, 1], q_bounds[:, 2]

    # And return what we were asked for!
    return volume, start, stop, step


def _project_to_1d(volume: np.ndarray,
                   start: np.ndarray,
                   stop: np.ndarray,
                   step: np.ndarray,
                   num_bins: int = 1000,
                   bin_size: float = None,
                   tth=False,
                   only_l=False,
                   energy=None):
    """
    Maps this experiment to a simple intensity vs two-theta representation.
    This virtual 'two-theta' axis is the total angle scattered by the light,
    *not* the projection of the total angle about a particular axis.

    Under the hood, this runs a binned reciprocal space map with an
    auto-generated resolution such that the reciprocal space volume is
    100MB. You really shouldn't need more than 100MB of reciprocal space
    for a simple line chart...

    TODO: one day, this shouldn't run the 3D binned reciprocal space map
    and should instead directly bin to a one-dimensional space. The
    advantage of this approach is that it is much easier to maintain. The
    disadvantage is that theoretical optimal performance is worse, memory
    footprint is higher and some data _could_ be incorrectly binned due to
    double binning. In reality, it's fine: inaccuracies are pedantic and
    performance is pretty damn good.

    TODO: Currently, this method assumes that the beam energy is constant
    throughout the scan. Assumptions aren't exactly in the spirit of this
    module - maybe one day someone can find the time to do this "properly".
    """
    q_x = np.arange(start[0], stop[0], step[0], dtype=np.float32)
    q_y = np.arange(start[1], stop[1], step[1], dtype=np.float32)
    q_z = np.arange(start[2], stop[2], step[2], dtype=np.float32)

    if tth:
        q_x = q_to_theta(q_x, energy)
        q_y = q_to_theta(q_y, energy)
        q_z = q_to_theta(q_z, energy)
    if only_l:
        q_x *= 0
        q_y *= 0

    q_x, q_y, q_z = np.meshgrid(q_x, q_y, q_z, indexing='ij')

    # Now we can work out the |Q| for each voxel.
    q_x_squared = np.square(q_x)
    q_y_squared = np.square(q_y)
    q_z_squared = np.square(q_z)

    q_lengths = np.sqrt(q_x_squared + q_y_squared + q_z_squared)

    # It doesn't make sense to keep the 3D shape because we're about to map
    # to a 1D space anyway.
    rsm = volume.ravel()
    q_lengths = q_lengths.ravel()

    # If we haven't been given a bin_size, we need to calculate it.
    min_q = float(np.nanmin(q_lengths))
    max_q = float(np.nanmax(q_lengths))

    if bin_size is None:
        # Work out the bin_size from the range of |Q| values.
        bin_size = (max_q - min_q)/num_bins

    # Now that we know the bin_size, we can make an array of the bin edges.
    bins = np.arange(min_q, max_q, bin_size)

    # Use this to make output & count arrays of the correct size.
    out = np.zeros_like(bins, dtype=np.float32)
    count = np.zeros_like(out, dtype=np.uint32)

    # Do the binning.
    out = weighted_bin_1d(q_lengths, rsm, out, count,
                          min_q, max_q, bin_size)

    # Normalise by the counts.
    out /= count.astype(np.float32)

    out = np.nan_to_num(out, nan=0)

    # Now return (intensity, Q).
    return out, bins


def intensity_vs_q(output_file_name: str,
                   volume: np.ndarray,
                   start: np.ndarray,
                   stop: np.ndarray,
                   step: np.ndarray,
                   num_bins: int = 1000,
                   bin_size: float = None):
    """
    Calculates intensity as a function of the magnitude of the scattering vector
    from the intensities and their finite differences geometry description.
    """
    intensity, q = _project_to_1d(
        volume, start, stop, step, num_bins, bin_size)

    to_save = np.transpose((q, intensity))
    output_file_name = str(output_file_name) + "_Q.txt"
    print(f"Saving intensity vs q to {output_file_name}")
    np.savetxt(output_file_name, to_save, header="|Q| intensity")
    return q, intensity


def intensity_vs_tth(output_file_name: str,
                     volume: np.ndarray,
                     start: np.ndarray,
                     stop: np.ndarray,
                     step: np.ndarray,
                     energy: float,
                     num_bins: int = 1000,
                     bin_size: float = None):
    """
    Calculates intensity as a function of 2θ, where θ is the total diffracted
    angle (not along any particular direction).
    """
    # This just aliases to self._project_to_1d, which handles the minor
    # difference between a projection to |Q| and 2θ.
    intensity, tth = _project_to_1d(
        volume, start, stop, step, num_bins, bin_size, tth=True, energy=energy)

    to_save = np.transpose((tth, intensity))
    output_file_name = str(output_file_name) + "_tth.txt"
    print(f"Saving intensity vs tth to {output_file_name}")
    np.savetxt(output_file_name, to_save, header="tth intensity")

    return tth, intensity


def intensity_vs_l(output_file_name: str,
                   volume: np.ndarray,
                   start: np.ndarray,
                   stop: np.ndarray,
                   step: np.ndarray,
                   num_bins: int = 1000,
                   bin_size: float = None):
    """
    Calculates intensity as a function of l, ignoring h and k. Only relevant if
    you have used the hkl coordinate system.
    """
    intensity, l = _project_to_1d(
        volume, start, stop, step, num_bins, bin_size, only_l=True)

    to_save = np.transpose((l, intensity))
    output_file_name = str(output_file_name) + "_l.txt"
    print(f"Saving intensity vs l to {output_file_name}")
    np.savetxt(output_file_name, to_save, header="l intensity")


def save_binoculars_hdf5(path_to_npy: np.ndarray, output_path: str):
    """
    Saves the .npy file as a binoculars-readable hdf5 file.
    """
    # Load the volume and the bounds.
    volume, start, stop, step = get_volume_and_bounds(path_to_npy)

    # Binoculars expects float64s with no NaNs.
    volume = volume.astype(np.float64)

    # Allow binoculars to generate the NaNs naturally.
    contributions = np.empty_like(volume)
    contributions[np.isnan(volume)] = 0
    contributions[~np.isnan(volume)] = 1
    volume = np.nan_to_num(volume)

    # Make h, k and l arrays in the expected format.
    h_arr = np.array([0, start[0], stop[0], step[0],
                      start[0]/step[0], stop[0]/step[0]])
    k_arr = np.array([1, start[1], stop[1], step[1],
                      start[1]/step[1], stop[1]/step[1]])
    l_arr = np.array([2, start[2], stop[2], step[2],
                      start[2]/step[2], stop[2]/step[2]])

    # Turn those into an axes group.
    axes_group = nx.NXgroup(h=h_arr, k=k_arr, l=l_arr)
    # Make a corresponding (mandatory) "binoculars" group.
    binoculars_group = nx.NXgroup(
        axes=axes_group, contributions=contributions, counts=(volume))
    binoculars_group.attrs['type'] = 'Space'

    # Make a root which contains the binoculars group.
    bin_hdf = nx.NXroot(binoculars=binoculars_group)

    # Save it!
    bin_hdf.save(output_path)


def most_recent_cluster_output():
    """
    Returns the filename of the most recent cluster stdout output.
    """
    # Get all the cluster job files that have been created.
    files = [x for x in os.listdir() if x.startswith("cluster_job.sh.o")]
    numbers = [int(x[16:]) for x in files]

    # Work out which cluster job is the most recent.
    most_recent_job_no = np.max(numbers)
    most_recent_file = ""
    for file in files:
        if str(most_recent_job_no) in file:
            most_recent_file = file

    return most_recent_file


def most_recent_cluster_error():
    """
    Returns the filename of the most recent cluster stderr output.
    """
    # Get all the cluster job files that have been created.
    files = [x for x in os.listdir() if x.startswith("cluster_job.sh.e")]
    numbers = [int(x[16:]) for x in files]

    # Work out which cluster job is the most recent.
    most_recent_job_no = np.max(numbers)
    most_recent_file = ""
    for file in files:
        if str(most_recent_job_no) in file:
            most_recent_file = file

    return most_recent_file
