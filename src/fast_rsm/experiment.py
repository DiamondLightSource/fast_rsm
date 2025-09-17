"""
This file contains the Experiment class, which contains all of the information
relating to your experiment.
"""
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Pool, Lock
from ast import literal_eval

import logging
import os
# from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from time import time
from typing import List, Tuple, Union
import numpy as np
from diffraction_utils import Frame, Region
from scipy.constants import physical_constants

from scipy.spatial.transform import Rotation as R
import transformations as tf

import pandas as pd
import fabio

# from datetime import datetime
import h5py
import tifffile
import fast_rsm.io as io
from fast_rsm.binning import finite_diff_shape
from fast_rsm.meta_analysis import get_step_from_filesize
from fast_rsm.scan import Scan, chunk, \
    pyfai_stat_qmap, pyfai_stat_ivsq, pyfai_stat_exitangles, \
    pyfai_init_worker, pyfai_move_qmap_worker, rsm_init_worker, bin_maps_with_indices_smm, \
    pyfai_move_ivsq_worker, pyfai_move_exitangles_worker
from fast_rsm.writing import linear_bin_to_vtk

from fast_rsm.logging_config import get_frsm_logger
logger=get_frsm_logger(__name__)

# from memory_profiler import profile


def combine_ranges(range1, range2):
    """
    combines two ranges to give the widest possible range
    """
    return (min(range1[0], range2[0]), max(range1[1], range2[1]))


def _remove_file(path: Union[str, Path]):
    """
    Removes a file if it exists. Doesn't do anything if it doesn't exist.
    """
    try:
        os.remove(path)
    except OSError:
        pass


def _sum_numpy_files(filenames: List[Union[Path, str]]):
    """
    Takes a list of paths to .npy files. Adds them all together. Returns this
    sum.

    TODO: this could be parallelised.

    Args:
        filenames:
            A list/tuple/iterable of strings/paths/np.load-able things.

    Returns:
        Sum of the contents of all the files in the filenames list.
    """
    total = np.load(filenames[0] + '.npy')

    for i, filename in enumerate(filenames):
        # Skip the zeroth file because we already loaded it.
        if i == 0:
            continue
        total += np.load(filename + '.npy')

    return total


def _q_to_theta(q_values, energy) -> np.array:
    """
    Calculates the diffractometer theta from scattering vector Q.

    Args:
        theta:
            Array of theta values to be converted.
        energy:
            Energy of the incident probe particle, in eV.
    """
    # First calculate the wavevector of the incident light.
    planck = physical_constants["Planck constant in eV s"][0]
    speed_of_light = physical_constants["speed of light in vacuum"][0] * 1e10
    # My q_values are angular, so my wavevector needs to be angular too.
    ang_wavevector = 2 * np.pi * energy / (planck * speed_of_light)

    # Do some basic geometry.
    theta_values = np.arccos(1 - np.square(q_values) / (2 * ang_wavevector**2))

    # Convert from radians to degrees.
    return theta_values * 180 / np.pi


class Experiment:
    """
    The Experiment class specifies all experimental details necessary to process
    experimental data. This can be a list of file paths, a central pixel
    location, details of the experimental geometry, the desired output, etc.

    Experiments contain lot of tools that can be used to work out what they are.
    For example, experiments can work out whether or not binning is required to
    reduce data volumes, and if so, generate finite difference volumes in which
    the data can be optimally stored.

    Attrs:
        scans:
            A list of all of the scans explored in this experiment.
    """

    def __init__(self, scans: List[Scan], setup: str) -> None:
        self.scans = scans
        self.setup = setup
        self._data_file_names = []
        self._normalisation_file_names = []
        self.spherical_bragg_vec= np.array([0, 0, 0])
        self.savedats= False
        self.savetiffs= False
        self.alphacritical= 0

        none_exp = ['pixel_size', 'entry', 'detector_distance',
                    'incident_wavelength', 'gammadata', 'deltadata', 'dcdrad']
        for val in none_exp:  # pylint: disable=attribute-defined-outside-init
            setattr(self, val, None)

    def _clean_temp_files(self) -> None:
        """
        Removes all temporary files.
        """
        for path in self._data_file_names:
            _remove_file(path)
        for path in self._normalisation_file_names:
            _remove_file(path)

        self._data_file_names = []
        self._normalisation_file_names = []

    def add_processing_step(self, processing_step: callable) -> None:
        """
        Adds a processing step to every scan in the experiment. The processing
        step is a function that takes a numpy array representing the input
        image, and outputs a numpy array representing the output image. A valid
        processing step could look something like

        def add_one(arr):
            return arr + 1

        If you, for some reason, wanted to add 1 to every raw image array, this
        function can be added to all the scans in this experiment using:

            experiment.add_processing_step(add_one)

        Note that this must currently (python 3.10) be a module-level named
        function. Because of the internals of how pythons serialization
        (pickling) works, anonymous (lambda) functions won't work.

        *Processing steps are applied in the order in which they are added.*
        """
        for scan in self.scans:
            scan.add_processing_step(processing_step)

    def mask_pixels(self, pixels: tuple) -> None:
        """
        Masks the requested pixels. Note that the pixels argument will be used
        to directly affect arrays, i.e.

        array[pixels] = np.nan

        is going to be used at somencounterede point.

        Args:
            pixels:
                The pixels to be masked.
        """
        # Masking is handled specially using RSMMetadata objects. I couldn't
        # think of a more elegant solution, and I don't love that this
        # masking information is somewhat randomly stored there, but I suppose
        # that pixel masks are metadata, and it is relevant to reciprocal space
        # maps, so I suppose it works well enough!
        for scan in self.scans:
            scan.metadata.mask_pixels = pixels

    def mask_regions(self, regions: List[Region]):
        """
        Masks the requested regions defined as regions in setup file.
        """
        # Make sure that we have a list of regions, not an individual region.
        if isinstance(regions, Region):
            regions = [regions]

        if self.scans[0].metadata.data_file.is_rotated is True:
            imshape = self.scans[0].metadata.data_file.image_shape
            for region in regions:
                newxend = imshape[0] - region.y_start
                newxstart = max(0, imshape[0] - region.y_end)
                newystart = region.x_start
                newyend = region.x_end
                region.x_start = newxstart
                region.x_end = newxend
                region.y_start = newystart
                region.y_end = newyend

        for scan in self.scans:
            scan.metadata.mask_regions = regions

    def mask_edf(self, edfmask):
        """
        apply any mask defined with a .edf file as created in pyFAI-calib GUI
        """

        if edfmask is not None:
            maskimg = fabio.open(edfmask)
            mask = maskimg.data
            if self.scans[0].metadata.data_file.is_rotated is True:
                mask = np.rot90(np.flip(mask, axis=0), 1)
        else:
            mask = None

        for scan in self.scans:
            scan.metadata.edfmask = mask

    # def binned_reciprocal_space_map(self,
    #                                 num_threads: int,
    #                                 map_frame: Frame,
    #                                 output_file_name: str = "mapped",
    #                                 min_intensity_mask: float = None,
    #                                 output_file_size: float = 100,
    #                                 save_vtk: bool = True,
    #                                 save_npy: bool = True,
    #                                 oop: str = 'y',
    #                                 volume_start: np.ndarray = None,
    #                                 volume_stop: np.ndarray = None,
    #                                 volume_step: np.ndarray = None,
    #                                 map_each_image: bool = False):
    #     """
    #     Carries out a binned reciprocal space map for this experimental data.

    #     Args:
    #         num_threads:
    #             How many threads should be used to carry out this map? You
    #             should probably set this to however many threads are available
    #             on your machine.
    #         map_frame:
    #             An instance of diffraction_utils.Frame that specifies what
    #             frame of reference and coordinate system you'd like your map to
    #             be carried out in.
    #         output_file_name:
    #             What should the final file be saved as?
    #         output_file_size:
    #             The desired output file size, in units of MB.
    #             Defaults to 100 MB.
    #         save_vtk:
    #             Should we save a vtk file for this map? Defaults to True.
    #         save_npy:
    #             Should we save a numpy file for this map? Defaults to True.
    #         oop:
    #             Which synchrotron axis should become the out-of-plane (001)
    #             direction. Defaults to 'y'; can be 'x', 'y' or 'z'.
    #     """
    #     # For simplicity, if qpar_qperp is asked for, we swap to the lab frame.
    #     # They're the same, but qpar_qperp is an average.
    #     #original_frame_name = map_frame.frame_name
    #     if map_frame.frame_name == Frame.qpar_qperp:
    #         map_frame.frame_name = Frame.lab
    #       # Compute the optimal finite differences volume.

    #     if volume_step is None:
    #         # Overwrite whichever of these we were given explicitly.
    #         if (volume_start is not None) & (volume_stop is not None):
    #             _start = np.array(volume_start)
    #             _stop = np.array(volume_stop)
    #         else:
    #             _start, _stop = self.q_bounds(
    #                 map_frame, oop)
    #         step = get_step_from_filesize(_start, _stop, output_file_size)

    #     else:
    #         step = np.array(volume_step)
    #         _start, _stop = self.q_bounds(map_frame, oop)

    #     if map_frame.coordinates == Frame.sphericalpolar:
    #         step = (0.02, np.pi / 180, np.pi / 180)
    #     # Make sure start and stop match the step as required by binoculars.
    #     start, stop = _match_start_stop_to_step(
    #         step=step,
    #         user_bounds=(volume_start, volume_stop),
    #         auto_bounds=(_start, _stop))

    #     locks = [Lock() for _ in range(num_threads)]
    #     shape = finite_diff_shape(start, stop, step)

    #     time_1 = time()
    #     # map_mem_total=[]
    #     # count_mem_total=[]
    #     map_arrays = 0
    #     count_arrays = 0
    #     images_so_far = 0

    #     for scan in self.scans:
    #         async_results = []
    #         # Make a pool on which we'll carry out the processing.
    #         with Pool(
    #             processes=num_threads,  # The size of our pool.
    #             initializer=init_process_pool,  # Our pool's initializer.
    #             initargs=(locks,  # The initializer makes this lock global.
    #                       num_threads,  # Initializer makes num_threads global.
    #                       self.scans[0].metadata,
    #                       map_frame,
    #                       shape,
    #                       output_file_name)
    #         ) as pool:

    #             for indices in chunk(list(range(
    #                     scan.metadata.data_file.scan_length)), num_threads):

    #                 new_motors = scan.metadata.data_file.get_motors()
    #                 new_metadata = scan.metadata.data_file.get_metadata()

    #                 # Submit the binning as jobs on the pool.
    #                 # Note that serializing the map_frame and the scan.metadata
    #                 # are the only things that take finite time.
    #                 async_results.append(pool.apply_async(
    #                     bin_maps_with_indices,
    #                     (indices, start, stop, step,
    #                      min_intensity_mask, new_motors, new_metadata,
    #                      scan.processing_steps, scan.skip_images, oop,
    #                      map_each_image, images_so_far)))

    #                 images_so_far += scan.metadata.data_file.scan_length

    #             print(f"Took {time() - time_1}s to prepare the calculation.")
    #             map_names = []
    #             count_names = []
    #             map_mem = []
    #             count_mem = []
    #             for result in async_results:
    #                 # Make sure that we're storing the location of the shared memory
    #                 # block.
    #                 shared_rsm_name, shared_count_name = result.get()
    #                 if shared_rsm_name not in map_names:
    #                     map_names.append(shared_rsm_name)
    #                 if shared_count_name not in count_names:
    #                     count_names.append(shared_count_name)

    #                 # Make sure that no error was thrown while mapping.
    #                 if not result.successful():
    #                     raise ValueError(
    #                         "Could not carry out map for an unknown reason. "
    #                         "Probably one of the threads segfaulted, or something.")
    #             scanname = scan.metadata.data_file.diamond_scan.nxfilename.split(
    #                 '/')[-1]
    #             print(f"\nCalculation for scan {scanname} complete.")
    #             map_mem = [SharedMemory(x) for x in map_names]
    #             count_mem = [SharedMemory(x) for x in count_names]

    #             new_map_arrays = np.array([
    #                 np.ndarray(shape=shape, dtype=np.float32, buffer=x.buf)
    #                 for x in map_mem])
    #             new_count_arrays = np.array([
    #                 np.ndarray(shape=shape, dtype=np.uint32, buffer=y.buf)
    #                 for y in count_mem])

    #             # map_mem_total+=map_mem
    #             # count_mem_total+=(count_mem)
    #             if np.size(map_arrays) == 1:
    #                 map_arrays = np.sum(new_map_arrays, axis=0)
    #                 count_arrays = np.sum(new_count_arrays, axis=0)
    #             else:
    #                 new_maps = np.sum(new_map_arrays, axis=0)
    #                 new_counts = np.sum(new_count_arrays, axis=0)
    #                 map_arrays = np.sum([map_arrays, new_maps], axis=0)
    #                 count_arrays = np.sum([count_arrays, new_counts], axis=0)
    #             #           normalised_map = map_arrays/(count_arrays.astype(np.float32))
    #             # Make sure all our shared memory has been closed nicely.
    #         for shared_mem in map_mem:
    #             shared_mem.close()
    #             try:
    #                 shared_mem.unlink()
    #             except BaseException:
    #                 pass
    #         for shared_mem in count_mem:
    #             shared_mem.close()
    #             try:
    #                 shared_mem.unlink()
    #             except BaseException:
    #                 pass

    #     # makes sure counts are floats ready for division
    #     fcounts = count_arrays.astype(np.float32)
    #     # need to specify out location to avoid working with non-initialised
    #     # data
    #     normalised_map = np.divide(
    #         map_arrays,
    #         fcounts,
    #         out=np.copy(map_arrays),
    #         where=fcounts != 0.0)

    #     # Only save the vtk/npy files if we've been asked to.
    #     if save_vtk:
    #         print("\n**READ THIS**")
    #         print(f"Saving vtk to {output_file_name}.vtk")
    #         print(
    #             "This is the file that you can open with paraview. "
    #             "Note that this can be done with 'module load paraview' "
    #             "followed by 'paraview' in the terminal. \n"
    #             "Then open with ctrl+o as usual, navigating to the above path. "
    #             "Once opened, apply a threshold filter to your data to view. \n"
    #             "To play with colours: on your threshold filter, go to the "
    #             "'coloring' section and click the small 'rescale to custom "
    #             "range' button. Paraview is really powerful, but this should "
    #             "be enough to at least get you playing with it.\n")
    #         linear_bin_to_vtk(
    #             normalised_map, output_file_name, start, stop, step)
    #     if save_npy:
    #         print(f"Saving numpy array to {output_file_name}.npy")
    #         np.save(output_file_name, normalised_map)
    #         # Also save the finite differences parameters.
    #         np.savetxt(str(output_file_name) + "_bounds.txt",
    #                    np.array((start, stop, step)).transpose(),
    #                    header="start stop step")

    #     # Finally, remove any .npy files we created along the way.
    #     self._clean_temp_files()

    #     # Return the normalised RSM.
    #     return normalised_map, start, stop, step

    # def _project_to_1d(self,
    #                    num_threads: int,
    #                    output_file_name: str = "mapped",
    #                    num_bins: int = 1000,
    #                    bin_size: float = None,
    #                    oop: str = 'y',
    #                    tth=False,
    #                    only_l=False):
    #     """
    #     Maps this experiment to a simple intensity vs two-theta representation.
    #     This virtual 'two-theta' axis is the total angle scattered by the light,
    #     *not* the projection of the total angle about a particular axis.

    #     Under the hood, this runs a binned reciprocal space map with an
    #     auto-generated resolution such that the reciprocal space volume is
    #     100MB. You really shouldn't need more than 100MB of reciprocal space
    #     for a simple line chart...

    #     TO DO: one day, this shouldn't run the 3D binned reciprocal space map
    #     and should instead directly bin to a one-dimensional space. The
    #     advantage of this approach is that it is much easier to maintain. The
    #     disadvantage is that theoretical optimal performance is worse, memory
    #     footprint is higher and some data _could_ be incorrectly binned due to
    #     double binning. In reality, it's fine: inaccuracies are pedantic and
    #     performance is pretty damn good.

    #     TO DO: Currently, this method assumes that the beam energy is constant
    #     throughout the scan. Assumptions aren't exactly in the spirit of this
    #     module - maybe one day someone can find the time to do this "properly".
    #     """
    #     map_frame = Frame(Frame.sample_holder, coordinates=Frame.cartesian)
    #     if only_l:
    #         map_frame = Frame(Frame.hkl, coordinates=Frame.cartesian)

    #     rsm, start, stop, step = self.binned_reciprocal_space_map(
    #         num_threads, map_frame, output_file_name, save_vtk=False,
    #         oop=oop)

    #     # RSM could have a lot of nan elements, representing unmeasured voxels.
    #     # Lets make sure we zero these before continuing.
    #     rsm = np.nan_to_num(rsm, nan=0)

    #     q_x = np.arange(start[0], stop[0], step[0], dtype=np.float32)
    #     q_y = np.arange(start[1], stop[1], step[1], dtype=np.float32)
    #     q_z = np.arange(start[2], stop[2], step[2], dtype=np.float32)

    #     if tth:
    #         energy = self.scans[0].metadata.data_file.probe_energy
    #         q_x = _q_to_theta(q_x, energy)
    #         q_y = _q_to_theta(q_y, energy)
    #         q_z = _q_to_theta(q_z, energy)
    #     if only_l:
    #         q_x *= 0
    #         q_y *= 0

    #     q_x, q_y, q_z = np.meshgrid(q_x, q_y, q_z, indexing='ij')

    #     # Now we can work out the |Q| for each voxel.
    #     q_x_squared = np.square(q_x)
    #     q_y_squared = np.square(q_y)
    #     q_z_squared = np.square(q_z)

    #     q_lengths = np.sqrt(q_x_squared + q_y_squared + q_z_squared)

    #     # It doesn't make sense to keep the 3D shape because we're about to map
    #     # to a 1D space anyway.
    #     rsm = rsm.ravel()
    #     q_lengths = q_lengths.ravel()

    #     # If we haven't been given a bin_size, we need to calculate it.
    #     min_q = float(np.min(q_lengths))
    #     max_q = float(np.max(q_lengths))

    #     if bin_size is None:
    #         # Work out the bin_size from the range of |Q| values.
    #         bin_size = (max_q - min_q) / num_bins

    #     # Now that we know the bin_size, we can make an array of the bin edges.
    #     bins = np.arange(min_q, max_q, bin_size)

    #     # Use this to make output & count arrays of the correct size.
    #     out = np.zeros_like(bins, dtype=np.float32)
    #     count = np.zeros_like(out, dtype=np.uint32)

    #     # Do the binning.
    #     out = weighted_bin_1d(q_lengths, rsm, out, count,
    #                           min_q, max_q, bin_size)

    #     # Normalise by the counts.
    #     out /= count.astype(np.float32)

    #     out = np.nan_to_num(out, nan=0)

    #     # Now return (intensity, Q).
    #     return out, bins

    # def intensity_vs_l(self,
    #                    num_threads: int,
    #                    output_file_name: str = "mapped",
    #                    num_bins: int = 1000,
    #                    bin_size: float = None,
    #                    oop: str = 'y'):
    #     """
    #     Maps this experiment to a simple intensity vs l representation.

    #     Under the hood, this runs a binned reciprocal space map with an
    #     auto-generated resolution such that the reciprocal space volume is
    #     100MB. You really shouldn't need more than 100MB of reciprocal space
    #     for a simple line chart...

    #     TO DO: one day, this shouldn't run the 3D binned reciprocal space map
    #     and should instead directly bin to a one-dimensional space. The
    #     advantage of this approach is that it is much easier to maintain. The
    #     disadvantage is that theoretical optimal performance is worse, memory
    #     footprint is higher and some data _could_ be incorrectly binned due to
    #     double binning. In reality, it's fine: inaccuracies are pedantic and
    #     performance is pretty damn good.

    #     TO DO: Currently, this method assumes that the beam energy is constant
    #     throughout the scan. Assumptions aren't exactly in the spirit of this
    #     module - maybe one day someone can find the time to do this "properly".
    #     """
    #     intensity, l = self._project_to_1d(
    #         num_threads, output_file_name, num_bins, bin_size, only_l=True,
    #         oop=oop)

    #     to_save = np.transpose((l, intensity))
    #     output_file_name = str(output_file_name) + "_l.txt"
    #     print(f"Saving intensity vs l to {output_file_name}")
    #     np.savetxt(output_file_name, to_save, header="l intensity")

    #     return l, intensity

    # def intensity_vs_tth(self,
    #                      num_threads: int,
    #                      output_file_name: str = "mapped",
    #                      num_bins: int = 1000,
    #                      bin_size: float = None,
    #                      oop: str = 'y'):
    #     """
    #     Maps this experiment to a simple intensity vs two-theta representation.
    #     This virtual 'two-theta' axis is the total angle scattered by the light,
    #     *not* the projection of the total angle about a particular axis.

    #     Under the hood, this runs a binned reciprocal space map with an
    #     auto-generated resolution such that the reciprocal space volume is
    #     100MB. You really shouldn't need more than 100MB of reciprocal space
    #     for a simple line chart...

    #     TO DO: one day, this shouldn't run the 3D binned reciprocal space map
    #     and should instead directly bin to a one-dimensional space. The
    #     advantage of this approach is that it is much easier to maintain. The
    #     disadvantage is that theoretical optimal performance is worse, memory
    #     footprint is higher and some data _could_ be incorrectly binned due to
    #     double binning. In reality, it's fine: inaccuracies are pedantic and
    #     performance is pretty damn good.

    #     TO DO: Currently, this method assumes that the beam energy is constant
    #     throughout the scan. Assumptions aren't exactly in the spirit of this
    #     module - maybe one day someone can find the time to do this "properly".
    #     """
    #     # This just aliases to self._project_to_1d, which handles the minor
    #     # difference between a projection to |Q| and 2θ.
    #     intensity, tth = self._project_to_1d(
    #         num_threads, output_file_name, num_bins, bin_size, tth=True,
    #         oop=oop)

    #     to_save = np.transpose((tth, intensity))
    #     output_file_name = str(output_file_name) + "_tth.txt"
    #     print(f"Saving intensity vs tth to {output_file_name}")
    #     np.savetxt(output_file_name, to_save, header="tth intensity")

    #     return tth, intensity

    # def intensity_vs_q(self,
    #                    num_threads: int,
    #                    output_file_name: str = "I_vs_Q",
    #                    num_bins: int = 1000,
    #                    bin_size: float = None,
    #                    oop: str = 'y'):
    #     """
    #     Maps this experiment to a simple insenity vs `|Q|` plot.

    #     Under the hood, this runs a binned reciprocal space map with an
    #     auto-generated resolution such that the reciprocal space volume is
    #     100MB. You really shouldn't need more than 100MB of reciprocal space
    #     for a simple line chart...

    #     TO DO: one day, this shouldn't run the 3D binned reciprocal space map
    #     and should instead directly bin to a one-dimensional space. The
    #     advantage of this approach is that it is much easier to maintain. The
    #     disadvantage is that theoretical optimal performance is worse, memory
    #     footprint is higher and some data could be incorrectly binned due to
    #     double binning. In reality, it's fine: inaccuracies are pedantic and
    #     performance is pretty damn good.
    #     """
    #     # This just aliases to self._project_to_1d, which handles the minor
    #     # difference between a projection to |Q| and 2θ.
    #     intensity, q = self._project_to_1d(
    #         num_threads, output_file_name, num_bins, bin_size, oop=oop)

    #     to_save = np.transpose((q, intensity))
    #     output_file_name = str(output_file_name) + "_Q.txt"
    #     print(f"Saving intensity vs q to {output_file_name}")
    #     np.savetxt(output_file_name, to_save, header="|Q| intensity")
    #     return q, intensity

    def q_bounds(self, frame: Frame, oop: str = 'y') -> Tuple[np.ndarray]:
        """
        Works out the region of reciprocal space sampled by every scan in this
        experiment. This is reasonably performant, but should really be
        parallelised at some point.

        Returns:
            (start, stop)
        """
        # Get a start and stop value for each scan.
        starts, stops = [], []
        for scan in self.scans:
            start, stop = scan.q_bounds(
                frame, spherical_bragg_vec=self.spherical_bragg_vec, oop=oop)
            starts.append(start)
            stops.append(stop)

        # Return the min of the starts and the max of the stops.
        starts, stops = np.array(starts), np.array(stops)
        return np.min(starts, axis=0), np.max(stops, axis=0)

    def calctheta(self, q, wavelength):
        """
        converts two theta value to q value for a given wavelength

        Parameters
        ----------
        float
            q value in m^-1 to be converted to angle in degrees
        wavelength : float
            value of wavelength for incident radiation in angstrom.

        Returns
        -------

        twotheta : float
            value of angle in degrees converted to  q.
        """
        return np.degrees(np.arcsin(q * (wavelength / (4 * np.pi)) * 1e10)) * 2

    def calcq(self, twotheta, wavelength):
        """
        converts two theta value to q value for a given wavelength

        Parameters
        ----------
        twotheta : float
            value of angle in degrees to be converted to  q.
        wavelength : float
            value of wavelength for incident radiation in angstrom.

        Returns
        -------
        float
            q value in m^-1.

        """
        return (4 * np.pi / wavelength) * \
            np.sin(np.radians(twotheta / 2)) * 1e-10

    def calcqstep(self, gammastep, gammastart, wavelength):
        """
        calculates the equivalent q-step for a given 2theta step
        """
        qstep = self.calcq(gammastart + gammastep, wavelength) - \
            self.calcq(gammastart, wavelength)
        return qstep

    def histogram_xy(self, x, y, step_size):
        """


        Parameters
        ----------
        x : array
            x dataset.
        y : array
            y dataset.
        step_size : float
            desired stepsize.

        Returns
        -------
        bin_centers : array
            value of centres for bin values.
        hist : array
            historgam output.
        hist_normalized : array
            normalised histogram output.
        counts : array
            counts for each bin position.

        """

        # Calculate the bin edges based on the step size
        bin_edges = np.arange(np.min(x), np.max(x) + step_size, step_size)

        # Use numpy's histogram function to bin the data
        hist, _ = np.histogram(x, bins=bin_edges, weights=y)

        # Count the number of contributions to each bin
        counts, _ = np.histogram(x, bins=bin_edges)

        # Normalize the histogram by dividing each bin by its number of contributions
        # Use np.where to avoid division by zero
        hist_normalized = np.where(counts > 0, hist / counts, 0)

        # Calculate the bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, hist, hist_normalized, counts

    def sohqcalc(self, angle, kmod):
        """
        calculates q value based on a opposite and hypotenuse of a
        right-angled triangle

        Parameters
        ----------
        angle : float
            angle of trig triangle being analysed.
        kmod : float
            wavevector value.

        Returns
        -------
        float
            q value.

        """
        return np.sin(np.radians(angle)) * kmod * 1e-10

    def get_limitcalc_vars(self, vertsetup, axis, slitvertratio, slithorratio):
        '''
        Calculates the pixel limits of the scan in either vertical or horizontal direction
        and returns dictionary with values relating to the limits calculated

        Parameter
        --------
        vertsetup

        axis

        slitvertratio

        slithorration

        Returns
        -------

        dict containing ['horindex', 'vertindex', 'vertangles', 'horangles',\
              'verscale', 'horscale', 'pixhigh','pixlow','outscale',\
                'pixscale', 'highsign', 'lowsign', 'highsection', 'lowsection'

        '''
        horvert_indices = {'vert0': [1, 0], 'hor0': [0, 1]}
        horvert_angles = {
            'thvert': [
                self.two_theta_start,
                self.deltadata],
            'delvert': [
                self.deltadata,
                self.two_theta_start]}
        if (vertsetup is True) & (self.scans[0].metadata.data_file.is_rotated):
            # GOOD
            [horindex, vertindex] = horvert_indices['hor0']
            [vertangles, horangles] = horvert_angles['thvert']
            verscale = -1
            horscale = 1

        elif vertsetup is True:
            # GOOD
            [horindex, vertindex] = horvert_indices['hor0']
            [vertangles, horangles] = horvert_angles['thvert']
            verscale = -1
            horscale = -1

        elif (self.setup == 'DCD') & (self.scans[0].metadata.data_file.is_rotated):
            [horindex, vertindex] = horvert_indices['vert0']
            [vertangles, horangles] = horvert_angles['delvert']
            verscale = -1
            horscale = -1

        elif self.setup == 'DCD':
            [horindex, vertindex] = horvert_indices['vert0']
            [vertangles, horangles] = horvert_angles['delvert']
            verscale = -1
            horscale = -1

        elif (vertsetup is False) & (self.scans[0].metadata.data_file.is_rotated):
            [horindex, vertindex] = horvert_indices['vert0']
            [vertangles, horangles] = horvert_angles['delvert']
            verscale = -1
            horscale = 1

        else:
            # GOOD
            [horindex, vertindex] = horvert_indices['vert0']
            [vertangles, horangles] = horvert_angles['delvert']
            verscale = -1
            horscale = -1
        # pylint: disable=unused-variable
        outscale = 1
        if axis == 'vert':
            pixlow = self.imshape[vertindex] - self.beam_centre[vertindex]
            pixhigh = self.beam_centre[vertindex]
            highsection = np.max(vertangles)
            lowsection = np.min(vertangles)
            outscale = verscale
        elif axis == 'hor':
            pixhigh = self.beam_centre[horindex]
            pixlow = self.imshape[horindex] - self.beam_centre[horindex]
            if (self.setup == 'vertical') & (
                    self.scans[0].metadata.data_file.is_rotated):
                pixhigh, pixlow = pixlow, pixhigh
            highsection = np.max(horangles)
            lowsection = np.min(horangles)
            outscale = horscale
        if (slitvertratio is not None) & (axis == 'vert'):
            pixscale = self.pixel_size * slitvertratio
        elif (slithorratio is not None) & (axis == 'hor'):
            pixscale = self.pixel_size * slithorratio
        else:
            pixscale = self.pixel_size
        [highsign, lowsign] = [1 if np.round(val, 5) == 0 else np.sign(
            val) for val in [highsection, lowsection]]
        outlist = ['horindex', 'vertindex', 'vertangles', 'horangles',\
                   'verscale', 'horscale', 'pixhigh', 'pixlow', 'outscale',\
                    'pixscale', 'highsign', 'lowsign', 'highsection', 'lowsection']
        outdict = {}
        for name in outlist:
            outdict[name] = locals().get(name, None)
        logger.debug(f"get_limitcalc_vars give outdict={outdict}")
        return outdict
    # horindex, vertindex, vertangles, horangles, verscale, horscale, pixhigh, \
   # pixlow,outscale,pixscale, highsign, lowsign, highsection, lowsection #

    def calcanglim(self, axis, vertsetup=False,
                   slitvertratio=None, slithorratio=None):
        """
        Calculates limits in exit angle for either vertical or horizontal axis

        Parameters
        ----------
        axis : string
            choose axis to calculate q limits for .
        vertsetup : TYPE, optional
            is the experimental setup horizontal. The default is False.

        Returns
        -------
        maxangle : float
            upper limit on exit angle in degrees.
        minangle : float
            lower limit on exit angle in degrees.

        """

        # horindex, vertindex, vertangles, horangles, verscale, horscale, pixhigh, \
        #      pixlow, outscale,pixscale, highsign, lowsign, highsection, lowsection
        limitdict = self.get_limitcalc_vars(
            vertsetup, axis, slitvertratio, slithorratio)
        limitkeys = [
            'pixhigh',
            'pixscale',
            'pixlow',
            'vertangles',
            'highsection',
            'lowsection',
            'outscale']
        pixhigh, pixscale, pixlow, vertangles, highsection, lowsection, outscale = [
            limitdict.get(key) for key in limitkeys]

        add_section = (
            np.degrees(
                np.arctan(
                    (pixhigh * pixscale) / self.detector_distance)))
        minus_section = (
            np.degrees(
                np.arctan(
                    (pixlow * pixscale) / self.detector_distance)))
        maxvertrad = np.radians(np.max(vertangles))
        if axis == 'hor':
            add_section = np.degrees(
                np.arctan(np.tan(np.radians(add_section)) / abs(np.cos(maxvertrad))))
            minus_section = np.degrees(
                np.arctan(np.tan(np.radians(minus_section)) / abs(np.cos(maxvertrad))))
        maxangle = highsection + (add_section)
        minangle = lowsection - (minus_section)

        # add incorrection factors to match direction with pyfai calculations
        if (vertsetup is True) & (self.scans[0].metadata.data_file.is_rotated):
            correctionscales = {'vert': 1, 'hor': -1}
        elif vertsetup is True:
            correctionscales = {'vert': 1, 'hor': -1}
        elif (self.setup == 'DCD') & (self.scans[0].metadata.data_file.is_rotated):
            correctionscales = {'vert': 1, 'hor': 1}
        elif self.setup == 'DCD':
            correctionscales = {'vert': 1, 'hor': 1}
        elif (vertsetup is False) & (self.scans[0].metadata.data_file.is_rotated):
            correctionscales = {'vert': 1, 'hor': 1}
        else:
            correctionscales = {'vert': 1, 'hor': 1}

        outscale *= correctionscales[axis]
        outvals = np.sort([minangle * outscale, maxangle * outscale])
        return outvals[0], outvals[1]
        # return maxangle*outscale,minangle*outscale

    def calcqlim(self, axis, vertsetup=False,
                 slitvertratio=None, slithorratio=None):
        """
        Calculates limits in q for either vertical or horizontal axis

        Parameters
        ----------
        axis : string
            choose axis to calculate q limits for .
        vertsetup : TYPE, optional
            is the experimental setup horizontal. The default is False.

        Returns
        -------
        qupp : float
            upper limit on q range.
        qlow : float
            lower limit on q range.

        """
        kmod = 2 * np.pi / (self.incident_wavelength)

        # horindex, vertindex, vertangles, horangles, verscale, horscale, pixhigh, \
        #     pixlow, outscale, pixscale, highsign, lowsign, highsection, lowsection = \
        # self.get_limitcalc_vars(
        #         vertsetup, axis, slitvertratio, slithorratio)

        limitdict = self.get_limitcalc_vars(
            vertsetup, axis, slitvertratio, slithorratio)
        limitkeys = ['vertindex', 'vertangles', 'horangles', 'pixhigh',
                     'pixlow', 'outscale', 'pixscale', 'highsection', 'lowsection']
        vertindex, vertangles, horangles, pixhigh, pixlow, \
            outscale, pixscale, highsection, lowsection = [
                limitdict.get(key) for key in limitkeys]
        maxangle = highsection + \
            (np.degrees(np.arctan((pixhigh * pixscale) / self.detector_distance)))
        minangle = lowsection - \
            (np.degrees(np.arctan((pixlow * pixscale) / self.detector_distance)))
        maxanglerad = np.radians(np.max(maxangle))
        minanglerad = np.radians(np.max(minangle))

        if axis == 'vert':
            qupp = self.sohqcalc(maxangle, kmod)  # *2
            qlow = self.sohqcalc(minangle, kmod)  # *2
            maxtthrad = np.radians(np.max(horangles))

            maxincrad = np.radians(np.max(self.incident_angle))
            extraincq = kmod * 1e-10 * np.sin(maxincrad)

            minusexitq_x = kmod * 1e-10 * \
                np.cos(maxanglerad) * np.cos(maxtthrad) * np.sin(maxincrad)
            minusexitq_z = kmod * 1e-10 * \
                np.sin(maxanglerad) * (1 - np.cos(maxincrad))
            extravert = extraincq - minusexitq_x - minusexitq_z
            qupp += extravert

            minusexitq_x = kmod * 1e-10 * \
                np.cos(minanglerad) * np.cos(maxtthrad) * np.sin(maxincrad)
            minusexitq_z = kmod * 1e-10 * \
                np.sin(minanglerad) * (1 - np.cos(maxincrad))
            extravert = extraincq - minusexitq_x - minusexitq_z
            qlow += extravert

        elif axis == 'hor':
            qupp = self.sohqcalc(maxangle, kmod)  # *2
            qlow = self.sohqcalc(minangle, kmod)  # *2
            vertsign = [1 if np.sign(np.max(vertangles)) >= 0 else -1]
            maxvert = np.max(vertangles) + vertsign[0] * np.degrees(np.arctan(
                (self.beam_centre[vertindex] * self.pixel_size) / self.detector_distance))
            maxvertrad = np.radians(maxvert)
            s1 = kmod * np.cos(maxvertrad) * np.sin(maxanglerad)
            s2 = kmod * (1 - np.cos(maxvertrad) * np.cos(maxanglerad))
            qupp_withvert = np.sqrt(
                np.square(s1) + np.square(s2)) * 1e-10 * np.sign(maxangle)
            s3 = kmod * np.cos(maxvertrad) * np.sin(minanglerad)
            s4 = kmod * (1 - np.cos(maxvertrad) * np.cos(minanglerad))
            qlow_withvert = np.sqrt(
                np.square(s3) + np.square(s4)) * 1e-10 * np.sign(minangle)
            if vertsetup is True:
                outscale *= -1
            if abs(qupp_withvert) > abs(qupp):
                qupp = qupp_withvert

            if abs(qlow_withvert) > abs(qlow):
                qlow = qlow_withvert

        outvals = np.sort([qupp * outscale, qlow * outscale])
        return outvals[0], outvals[1]

    def do_savetiffs(self, hf, data, axespara, axesperp):
        """
        save separate tiffs for all 2d image data in data
        """
        datashape = np.shape(data)
        extradims = len(datashape) - 2
        outdir = hf.filename.strip('.hdf5')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outname = outdir.split('\\')[-1]
        if extradims == 0:
            imdata = data
            parainfo = axespara
            perpinfo = axesperp
            metadata = {
                'Description': f'Image data identical to data saved in {hf.filename}',
                'Xlimits': f'min {parainfo.min()}, max {parainfo.max()}',
                'Ylimits': f'min {perpinfo.min()}, max {perpinfo.max()}',
            }
            tifffile.imwrite(
                f'{outdir}/{outname}.tiff',
                imdata,
                metadata=metadata)
        if extradims == 1:
            for i1 in np.arange(datashape[0]):
                imdata = data[i1]
                parainfo = axespara[i1]
                perpinfo = axesperp[i1]
                metadata = {
                    'Description': f'Image data identical to data saved in {hf.filename}',
                    'Xlimits': f'min {parainfo.min()}, max {parainfo.max()}',
                    'Ylimits': f'min {perpinfo.min()}, max {perpinfo.max()}',
                }
                tifffile.imwrite(
                    f'{outdir}/{outname}_{i1}.tiff',
                    imdata,
                    metadata=metadata)
        if extradims == 2:
            for i1 in np.arange(datashape[0]):
                for i2 in np.arange(datashape[1]):
                    imdata = data[i1][i2]
                    parainfo = axespara[i1][i2]
                    perpinfo = axesperp[i1][i2]
                    metadata = {
                        'Description': f'Image data identical to data saved in {hf.filename}',
                        'Xlimits': f'min {parainfo.min()}, max {parainfo.max()}',
                        'Ylimits': f'min {perpinfo.min()}, max {perpinfo.max()}',
                    }
                    tifffile.imwrite(
                        f'{outdir}/{outname}_{i1}_{i2}.tiff',
                        imdata,
                        metadata=metadata)

    def do_savedats(self, hf, intdata, qdata, tthdata):
        """
        save all 1d datasets to .dat files
        """
        datashape = np.shape(intdata)
        extradims = len(datashape) - 1
        outdir = hf.filename.strip('.hdf5')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        metadata = f'Intensity data identical to data saved in {hf.filename}\n'
        outname = outdir.split('/')[-1]
        if extradims == 0:

            intvals = intdata
            qvals = qdata
            tthetavals = tthdata
            outdf = pd.DataFrame(
                {'Q_angstrom^-1': qvals, 'Intensity': intvals, 'two_theta': tthetavals})
            with open(f'{outdir}/{outname}.dat', "w", encoding='utf-8') as f:
                f.write(metadata)
                outdf.to_csv(f, sep='\t', index=False)

        if extradims == 1:
            for i1 in np.arange(datashape[0]):
                intvals = intdata[i1]
                qvals = qdata[i1]
                tthetavals = tthdata[i1]
                outdf = pd.DataFrame(
                    {'Q_angstrom^-1': qvals, 'Intensity': intvals, 'two_theta': tthetavals})
                with open(f'{outdir}/{outname}_{i1}.dat', "w", encoding='utf-8') as f:
                    f.write(metadata)
                    outdf.to_csv(f, sep='\t', index=False)
        if extradims == 2:
            for i1 in np.arange(datashape[0]):
                for i2 in np.arange(datashape[1]):
                    intvals = intdata[i1][i2]
                    qvals = qdata[i1][i2]
                    tthetavals = tthdata[i1][i2]
                    outdf = pd.DataFrame(
                        {'Q_angstrom^-1': qvals,
                         'Intensity': intvals,
                         'two_theta_degree': tthetavals})
                    with open(f'{outdir}/{outname}_{i1}_{i2}.dat', "w", encoding='utf-8') as f:
                        f.write(metadata)
                        outdf.to_csv(f, sep='\t', index=False)

    def get_bin_axvals(self, data_in, ind):
        """
        create axes information for binoviewer output in the form
        ind,start,stop,step,startind,stopind
        """
        # print(data_in,type(data_in[0]))
        single_list = [np.int64, np.float64, int, float]
        if type(data_in[0]) in single_list:
            data = data_in
        else:
            data = data_in[0]
        startval = data[0]
        stopval = data[-1]
        stepval = data[1] - data[0]
        startind = int(np.floor(startval / stepval))
        stopind = int(startind + len(data) - 1)
        return [ind, startval, stopval, stepval,
                float(startind), float(stopind)]

    # def calcnewrange(self,range1, range2):
    #     return lambda range1, range2: [
    #         min(range1[0], range2[0]), max(range1[1], range2[1])]

    def gamdel2rots(self, gamma, delta):
        """


        Parameters
        ----------
        gamma : float
            angle rotation of gamma diffractometer circle in degrees.
        delta : float
            angle rotation of delta diffractometer circle in degrees.

        Returns
        -------
        rots : list of rotations rot1,rot2,rot3 in radians to be using by pyFAI.

        """
        rotdelta = R.from_euler('y', -delta, degrees=True)
        rotgamma = R.from_euler('z', gamma, degrees=True)
        totalrot = rotgamma * rotdelta
        fullrot = np.identity(4)
        fullrot[0:3, 0:3] = totalrot.as_matrix()
        vals = tf.euler_from_matrix(fullrot, 'rxyz')
        rots = vals[2], -vals[1], vals[0]
        return rots

    def load_curve_values(self, scan: Scan):
        """
        set attributes of experiment for easier access to key variables

        Parameters
        ----------
        scan : scan object
            scan to load experiment attributes from.

        Returns
        -------
        None.

        """
        # pylint: disable=attribute-defined-outside-init
        p2mnames = ['pil2stats', 'p2r', 'pil2roi']
        self.pixel_size = scan.metadata.diffractometer.data_file.pixel_size
        self.entry = scan.metadata.data_file.nx_entry

        self.detector_distance = scan.metadata.get_detector_distance(0)
        # else:
        #     self.detector_distance=scan.metadata.diffractometer.data_file.detector_distance
        self.incident_wavelength = 1e-10 * scan.metadata.incident_wavelength
        try:
            self.gammadata = np.array(
                self.entry.instrument.diff1gamma.value_set).ravel()
        except BaseException:
            self.gammadata = np.array(
                self.entry.instrument.diff1gamma.value).ravel()
        # self.deltadata=np.array( self.entry.instrument.diff1delta.value)
        try:
            self.deltadata = np.array(
                self.entry.instrument.diff1delta.value_set).ravel()
        except BaseException:
            self.deltadata = np.array(
                self.entry.instrument.diff1delta.value).ravel()

        if self.setup == 'DCD':
            self.dcdrad = np.array(self.entry.instrument.dcdc2rad.value)
            self.dcdomega = np.array(self.entry.instrument.dcdomega.value)
            self.projectionx = 1e-3 * self.dcdrad * \
                np.cos(np.radians(self.dcdomega))
            self.projectiony = 1e-3 * self.dcdrad * \
                np.sin(np.radians(self.dcdomega))
            dcd_sample_dist = 1e-3 * \
                scan.metadata.diffractometer._dcd_sample_distance[0]
            self.dcd_incdeg = np.degrees(
                np.arctan(
                    self.projectiony /
                    (np.sqrt(np.square(self.projectionx) + np.square(dcd_sample_dist)))))
            self.incident_angle = self.dcd_incdeg
            # self.deltadata+=self.dcd_incdeg
        elif (scan.metadata.data_file.is_eh1) & (self.setup != 'DCD'):
            self.incident_angle = scan.metadata.data_file.chi
        elif scan.metadata.data_file.is_eh2:
            self.incident_angle = scan.metadata.data_file.alpha
        else:
            self.incident_angle = [0]
        if scan.metadata.data_file.detector_name in p2mnames:
            self.deltadata = 0

        self.imshape = scan.metadata.data_file.image_shape
        self.beam_centre = scan.metadata.beam_centre
        self.rotval = round(scan.metadata.data_file.det_rot)

    def createponi(self, outpath, image2dshape, beam_centre=0, offset=0):
        """
        creates a poni file from experiment settings to use in pyFAI functions

        Parameters
        ----------
        outpath : string
            directory to save poni file to.
        image2dshape : array
            shape of image data.
        beam_centre : array, optional
            x,y values for beam centre. The default is 0.
        offset : float, optional
            offset in x for PONI1 value. The default is 0.

        Returns
        -------
        ponioutpath : string
            path to saved poni file.

        """
        datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        ponioutpath = fr'{outpath}/fast_rsm_{datetime_str}.poni'
        with open(ponioutpath, 'w', encoding='utf-8') as f:
            f.write('# PONI file created by fast_rsm\n#\n')
            f.write('poni_version: 2\n')
            f.write('Detector: Detector\n')
            f.write('Detector_config: {"pixel1":')
            pixel_line= (
                f'{self.pixel_size}, '
                f'"pixel2": {self.pixel_size}, '
                f'"max_shape": [{image2dshape[0]}, {image2dshape[1]}]'
                )

            f.write(pixel_line)
                # f'{self.pixel_size},\
                #     "pixel2": {self.pixel_size},\
                #     "max_shape": [{image2dshape[0]}, {image2dshape[1]}]')
            f.write('}\n')
            f.write(f'Distance: {self.detector_distance}\n')
            if beam_centre == 0:
                poni1 = (image2dshape[0] - offset) * self.pixel_size
                poni2 = image2dshape[1] * self.pixel_size
            elif (offset == 0) & (self.setup != 'vertical'):
                poni1 = (beam_centre[0]) * self.pixel_size
                poni2 = beam_centre[1] * self.pixel_size
            else:  # (offset == 0) & (self.setup == 'vertical'):
                poni1 = beam_centre[1] * self.pixel_size
                poni2 = (image2dshape[0] - beam_centre[0]) * self.pixel_size

            f.write(f'Poni1: {poni1}\n')
            f.write(f'Poni2: {poni2}\n')
            f.write('Rot1: 0.0\n')
            f.write('Rot2: 0.0\n')
            f.write('Rot3: 0.0\n')
            f.write(f'Wavelength: {self.incident_wavelength}')
        return ponioutpath

    # def save_projection(self, hf, projected2d, twothetas,
    #                     q_angs, intensities, config):
    #     dset = hf.create_group("projection")
    #     dset.create_dataset("projection_2d", data=projected2d[0])
    #     dset.create_dataset("config", data=str(config))

    #     dset = hf.create_group("integrations")
    #     dset.create_dataset("2thetas", data=twothetas)
    #     dset.create_dataset("Q_angstrom^-1", data=q_angs)
    #     dset.create_dataset("Intensity", data=intensities)

    def save_integration(self, hf, twothetas, q_angs,
                         intensities, configs, scan=0):
        """
        save 1d Intensity Vs Q profile to hdf5 file
        """
        dset = hf.create_group("integrations")
        dset.create_dataset("configs", data=str(configs))
        dset.create_dataset("2thetas", data=twothetas)
        dset.create_dataset("Q_angstrom^-1", data=q_angs)
        dset.create_dataset("Intensity", data=intensities)
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)
        if self.savedats is True:
            self.do_savedats(hf, intensities, q_angs, twothetas)

    def save_qperp_qpara(self, hf, qperp_qpara_map, scan=0):
        """
        save a qpara vs qperp map to hdf5 file

        """
        dset = hf.create_group("qperp_qpara")
        dset.create_dataset("images", data=qperp_qpara_map[0])
        dset.create_dataset("qpararanges", data=qperp_qpara_map[1])
        dset.create_dataset("qperpranges", data=qperp_qpara_map[2])
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)

        if self.savetiffs is True:
            self.do_savetiffs(
                hf,
                qperp_qpara_map[0],
                qperp_qpara_map[1],
                qperp_qpara_map[2])

    def save_config_variables(self, hf, joblines, pythonlocation, globalvals):
        """
        save all variables in the configuration file to the output hdf5 file
        """
        config_group = hf.create_group('i07configuration')
        configlist = [
            'setup',
            'experimental_hutch',
            'using_dps',
            'beam_centre',
            'detector_distance',
            'dpsx_central_pixel',
            'dpsy_central_pixel',
            'dpsz_central_pixel',
            'local_data_path',
            'local_output_path',
            'output_file_size',
            'save_binoculars_h5',
            'map_per_image',
            'volume_start',
            'volume_step',
            'volume_stop',
            'load_from_dat',
            'edfmaskfile',
            'specific_pixels',
            'mask_regions',
            'process_outputs',
            'scan_numbers']
        for name in configlist:
            if name in globalvals:
                outval = globalvals[f'{name}']
                if outval is None:
                    outval = 'None'
                config_group.create_dataset(f"{name}", data=outval)
        if 'ubinfo' in globalvals:
            for i, coll in enumerate(globalvals['ubinfo']):
                ubgroup = config_group.create_group(f'ubinfo_{i+1}')
                ubgroup.create_dataset(
                    f'lattice_{i+1}', data=coll['diffcalc_lattice'])
                ubgroup.create_dataset(f'u_{i+1}', data=coll['diffcalc_u'])
                ubgroup.create_dataset(f'ub_{i+1}', data=coll['diffcalc_ub'])

        config_group.create_dataset('joblines', data=joblines)
        config_group.create_dataset('python_location', data=pythonlocation)

    def reshape_to_signalshape(self, arr, signal_shape):
        """
        reshape data to match expected signal shape
        """
        testsize = int(np.prod(signal_shape)) - np.shape(arr)[0]

        fullshape = signal_shape + np.shape(arr)[1:]
        if testsize == 0:
            return np.reshape(arr, fullshape)
        else:
            extradata = np.zeros((testsize,) + (np.shape(arr)[1:]))
            outarr = np.concatenate((arr, extradata))
            return np.reshape(outarr, fullshape)

    def save_scan_field_values(self, hf, scan):
        """
        saves scanfields recorded in nexus file to hdf5 output
        """

        try:
            rank = scan.metadata.data_file.diamond_scan.scan_rank.nxdata
            fields = scan.metadata.data_file.diamond_scan.scan_fields
            scanned = [x.decode('utf-8').split('.')[0]
                       for x in fields[:rank].nxdata]
            scannedvalues = [
                np.unique(
                    scan.metadata.data_file.nx_instrument[field].value)for field in scanned]
            scannedvaluesout = [scannedvals[~np.isnan(
                scannedvals)] for scannedvals in scannedvalues]
        except BaseException:
            scanned, scannedvaluesout = None, None

        dset = hf.create_group("scanfields")
        if scan != 0:
            if scanned is not None:
                for i, field in enumerate(scanned):
                    dset.create_dataset(
                        f"dim{i}_{field}", data=scannedvaluesout[i])

    def deprecation_msg(self, option):
        """
        check list of deprecated functions, and print out warning message if needed
        """
        giwaxsdeplist = [
            'curved_projection_2D',
            'pyfai_1D',
            'qperp_qpara_map',
            'large_moving_det',
            'pyfai_2dqmap_IvsQ']
        if option in giwaxsdeplist:
            return f"option {option} has been deprecated. \
                GIWAXS mapping calculations now use pyFAI. \
            Please use process outputs 'pyfai_ivsq'  , 'pyfai_qmap'\
                  and 'pyfai_exitangles'"

# ==============testing section

    def binned_reciprocal_space_map_smm(self,
                                        num_threads: int,
                                        map_frame: Frame,
                                        output_file_name: str = "mapped",
                                        min_intensity_mask: float = None,
                                        output_file_size: float = 100,
                                        save_vtk: bool = True,
                                        save_npy: bool = True,
                                        oop: str = 'y',
                                        volume_start: np.ndarray = None,
                                        volume_stop: np.ndarray = None,
                                        volume_step: np.ndarray = None,
                                        map_each_image: bool = False):
        """
        Carries out a binned reciprocal space map for this experimental data.\
        New version using SharedMemoryManager

        Args:
            num_threads:
                How many threads should be used to carry out this map? You
                should probably set this to however many threads are available
                on your machine.
            map_frame:
                An instance of diffraction_utils.Frame that specifies what
                frame of reference and coordinate system you'd like your map to
                be carried out in.
            output_file_name:
                What should the final file be saved as?
            output_file_size:
                The desired output file size, in units of MB.
                Defaults to 100 MB.
            save_vtk:
                Should we save a vtk file for this map? Defaults to True.
            save_npy:
                Should we save a numpy file for this map? Defaults to True.
            oop:
                Which synchrotron axis should become the out-of-plane (001)
                direction. Defaults to 'y'; can be 'x', 'y' or 'z'.
        """

        # For simplicity, if qpar_qperp is asked for, we swap to the lab frame.
        # They're the same, but qpar_qperp is an average.
        # original_frame_name = map_frame.frame_name
        if map_frame.frame_name == Frame.qpar_qperp:
            map_frame.frame_name = Frame.lab
          # Compute the optimal finite differences volume.

        if volume_step is None:
            # Overwrite whichever of these we were given explicitly.
            if (volume_start is not None) & (volume_stop is not None):
                _start = np.array(volume_start)
                _stop = np.array(volume_stop)
            else:

                _start, _stop = self.q_bounds(map_frame, oop)

            step = get_step_from_filesize(_start, _stop, output_file_size)

        else:
            step = np.array(volume_step)
            _start, _stop = self.q_bounds(map_frame, oop)

        if map_frame.coordinates == Frame.sphericalpolar:
            step = np.array([0.02, np.pi / 180, np.pi / 180])

        # Make sure start and stop match the step as required by binoculars.
        start, stop = _match_start_stop_to_step(
            step=step,
            user_bounds=(volume_start, volume_stop),
            auto_bounds=(_start, _stop))

        # locks = [Lock() for _ in range(num_threads)]
        shapersm = finite_diff_shape(start, stop, step)

        # time_1 = time()
        # map_mem_total=[]
        # count_mem_total=[]
        # map_arrays = 0
        # count_arrays = 0
        # norm_arrays = 0
        images_so_far = 0

        with SharedMemoryManager() as smm:
            # shapecake = (2, 2, 2)
            shm_rsm = smm.SharedMemory(
                size=np.zeros(
                    shapersm,
                    dtype=np.float32).nbytes)
            shm_counts = smm.SharedMemory(
                size=np.zeros(shapersm, dtype=np.uint32).nbytes)
            rsm_arr = np.ndarray(
                shapersm,
                dtype=np.float32,
                buffer=shm_rsm.buf)
            counts_arr = np.ndarray(
                shapersm, dtype=np.uint32, buffer=shm_counts.buf)
            rsm_arr.fill(0)
            counts_arr.fill(0)
            l = Lock()
            for scanind, scan in enumerate(self.scans):
                new_motors = scan.metadata.data_file.get_motors()
                new_metadata = scan.metadata.data_file.get_metadata()
                bin_args = [
                    (indices,
                     start,
                     stop,
                     step,
                     min_intensity_mask,
                     scan.processing_steps,
                     scan.skip_images,
                     oop,
                     self.spherical_bragg_vec,
                     map_each_image,
                     images_so_far) for indices in chunk(
                        list(
                            range(
                                scan.metadata.data_file.scan_length)),
                        num_threads)]
                # Make a pool on which we'll carry out the processing.
                # with Pool(processes=num_threads,
                # initializer=init_process_pool,initargs=(locks,
                # num_threads,self.scans[0].metadata,map_frame,shape,output_file_name))
                # as pool:

                with Pool(num_threads, initializer=rsm_init_worker, \
            initargs=(l, shm_rsm.name, shm_counts.name, shapersm, scan.metadata,\
            new_metadata, new_motors, num_threads, map_frame, output_file_name)) as pool:
                    print(f'started pool with num_threads={num_threads}')
                    pool.starmap(bin_maps_with_indices_smm, bin_args)

                print(
                    f'finished process pool for scan {scanind+1}/{len(self.scans)}')
                images_so_far += scan.metadata.data_file.scan_length

    # fcounts= count_arrays.astype(np.float32)
        #makes sure counts are floats ready for division
    # normalised_map=np.divide(map_arrays,fcounts,
    # out=np.copy(map_arrays),where=fcounts!=0.0)#need to specify out
    # location to avoid working with non-initialised data
        normalised_map = np.divide(
            rsm_arr,
            counts_arr,
            out=np.copy(rsm_arr),
            where=counts_arr != 0.0)

        # Only save the vtk/npy files if we've been asked to.
        if save_vtk:
            print("\n**READ THIS**")
            print(f"Saving vtk to {output_file_name}.vtk")
            print(
                "This is the file that you can open with paraview. "
                "Note that this can be done with 'module load paraview' "
                "followed by 'paraview' in the terminal. \n"
                "Then open with ctrl+o as usual, navigating to the above path. "
                "Once opened, apply a threshold filter to your data to view. \n"
                "To play with colours: on your threshold filter, go to the "
                "'coloring' section and click the small 'rescale to custom "
                "range' button. Paraview is really powerful, but this should "
                "be enough to at least get you playing with it.\n")
            linear_bin_to_vtk(
                normalised_map, output_file_name, start, stop, step)
        if save_npy:
            print(f"Saving numpy array to {output_file_name}.npy")
            np.save(output_file_name, normalised_map)
            # Also save the finite differences parameters.
            np.savetxt(str(output_file_name) + "_bounds.txt",
                       np.array((start, stop, step)).transpose(),
                       header="start stop step")

        # Finally, remove any .npy files we created along the way.
        self._clean_temp_files()

        # Return the normalised RSM.
        return normalised_map, start, stop, step

    def pyfai_setup_limits(self, scanlist, limitfunction, slitdistratios):
        """
        calculate setup values needed for pyfai calculations
        """
        logger.debug("getting pyFAI limits in experiment.py")
        # pylint: disable=attribute-defined-outside-init
        if isinstance(scanlist, Scan):
            scanlistnew = [scanlist]
        else:
            scanlistnew = scanlist

        limhor = None
        limver = None
        for scan in scanlistnew:

            # anglimits,scanlength = self.pyfai_setup_limits(scan,self.calcanglim,slitdistratios)

            self.load_curve_values(scan)
            dcd_sample_dist = 1e-3 * scan.metadata.diffractometer._dcd_sample_distance
            if self.setup == 'DCD':
                tthdirect = -1 * \
                    np.degrees(np.arctan(self.projectionx / dcd_sample_dist))
            else:
                tthdirect = 0

            self.two_theta_start = self.gammadata - tthdirect

            if slitdistratios is not None:
                scanlimhor = limitfunction(
                    'hor',
                    vertsetup=(
                        self.setup == 'vertical'),
                    slithorratio=slitdistratios[1])
                scanlimver = limitfunction(
                    'vert',
                    vertsetup=(
                        self.setup == 'vertical'),
                    slitvertratio=slitdistratios[0])
            else:
                scanlimhor = limitfunction(
                    'hor', vertsetup=(
                        self.setup == 'vertical'))
                scanlimver = limitfunction(
                    'vert', vertsetup=(
                        self.setup == 'vertical'))

            scanlimits = [
                scanlimhor[0],
                scanlimhor[1],
                scanlimver[0],
                scanlimver[1]]
            if limhor is None:
                limhor = scanlimits[0:2]
                limver = scanlimits[2:]
            else:
                limhor = combine_ranges(limhor, scanlimits[0:2])
                limver = combine_ranges(limver, scanlimits[2:])

        outlimits = [limhor[0], limhor[1], limver[0], limver[1]]
        if self.setup == 'vertical':
            self.beam_centre = [self.beam_centre[1], self.beam_centre[0]]
            self.beam_centre[1] = self.imshape[0] - self.beam_centre[1]

        datacheck = 'data' in list(scan.metadata.data_file.nx_detector)
        localpathcheck = 'local_image_paths' in \
        scan.metadata.data_file.__dict__.keys()
        intcheck = isinstance(scan.metadata.data_file.scan_length, int)
        if datacheck & intcheck:
            scanlength = np.shape(
                scan.metadata.data_file.nx_detector.data[:, 1, :])[0]
            scanlength = min(scanlength, scan.metadata.data_file.scan_length)
        elif datacheck:
            scanlength = np.shape(
                scan.metadata.data_file.nx_detector.data[:, 1, :])[0]
        elif localpathcheck:
            scanlength = len(scan.metadata.data_file.local_image_paths)
        else:
            scanlength = scan.metadata.data_file.scan_length

        return outlimits, scanlength, scanlistnew

    def start_smm(self, smm, memshape):
        """
        start up the shared memory manager and associated data arrays
        """
        shm_intensities = smm.SharedMemory(
            size=np.zeros(memshape, dtype=np.float32).nbytes)
        shm_counts = smm.SharedMemory(
            size=np.zeros(
                memshape,
                dtype=np.float32).nbytes)
        arrays_arr = np.ndarray(
            memshape,
            dtype=np.float32,
            buffer=shm_intensities.buf)
        counts_arr = np.ndarray(
            memshape,
            dtype=np.float32,
            buffer=shm_counts.buf)
        arrays_arr.fill(0)
        counts_arr.fill(0)
        l = Lock()
        return shm_intensities, shm_counts, arrays_arr, counts_arr, l

    def get_input_args(self, scanlength, scalegamma,
                       multi, num_threads, fullargs):

        fullrange = np.arange(0, scanlength, scalegamma)
        selectedindices = [
            n for n in fullrange if n not in fullargs[0].skip_images]
        if multi:
            inputindices = chunk(selectedindices, num_threads)
        else:
            inputindices = selectedindices

        if fullargs[-1] is not None:
            endlist = fullargs[:-1] + [fullargs[-1][0], fullargs[-1][1]]
        else:
            endlist = fullargs[:-1]
        input_args = [[self, indices] +
                      endlist for indices in inputindices]
        return input_args

    def save_hf_map(self, hf, mapname, sum_array,
                    counts_array, mapaxisinfo, start_time):
        norm_array = np.divide(
            sum_array,
            counts_array,
            out=np.copy(sum_array),
            where=counts_array != 0.0)
        end_time = time()
        times = [start_time, end_time]
        dset = hf.create_group(f"{mapname}")
        dset.create_dataset(f"{mapname}_map", data=norm_array)
        dset.create_dataset("map_para", data=mapaxisinfo[1])
        dset.create_dataset("map_para_unit", data=mapaxisinfo[3])
        # list(reversed(mapaxisinfo[0])))
        dset.create_dataset("map_perp", data=mapaxisinfo[0])
        dset.create_dataset("map_perp_unit", data=mapaxisinfo[2])
        dset.create_dataset("map_perp_indices", data=[0, 1, 2])
        dset.create_dataset("map_para_indices", data=[0, 1, 3])

        if self.savetiffs:
            self.do_savetiffs(hf, norm_array, mapaxisinfo[1], mapaxisinfo[0])

        minutes = (times[1] - times[0]) / 60
        print(f'total calculation took {minutes}  minutes')

    def pyfai_moving_exitangles_smm(self,
                                    hf,
                                    scanlist,
                                    num_threads,
                                    output_file_path,
                                    pyfaiponi,
                                    radrange,
                                    radstepval,
                                    qmapbins=np.array([800,
                                                       800]),
                                    slitdistratios=None):
        """
        calculate exit angle map with moving detector
        """

        # pylint: disable=unused-argument
        # pylint: disable=unused-variable

        exhexv_array_total = 0
        exhexv_counts_total = 0
        anglimitsout, scanlength, scanlistnew = self.pyfai_setup_limits(
            scanlist, self.calcanglim, slitdistratios)
        with SharedMemoryManager() as smm:
            shapeexhexv = (qmapbins[1], qmapbins[0])
            shm_intensities, shm_counts, arrays_arr, counts_arr, l = self.start_smm(
                smm, shapeexhexv)
            start_time = time()
            for scanind, scan in enumerate(scanlistnew):

                anglimits, scanlength, scanlistnew = self.pyfai_setup_limits(
                    scan, self.calcanglim, slitdistratios)
                scalegamma = 1
                # fullargs needs to start with scan and end with slitdistratios
                fullargs = [
                    scan,
                    shapeexhexv,
                    pyfaiponi,
                    anglimitsout,
                    qmapbins,
                    slitdistratios]
                input_args = self.get_input_args(
                    scanlength, scalegamma, True, num_threads, fullargs)
                print(f'starting process pool with num_threads=\
        {num_threads} for scan {scanind+1}/{len(scanlistnew)}')

                with Pool(num_threads, initializer=pyfai_init_worker, \
        initargs=(l, shm_intensities.name, shm_counts.name, shapeexhexv)) as pool:
                    mapaxisinfolist = pool.starmap(
                        pyfai_move_exitangles_worker, input_args)
                print(
                    f'finished process pool for scan {scanind+1}/{len(scanlistnew)}')

        mapaxisinfo = mapaxisinfolist[0]
        exhexv_array_total = arrays_arr
        exhexv_counts_total = counts_arr
        self.save_hf_map(
            hf,
            "exit_angles",
            exhexv_array_total,
            exhexv_counts_total,
            mapaxisinfo,
            start_time)
        return mapaxisinfo

    def pyfai_moving_qmap_smm(
            self,
            hf,
            scanlist,
            num_threads,
            output_file_path,
            pyfaiponi,
            radrange,
            radstepval,
            qmapbins=(
                1200,
                1200),
            slitdistratios=None):
        """
        calculate q_para vs q_perp map for a moving detector scan
        """

        # pylint: disable=unused-argument
        # pylint: disable=unused-variable
        qlimitsout = [0, 0, 0, 0]
        qpqp_array_total = 0
        qpqp_counts_total = 0

        qlimitsout, scanlength, scanlistnew = self.pyfai_setup_limits(
            scanlist, self.calcqlim, slitdistratios)

        with SharedMemoryManager() as smm:

            shapeqpqp = (qmapbins[1], qmapbins[0])
            shm_intensities, shm_counts, arrays_arr, counts_arr, l = self.start_smm(
                smm, shapeqpqp)

            for scanind, scan in enumerate(scanlistnew):
                qlimits, scanlength, scanlistnew = self.pyfai_setup_limits(
                    scan, self.calcqlim, slitdistratios)
                start_time = time()
                scalegamma = 1
                # fullargs needs to start with scan and end with slitdistratios

                fullargs = [
                    scan,
                    shapeqpqp,
                    pyfaiponi,
                    qmapbins,
                    qlimitsout,
                    slitdistratios]
                input_args = self.get_input_args(
                    scanlength, scalegamma, True, num_threads, fullargs)
                # print(np.shape(input_args))
                print(
                    f'starting process pool with num_threads=\
                    {num_threads} for scan {scanind+1}/{len(scanlistnew)}')

                with Pool(num_threads, \
                          initializer=pyfai_init_worker, \
                initargs=(l, shm_intensities.name, shm_counts.name, shapeqpqp)) as pool:
                    mapaxisinfolist = pool.starmap(
                        pyfai_move_qmap_worker, input_args)
                print(
                    f'finished process pool for scan {scanind+1}/{len(scanlistnew)}')

        mapaxisinfo = mapaxisinfolist[0]
        qpqp_array_total = arrays_arr
        qpqp_counts_total = counts_arr
        self.save_hf_map(
            hf,
            "qpara_qperp",
            qpqp_array_total,
            qpqp_counts_total,
            mapaxisinfo,
            start_time)
        return mapaxisinfo

    def pyfai_moving_ivsq_smm(
            self,
            hf,
            scanlist,
            num_threads,
            output_file_path,
            pyfaiponi,
            radrange,
            radstepval,
            qmapbins=(
                1200,
                1200),
            slitdistratios=None):
        """
        calculate 1d Intensity Vs Q profile for a moving detector scan
        """

        # pylint: disable=unused-argument
        # pylint: disable=unused-variable
        fullranges, scanlength, scanlistnew = self.pyfai_setup_limits(
            scanlist, self.calcanglim, slitdistratios)
        absranges = np.abs(fullranges)
        radmax = np.max(absranges)
        # radrange=(0,radmax)
        con1 = np.abs(fullranges[0]) < np.abs(fullranges[0] - fullranges[1])
        con2 = np.abs(fullranges[2]) < np.abs(fullranges[2] - fullranges[3])

        if (con1) & (con2):
            radrange = (0, radmax)

        elif con1:
            radrange = np.sort([absranges[2], absranges[3]])
        elif con2:
            radrange = np.sort([absranges[0], absranges[1]])
        else:

            radrange = (np.max([absranges[0], absranges[2]]),
                        np.max([absranges[1], absranges[3]]))

        nqbins = int(np.ceil((radrange[1] - radrange[0]) / radstepval))

        with SharedMemoryManager() as smm:

            shapeqi = (3, np.abs(nqbins))
            shm_intensities, shm_counts, arrays_arr, counts_arr, l = self.start_smm(
                smm, shapeqi)

            for scanind, scan in enumerate(scanlistnew):
                qlimits, scanlength, scanlistnew = self.pyfai_setup_limits(
                    scan, self.calcqlim, slitdistratios)
                start_time = time()
                scalegamma = 1
                # fullargs needs to start with scan and end with slitdistratios
                fullargs = [scan, shapeqi, pyfaiponi, radrange, slitdistratios]
                input_args = self.get_input_args(
                    scanlength, scalegamma, True, num_threads, fullargs)
                print(
                    f'starting process pool with num_threads=\
                    {num_threads} for scan {scanind+1}/{len(scanlistnew)}')

                with Pool(num_threads, \
                          initializer=pyfai_init_worker, \
                    initargs=(l, shm_intensities.name, shm_counts.name, shapeqi)) as pool:
                    pool.starmap(pyfai_move_ivsq_worker, input_args)
                print(
                    f'finished process pool for scan {scanind+1}/{len(scanlistnew)}')
        qi_array = np.divide(
            arrays_arr[0],
            counts_arr[0],
            out=np.copy(
                arrays_arr[0]),
            where=counts_arr[0] != 0)
        end_time = time()

        dset = hf.create_group("integrations")
        dset.create_dataset("Intensity", data=qi_array)
        dset.create_dataset("Q_angstrom^-1", data=arrays_arr[1])
        dset.create_dataset("2thetas", data=arrays_arr[2])

        if self.savedats:
            self.do_savedats(hf, qi_array, arrays_arr[1], arrays_arr[2])
        minutes = (end_time - start_time) / 60
        print(f'total calculation took {minutes}  minutes')

    def pyfai_static_exitangles(self, hf, scan, num_threads, pyfaiponi, ivqbins,
                                qmapbins=np.array([1200, 1200]), slitdistratios=None):
        """
        calculate the map of vertical exit angle Vs horizontal exit angle using pyFAI

        Parameters
        ----------
        hf : hdf5 file
            open hdf5 file to write data to.
        scan : scan object
            scan to be analysed.
        num_threads : int
            number of threads used in calculation.
        pyfaiponi : string
            path to PONI file.
        ivqbins : int
            number of bins for I Vs Q profile.
        qmapbins : array, optional
            number of x,y bins for map. The default is 0.

        Returns
        -------
        None.

        """
        # pylint: disable=unused-argument
        # pylint: disable=unused-variable
        start_time = time()
        anglimits, scanlength, scanlistnew = self.pyfai_setup_limits(
            scan, self.calcanglim, slitdistratios)
        # calculate map bins if not specified using resolution of 0.01 degrees

        scalegamma = 1

        print(f'starting process pool with num_threads={num_threads}')
        all_maps = []
        all_xlabels = []
        all_ylabels = []
        all_mapaxisinfo = []

        with Pool(processes=num_threads) as pool:

            # fullargs needs to start with scan and end with slitdistratios
            fullargs = [
                scan,
                self.two_theta_start,
                pyfaiponi,
                anglimits,
                qmapbins,
                ivqbins,
                slitdistratios]
            input_args = self.get_input_args(
                scanlength, scalegamma, False, num_threads, fullargs)
            results = pool.starmap(pyfai_stat_exitangles, input_args)
            maps = [result[0] for result in results]
            xlabels = [result[1] for result in results]
            ylabels = [result[2] for result in results]
            mapaxisinfo = [result[3] for result in results]
            all_maps.append(maps)
            all_xlabels.append(xlabels)
            all_ylabels.append(ylabels)
            all_mapaxisinfo.append(mapaxisinfo)

        print('finished process pool')

        signal_shape = np.shape(scan.metadata.data_file.default_signal)
        if len(signal_shape) > 1:
            savemaps = self.reshape_to_signalshape(all_maps[0], signal_shape)
        else:
            savemaps = all_maps[0]
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)
        self.save_hf_map(
            hf,
            "exit_angles",
            savemaps,
            np.ones(
                np.shape(savemaps)),
            all_mapaxisinfo[0][0],
            start_time)

    def pyfai_static_qmap(self, hf, scan, num_threads, output_file_path,
                          pyfaiponi, ivqbins, qmapbins=0, slitdistratios=None):
        """
        calculate 2d q_para vs q_perp mape for a static detector scan
        """

        # pylint: disable=unused-argument
        # pylint: disable=unused-variable
        qlimits, scanlength, scanlistnew = self.pyfai_setup_limits(
            scan, self.calcqlim, slitdistratios)

        # calculate map bins if not specified using resolution of 0.01 degrees
        logger.debug(f'from experiment.py calculating static qmap for scan {scan}')
        if qmapbins == 0:
            qstep = round(self.calcq(1.00, self.incident_wavelength) -
                          self.calcq(1.01, self.incident_wavelength), 4)
            binshor = abs(round(((qlimits[1] - qlimits[0]) / qstep) * 1.05))
            binsver = abs(round(((qlimits[3] - qlimits[2]) / qstep) * 1.05))
            qmapbins = (binshor, binsver)

        scalegamma = 1

        print(f'starting process pool with num_threads={num_threads}')
        all_maps = []
        all_xlabels = []
        all_ylabels = []

        with Pool(processes=num_threads) as pool:
            # fullargs needs to start with scan and end with slitdistratios
            fullargs = [
                scan,
                self.two_theta_start,
                pyfaiponi,
                qlimits,
                qmapbins,
                ivqbins,
                slitdistratios]
            input_args = self.get_input_args(
                scanlength, scalegamma, False, num_threads, fullargs)
            results = pool.starmap(pyfai_stat_qmap, input_args)
            maps = [result[0] for result in results]
            xlabels = [result[1] for result in results]
            ylabels = [result[2] for result in results]
            all_maps.append(maps)
            all_xlabels.append(xlabels)
            all_ylabels.append(ylabels)

        print('finished process pool')

        signal_shape = np.shape(scan.metadata.data_file.default_signal)
        outlist = [all_maps[0], all_xlabels[0], all_ylabels[0]]
        if len(signal_shape) > 1:
            outlist = [
                self.reshape_to_signalshape(
                    arr, signal_shape) for arr in outlist]

        binset = hf.create_group("binoculars")
        binset.create_dataset("counts", data=outlist[0])
        binset.create_dataset(
            "contributions",
            data=np.ones(
                np.shape(
                    outlist[0])))
        axgroup = binset.create_group("axes", track_order=True)

        zlen = np.shape(outlist[0])[0]
        if zlen > 1:
            axgroup.create_dataset(
                "1_index", data=self.get_bin_axvals(
                    np.arange(zlen), 0), track_order=True)
        else:
            axgroup.create_dataset(
                "1_index",
                data=[
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0],
                track_order=True)

        axgroup.create_dataset(
            "2_q_perp", data=self.get_bin_axvals(
                outlist[1][0], 1), track_order=True)
        axgroup.create_dataset(
            "3_q_para", data=self.get_bin_axvals(
                outlist[2][0], 2), track_order=True)

        dset = hf.create_group("qpara_qperp")
        dset["qpara_qperp_map"] = h5py.SoftLink('/binoculars/counts')
        dset.create_dataset("map_para", data=outlist[1])
        dset.create_dataset("map_perp", data=outlist[2])
        dset.create_dataset("map_perp_indices", data=[0, 1, 2])
        dset.create_dataset("map_para_indices", data=[0, 1, 3])

        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)
        if self.savetiffs:
            self.do_savetiffs(hf, outlist[0], outlist[1], outlist[2])

    def pyfai_static_ivsq(self, hf, scan, num_threads, output_file_path,
                          pyfaiponi, ivqbins, qmapbins=0, slitdistratios=None):
        """
        calculate Intensity Vs Q 1d profile from static detector scan
        """
        # pylint: disable=unused-argument
        # pylint: disable=unused-variable
        qlimits, scanlength, scanlistnew = self.pyfai_setup_limits(
            scan, self.calcqlim, slitdistratios)

        # calculate map bins if not specified using resolution of 0.01 degrees

        if qmapbins == 0:
            qstep = round(self.calcq(1.00, self.incident_wavelength) -
                          self.calcq(1.01, self.incident_wavelength), 4)
            binshor = abs(round(((qlimits[1] - qlimits[0]) / qstep) * 1.05))
            binsver = abs(round(((qlimits[3] - qlimits[2]) / qstep) * 1.05))
            qmapbins = (binshor, binsver)

        scalegamma = 1

        print(f'starting process pool with num_threads={num_threads}')
        all_ints = []
        all_two_ths = []
        all_qs = []

        with Pool(processes=num_threads) as pool:

            # fullargs needs to start with scan and end with slitdistratios
            fullargs = [
                scan,
                self.two_theta_start,
                pyfaiponi,
                qmapbins,
                ivqbins,
                slitdistratios]
            input_args = self.get_input_args(
                scanlength, scalegamma, False, num_threads, fullargs)

            results = pool.starmap(pyfai_stat_ivsq, input_args)
            intensities = [result[0] for result in results]
            two_th_vals = [result[1] for result in results]
            q_vals = [result[2] for result in results]
            all_ints.append(intensities)
            all_two_ths.append(two_th_vals)
            all_qs.append(q_vals)

        print('finished process pool')

        signal_shape = np.shape(scan.metadata.data_file.default_signal)
        outlist = [all_ints[0], all_qs[0], all_two_ths[0]]
        if len(signal_shape) > 1:
            outlist = [
                self.reshape_to_signalshape(
                    arr, signal_shape) for arr in outlist]

        dset = hf.create_group("integrations")
        dset.create_dataset("Intensity", data=outlist[0])
        dset.create_dataset("Q_angstrom^-1", data=outlist[1])
        dset.create_dataset("2thetas", data=outlist[2])
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)
        if self.savedats is True:
            self.do_savedats(hf, outlist[0], outlist[1], outlist[2])

    @classmethod
    def from_i07_nxs(cls,
                     nexus_paths: List[Union[str, Path]],
                     beam_centre: Tuple[int],
                     detector_distance: float,
                     setup: str,
                     path_to_data: str = '',
                     using_dps: bool = False,
                     experimental_hutch=0):
        """
        Generates an instance of Experiment from paths to nexus files obtained
        from the i07 beamline. Also requires a few other essential pieces of
        information to uniquely specify the experimental setup (because you
        could be doing basically anything on i07...)

        Args:
            nexus_paths:
                Path to the nexus files containing all of the scans' metadata.
                If the experiment consists of only one scan, then you can pass
                a single string/Path in place of a list of strings/paths.
            beam_centre:
                A (y, x) tuple of the beam centre, measured in the usual image
                coordinate system, in units of pixels.
            detector_distance:
                The distance between the sample and the detector.
            setup:
                What was the experimental setup? Can be "vertical", "horizontal"
                or "DCD".
            path_to_data:
                Path to the directory in which the images are stored. Defaults
                to '', in which case a bunch of reasonable directories will be
                searched for the images.

        Returns:
            Corresponding instance of Experiment.
        """

        t1 = time()
        # Make sure that we have a list, in case we just received a single
        # path.
        if isinstance(nexus_paths, (str, Path)):
            nexus_paths = [nexus_paths]

        # Instantiate all of the scans.
        scans = [
            io.from_i07(x, beam_centre, detector_distance,
                        setup, path_to_data, using_dps, experimental_hutch)
            for x in nexus_paths]
        print(f"Took {time() - t1}s to load all nexus files.")
        return cls(scans, setup)


def _match_start_stop_to_step(
        step,
        user_bounds,
        auto_bounds,
        eps=1e-5):
    warning_str = ("User provided bounds (volume_start, volume_stop) do not "
                   "match the step size volume_step. Bounds will be adjusted "
                   "automatically. If you want to avoid this warning, make "
                   "that the bounds match the step size, i.e. volume_bound = "
                   "volume_step * integer.")

    if user_bounds == (None, None):
        # use auto bounds and expand both ways
        return (np.floor(auto_bounds[0] / step) * step,
                np.ceil(auto_bounds[1] / step) * step)
    if user_bounds[0] is None:
        # keep user value and expand to rightdone image {i+1}/{totalimages}
        stop = np.ceil(user_bounds[1] / step) * step
        checkstop = np.sum(np.any(abs(stop - user_bounds[1]) > eps))
        if checkstop > 0:
            print(warning_str)
        return np.floor(auto_bounds[0] / step) * step, stop
    if user_bounds[1] is None:
        # keep user value and expand to left
        start = np.floor(user_bounds[0] / step) * step
        checkstart = np.sum(abs(user_bounds[0] - start) > eps)
        if checkstart > 0:
            print(warning_str)
        return start, np.ceil(auto_bounds[1] / step) * step

    start, stop = (np.floor(user_bounds[0] / step) * step,
                    np.ceil(user_bounds[1] / step) * step)
    checkstart = np.sum(abs(user_bounds[0] - start) > eps)
    checkstop = np.sum(np.any(abs(stop - user_bounds[1]) > eps))
    checkboth = checkstart + checkstop
    if checkboth > 0:
        print(warning_str)
    return start, stop
