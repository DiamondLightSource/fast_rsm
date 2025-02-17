"""
This file contains the Experiment class, which contains all of the information
relating to your experiment.
"""

import os
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Lock
from pathlib import Path
from time import time
from typing import List, Tuple, Union
import mapper_c_utils
import numpy as np
from diffraction_utils import Frame, Region
from scipy.constants import physical_constants
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import transformations as tf

import fast_rsm.io as io
from fast_rsm.binning import weighted_bin_1d, finite_diff_shape
from fast_rsm.meta_analysis import get_step_from_filesize
from fast_rsm.scan import Scan, init_process_pool, bin_maps_with_indices, chunk, \
    init_pyfai_process_pool,pyfaicalcint,pyfai_qmap_qvsI,pyfai_stat_qmap,pyfai_stat_ivsq,pyfai_move_qmap,pyfai_move_ivsq, pyfai_stat_exitangles,pyfai_move_exitangles
from fast_rsm.writing import linear_bin_to_vtk
import pandas as pd
import pyFAI,fabio
#from datetime import datetime
import h5py
import tifffile
from fast_rsm.scan import Scan
#from memory_profiler import profile



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
    ang_wavevector = 2*np.pi*energy/(planck*speed_of_light)

    # Do some basic geometry.
    theta_values = np.arccos(1 - np.square(q_values)/(2*ang_wavevector**2))

    # Convert from radians to degrees.
    return theta_values*180/np.pi


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

    def __init__(self, scans: List[Scan]) -> None:
        self.scans = scans
        self._data_file_names = []
        self._normalisation_file_names = []

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
        Masks the requested regions.
        """
        # Make sure that we have a list of regions, not an individual region.
        if isinstance(regions, Region):
            regions = [regions]
        for scan in self.scans:
            scan.metadata.mask_regions = regions
            
    def mask_edf(self,edfmask):


        if edfmask!=None:
            maskimg=fabio.open(edfmask)
            mask=maskimg.data
            if self.scans[0].metadata.data_file.is_rotated==True:
                mask=np.flip(mask.transpose(),axis=0)
        else:
            mask=None

        
        for scan in self.scans:
            scan.metadata.edfmask = mask

            

    def binned_reciprocal_space_map(self,
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
        Carries out a binned reciprocal space map for this experimental data.

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
        original_frame_name = map_frame.frame_name
        if map_frame.frame_name == Frame.qpar_qperp:
            map_frame.frame_name = Frame.lab
          # Compute the optimal finite differences volume.
          

        if volume_step is None:
            # Overwrite whichever of these we were given explicitly.
            if (volume_start is not None)&(volume_stop is not None):
                _start = np.array(volume_start)
                _stop = np.array(volume_stop)
            else:
                _start, _stop = self.q_bounds(map_frame, oop)
            step = get_step_from_filesize(_start, _stop, output_file_size)
            start, stop = _match_start_stop_to_step(
                step=step,
                user_bounds=(volume_start, volume_stop),
                auto_bounds=(_start, _stop))

        else:
            step = np.array(volume_step)
            _start, _stop = self.q_bounds(map_frame, oop)
        
        # Make sure start and stop match the step as required by binoculars.
            start, stop = _match_start_stop_to_step(
                step=step,
                user_bounds=(volume_start, volume_stop),
                auto_bounds=(_start, _stop))

        locks = [Lock() for _ in range(num_threads)]
        shape = finite_diff_shape(start, stop, step)


        time_1 = time()
        #map_mem_total=[]
        #count_mem_total=[]
        map_arrays=0
        count_arrays=0
        norm_arrays=0
        images_so_far = 0

        for scan in self.scans:
            async_results = []
            # Make a pool on which we'll carry out the processing.
            with Pool(
            processes=num_threads,  # The size of our pool.
            initializer=init_process_pool,  # Our pool's initializer.
            initargs=(locks,  # The initializer makes this lock global.
                  num_threads,  # Initializer makes num_threads global.
                  self.scans[0].metadata,
                  map_frame,
                  shape,
                  output_file_name)
        ) as pool:

                for indices in chunk(list(range(
                    scan.metadata.data_file.scan_length)), num_threads):

                    new_motors = scan.metadata.data_file.get_motors()
                    new_metadata = scan.metadata.data_file.get_metadata()

                    # Submit the binning as jobs on the pool.
                    # Note that serializing the map_frame and the scan.metadata
                    # are the only things that take finite time.
                    async_results.append(pool.apply_async(
                    bin_maps_with_indices,
                    (indices, start, stop, step,
                     min_intensity_mask,  new_motors, new_metadata,
                     scan.processing_steps, scan.skip_images, oop,
                     map_each_image, images_so_far)))

                    images_so_far += scan.metadata.data_file.scan_length

                print(f"Took {time() - time_1}s to prepare the calculation.")
                map_names = []
                count_names = []
                map_mem =[]
                count_mem =[]
                for result in async_results:
                    # Make sure that we're storing the location of the shared memory
                    # block.
                    shared_rsm_name, shared_count_name = result.get()
                    if shared_rsm_name not in map_names:
                        map_names.append(shared_rsm_name)
                    if shared_count_name not in count_names:
                        count_names.append(shared_count_name)

                    # Make sure that no error was thrown while mapping.
                    if not result.successful():
                        raise ValueError(
                        "Could not carry out map for an unknown reason. "
                        "Probably one of the threads segfaulted, or something.")
                scanname=scan.metadata.data_file.diamond_scan.nxfilename.split('/')[-1]
                print(f"\nCalculation for scan {scanname} complete.")
                map_mem = [SharedMemory(x) for x in map_names]
                count_mem = [SharedMemory(x) for x in count_names]

                new_map_arrays = np.array([
                    np.ndarray(shape=shape, dtype=np.float32, buffer=x.buf)
                    for x in map_mem])
                new_count_arrays = np.array([
                    np.ndarray(shape=shape, dtype=np.uint32, buffer=y.buf)
                    for y in count_mem])

                #map_mem_total+=map_mem
                #count_mem_total+=(count_mem)
                if np.size(map_arrays)==1:
                    map_arrays=np.sum(new_map_arrays,axis=0)
                    count_arrays=np.sum(new_count_arrays,axis=0)
                else:
                    new_maps=np.sum(new_map_arrays,axis=0)
                    new_counts=np.sum(new_count_arrays,axis=0)
                    map_arrays=np.sum([map_arrays,new_maps],axis=0)
                    count_arrays=np.sum([count_arrays,new_counts],axis=0)
                #           normalised_map = map_arrays/(count_arrays.astype(np.float32))
                # Make sure all our shared memory has been closed nicely.
            for shared_mem in map_mem:
                shared_mem.close()
                try:
                    shared_mem.unlink()
                except:
                    pass
            for shared_mem in count_mem:
                shared_mem.close()
                try:
                    shared_mem.unlink()
                except:
                    pass
                
        fcounts= count_arrays.astype(np.float32) #makes sure counts are floats ready for division
        normalised_map=np.divide(map_arrays,fcounts, out=np.copy(map_arrays),where=fcounts!=0.0)#need to specify out location to avoid working with non-initialised data

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

    def _project_to_1d(self,
                       num_threads: int,
                       output_file_name: str = "mapped",
                       num_bins: int = 1000,
                       bin_size: float = None,
                       oop: str = 'y',
                       tth=False,
                       only_l=False):
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
        map_frame = Frame(Frame.sample_holder, coordinates=Frame.cartesian)
        if only_l:
            map_frame = Frame(Frame.hkl, coordinates=Frame.cartesian)

        rsm, start, stop, step = self.binned_reciprocal_space_map(
            num_threads, map_frame, output_file_name, save_vtk=False,
            oop=oop)

        # RSM could have a lot of nan elements, representing unmeasured voxels.
        # Lets make sure we zero these before continuing.
        rsm = np.nan_to_num(rsm, nan=0)

        q_x = np.arange(start[0], stop[0], step[0], dtype=np.float32)
        q_y = np.arange(start[1], stop[1], step[1], dtype=np.float32)
        q_z = np.arange(start[2], stop[2], step[2], dtype=np.float32)

        if tth:
            energy = self.scans[0].metadata.data_file.probe_energy
            q_x = _q_to_theta(q_x, energy)
            q_y = _q_to_theta(q_y, energy)
            q_z = _q_to_theta(q_z, energy)
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
        rsm = rsm.ravel()
        q_lengths = q_lengths.ravel()

        # If we haven't been given a bin_size, we need to calculate it.
        min_q = float(np.min(q_lengths))
        max_q = float(np.max(q_lengths))

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

    def intensity_vs_l(self,
                       num_threads: int,
                       output_file_name: str = "mapped",
                       num_bins: int = 1000,
                       bin_size: float = None,
                       oop: str = 'y'):
        """
        Maps this experiment to a simple intensity vs l representation.

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
        intensity, l = self._project_to_1d(
            num_threads, output_file_name, num_bins, bin_size, only_l=True,
            oop=oop)

        to_save = np.transpose((l, intensity))
        output_file_name = str(output_file_name) + "_l.txt"
        print(f"Saving intensity vs l to {output_file_name}")
        np.savetxt(output_file_name, to_save, header="l intensity")

        return l, intensity

    def intensity_vs_tth(self,
                         num_threads: int,
                         output_file_name: str = "mapped",
                         num_bins: int = 1000,
                         bin_size: float = None,
                         oop: str = 'y'):
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
        # This just aliases to self._project_to_1d, which handles the minor
        # difference between a projection to |Q| and 2θ.
        intensity, tth = self._project_to_1d(
            num_threads, output_file_name, num_bins, bin_size, tth=True,
            oop=oop)

        to_save = np.transpose((tth, intensity))
        output_file_name = str(output_file_name) + "_tth.txt"
        print(f"Saving intensity vs tth to {output_file_name}")
        np.savetxt(output_file_name, to_save, header="tth intensity")

        return tth, intensity

    def intensity_vs_q(self,
                       num_threads: int,
                       output_file_name: str = "I vs Q",
                       num_bins: int = 1000,
                       bin_size: float = None,
                       oop: str = 'y'):
        """
        Maps this experiment to a simple insenity vs |Q| plot.

        Under the hood, this runs a binned reciprocal space map with an
        auto-generated resolution such that the reciprocal space volume is
        100MB. You really shouldn't need more than 100MB of reciprocal space
        for a simple line chart...

        TODO: one day, this shouldn't run the 3D binned reciprocal space map
        and should instead directly bin to a one-dimensional space. The
        advantage of this approach is that it is much easier to maintain. The
        disadvantage is that theoretical optimal performance is worse, memory
        footprint is higher and some data could be incorrectly binned due to
        double binning. In reality, it's fine: inaccuracies are pedantic and
        performance is pretty damn good.
        """
        # This just aliases to self._project_to_1d, which handles the minor
        # difference between a projection to |Q| and 2θ.
        intensity, q = self._project_to_1d(
            num_threads, output_file_name, num_bins, bin_size, oop=oop)

        to_save = np.transpose((q, intensity))
        output_file_name = str(output_file_name) + "_Q.txt"
        print(f"Saving intensity vs q to {output_file_name}")
        np.savetxt(output_file_name, to_save, header="|Q| intensity")
        return q, intensity

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
            start, stop = scan.q_bounds(frame, oop)
            starts.append(start)
            stops.append(stop)

        # Return the min of the starts and the max of the stops.
        starts, stops = np.array(starts), np.array(stops)
        return np.min(starts, axis=0), np.max(stops, axis=0)
    
    def calcq(self,twotheta,wavelength):
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
        return (4*np.pi/wavelength)*np.sin(np.radians(twotheta/2))*1e-10
    
    def calcqstep(self,gammastep,gammastart,wavelength):
        qstep=self.calcq(gammastart+gammastep,wavelength)-self.calcq(gammastart,wavelength)
        return qstep
    
    def histogram_xy(self,x, y, step_size):
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

    def SOHqcalc(self,angle,kmod):
        """
        

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
        return np.sin(np.radians(angle))*kmod*1e-10
    
    def calcanglim(self,axis,vertsetup=False):
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
        
        switchaxis = {'vert': 'hor', 'hor': 'vert'}            
        if (vertsetup==True):
            axis=switchaxis[axis]
        if axis=='vert':
            pixlow=self.imshape[0]-self.beam_centre[0]
            pixhigh=self.beam_centre[0]
            highsection=np.max(self.deltadata)
            lowsection=np.min(self.deltadata)
        elif axis=='hor':
            pixhigh=(self.beam_centre[1])
            pixlow=(self.imshape[1]-self.beam_centre[1])
            highsection=np.max(self.two_theta_start)
            lowsection=np.min(self.two_theta_start)

        maxangle=highsection+np.degrees(np.arctan((pixhigh*self.pixel_size)/self.detector_distance))
        minangle=lowsection-np.degrees(np.arctan((pixlow*self.pixel_size)/self.detector_distance))
        
        return maxangle,minangle
        
        
    
    def calcqlim(self,axis,vertsetup=False):
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
        kmod=2*np.pi/ (self.incident_wavelength)
        
        switchaxis = {'vert': 'hor', 'hor': 'vert'}    
        if (vertsetup==True):
            axis=switchaxis[axis]
            horpix=self.beam_centre[1]
            vertangles=self.gammadata
        else:
            horpix= self.beam_centre[0]  
            vertangles=self.deltadata
                    
        
        if axis=='vert':
            pixlow=self.imshape[0]-self.beam_centre[0]
            pixhigh=self.beam_centre[0]
            highsection=np.max(self.deltadata)
            lowsection=np.min(self.deltadata)
        elif axis=='hor':
            pixhigh=(self.beam_centre[1])
            pixlow=(self.imshape[1]-self.beam_centre[1])
            highsection=np.max(self.two_theta_start)
            lowsection=np.min(self.two_theta_start)

        maxangle=highsection+np.degrees(np.arctan((pixhigh*self.pixel_size)/self.detector_distance))
        minangle=lowsection-np.degrees(np.arctan((pixlow*self.pixel_size)/self.detector_distance))

        if (axis=='vert'):
            qupp=self.SOHqcalc(maxangle,kmod)
            qlow=self.SOHqcalc(minangle,kmod)
            maxtthrad=np.radians(np.max(self.two_theta_start))
            maxanglerad=np.radians(np.max(maxangle))
            maxincrad=np.radians(np.max(self.incident_angle))
            extraincq=kmod*1e-10*np.sin(maxincrad)
            minusexitq_x=kmod*1e-10*np.cos(maxanglerad)*np.cos(maxtthrad)*np.sin(maxincrad)
            minusexitq_z=kmod*1e-10*np.sin(maxanglerad)*(1-np.cos(maxincrad))
            extravert=extraincq-minusexitq_x-minusexitq_z
            qupp+=extravert
            qlow-=extravert  
        
        elif axis=='hor':
            qupp=self.SOHqcalc(maxangle/2,kmod)*2
            qlow=self.SOHqcalc(minangle/2,kmod)*2
            maxdel=np.max(vertangles) +np.degrees(np.arctan((horpix*self.pixel_size)/self.detector_distance))
            s1=kmod*np.cos(np.radians(maxdel))*np.sin(np.radians(maxangle))
            s2=kmod*(1-np.cos(np.radians(maxdel))*np.cos(np.radians(maxangle)))
            qupp_withdelta=np.sqrt(np.square(s1)+np.square(s2))*1e-10*np.sign(maxangle)
            s3=kmod*np.cos(np.radians(maxdel))*np.sin(np.radians(minangle))
            s4=kmod*(1-np.cos(np.radians(maxdel))*np.cos(np.radians(minangle)))
            qlow_withdelta=np.sqrt(np.square(s3)+np.square(s4))*1e-10*np.sign(minangle)

            if abs(qupp_withdelta)>abs(qupp):
                qupp=-1*qupp_withdelta
            else:
                qupp*=-1
            if abs(qlow_withdelta)>abs(qlow):
                qlow=-1*qlow_withdelta
            else:
                qlow*=-1
        
        return qupp,qlow
    
    def do_savetiffs(self,hf,data,axespara,axesperp):
        datashape=np.shape(data)
        extradims=len(datashape)-2
        outdir=hf.filename.strip('.hdf5')
        try:
            os.mkdir(outdir)
        except:
            print('directory already exists')
        outname=outdir.split('\\')[-1]
        if extradims==0:
            imdata=data
            parainfo=axespara
            perpinfo=axesperp
            metadata={
                'Description':f'Image data identical to data saved in {hf.filename}',
                'Xlimits': f'min {parainfo.min()}, max {parainfo.max()}',
                'Ylimits': f'min {perpinfo.min()}, max {perpinfo.max()}',
                }
            tifffile.imwrite(f'{outdir}/{outname}.tiff',imdata,metadata=metadata)
        if extradims==1:
            for i1 in np.arange(datashape[0]):
                imdata=data[i1]
                parainfo=axespara[i1]
                perpinfo=axesperp[i1]
                metadata={
                    'Description':f'Image data identical to data saved in {hf.filename}',
                    'Xlimits': f'min {parainfo.min()}, max {parainfo.max()}',
                    'Ylimits': f'min {perpinfo.min()}, max {perpinfo.max()}',
                    }
                tifffile.imwrite(f'{outdir}/{outname}_{i1}.tiff',imdata,metadata=metadata)
        if extradims==2:
            for i1 in np.arange(datashape[0]):
                for i2 in np.arange(datashape[1]):
                    imdata=data[i1][i2]
                    parainfo=axespara[i1][i2]
                    perpinfo=axesperp[i1][i2]
                    metadata={
                        'Description':f'Image data identical to data saved in {hf.filename}',
                        'Xlimits': f'min {parainfo.min()}, max {parainfo.max()}',
                        'Ylimits': f'min {perpinfo.min()}, max {perpinfo.max()}',
                        }
                    tifffile.imwrite(f'{outdir}/{outname}_{i1}_{i2}.tiff',imdata,metadata=metadata)
                    
    def do_savedats(self,hf,Idata,qdata,tthdata):
        datashape=np.shape(Idata)
        extradims=len(datashape)-1
        outdir=hf.filename.strip('.hdf5')
        try:
            os.mkdir(outdir)
        except:
            print('directory already exists')
        metadata=f'Intensity data identical to data saved in {hf.filename}\n'
        outname=outdir.split('/')[-1]
        if extradims==0:

            intvals=Idata
            qvals=qdata
            tthetavals=tthdata
            outdf=pd.DataFrame({'Q_angstrom^-1':qvals,'Intensity':intvals,'two_theta':tthetavals})
            with open(f'{outdir}/{outname}.dat',"w") as f:
                f.write(metadata)
                outdf.to_csv(f,sep='\t',index=False)
        
        
        if extradims==1:
            for i1 in np.arange(datashape[0]):
                intvals=Idata[i1]
                qvals=qdata[i1]
                tthetavals=tthdata[i1]
                outdf=pd.DataFrame({'Q_angstrom^-1':qvals,'Intensity':intvals,'two_theta':tthetavals})
                with open(f'{outdir}/{outname}_{i1}.dat',"w") as f:
                    f.write(metadata)
                    outdf.to_csv(f,sep='\t',index=False)
        if extradims==2:
            for i1 in np.arange(datashape[0]):
                for i2 in np.arange(datashape[1]):                   
                    intvals=Idata[i1][i2] 
                    qvals=qdata[i1][i2]
                    tthetavals=tthdata[i1][i2]
                    outdf=pd.DataFrame({'Q_angstrom^-1':qvals,'Intensity':intvals,'two_theta_degree':tthetavals})
                    with open(f'{outdir}/{outname}_{i1}_{i2}.dat',"w") as f:
                        f.write(metadata)
                        outdf.to_csv(f,sep='\t',index=False)
    
    def pyfai_stat_exitangles_wrapper(self,args):
        current_experiment, index, scan, pyfaiponi, qmapbins,ivqbins = args
        return pyfai_stat_exitangles(current_experiment, index, scan, current_experiment.two_theta_start, pyfaiponi, qmapbins, ivqbins)
    
    def pyfai_stat_qmap_wrapper(self,args):
        current_experiment, index, scan, pyfaiponi, qmapbins,ivqbins = args
        return pyfai_stat_qmap(current_experiment, index, scan, current_experiment.two_theta_start, pyfaiponi, qmapbins,ivqbins)#
    
    def pyfai_stat_ivsq_wrapper(self,args):
        current_experiment, index, scan, pyfaiponi, qmapbins,ivqbins = args
        return  pyfai_stat_ivsq(current_experiment,index, scan, current_experiment.two_theta_start, pyfaiponi, qmapbins,ivqbins)
    
    def pyfai_static_exitangles(self,hf,scan,num_threads,pyfaiponi,ivqbins,qmapbins=0):
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
        self.load_curve_values(scan)
        
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        if self.setup=='DCD':
            tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        else:
            tthdirect=0
            
        self.two_theta_start=self.gammadata-tthdirect
        
        
        #calculate map bins if not specified using resolution of 0.01 degrees 
        
        if qmapbins==0:
            qmapbins=[800,800]
            
        scalegamma=1
        
        datacheck=('data' in list(scan.metadata.data_file.nx_detector))
        localpathcheck=('local_image_paths' in scan.metadata.data_file.__dict__.keys())
        if datacheck:
            scanlength=np.shape(scan.metadata.data_file.nx_detector.data[:,1,:])[0]
        elif localpathcheck:
            scanlength=len(scan.metadata.data_file.local_image_paths)
        else:
            scanlength=scan.metadata.data_file.scan_length
            
        print('starting process pool')
        all_maps=[]
        all_xlabels=[]
        all_ylabels=[]
        
        with Pool(processes=num_threads) as pool:
            
            print(f'started pool with num_threads={num_threads}')
            indices=np.arange(0,scanlength,scalegamma)
            input_list = [(self,index,scan,pyfaiponi,qmapbins,ivqbins) for index in indices]
            results=pool.map(self.pyfai_stat_exitangles_wrapper,input_list)
            maps=[result[0] for result in results]
            xlabels=[result[1] for result in results]
            ylabels=[result[2] for result in results]
            all_maps.append(maps)
            all_xlabels.append(xlabels)
            all_ylabels.append(ylabels)
            
            print('finished preparing chunked data')
            
        signal_shape=np.shape(scan.metadata.data_file.default_signal)
        outlist=[all_maps[0],all_xlabels[0],all_ylabels[0]]
        if len(signal_shape)>1:
            outlist=[self.reshape_to_signalshape(arr, signal_shape) for arr in outlist]
        
        dset=hf.create_group("horiz_vert_exit")
        dset.create_dataset("exit_angle_image",data=outlist[0])
        dset.create_dataset("exit_para",data=outlist[1])
        dset.create_dataset("exit_perp",data=outlist[2])
        dset.create_dataset("exit_perp_indices",data = [0,1,2])
        dset.create_dataset("exit_para_indices",data = [0,1,3])
        
                        
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)    
        if self.savetiffs==True:
            self.do_savetiffs(hf, outlist[0],outlist[1], outlist[2])
        
    
    def pyfai_static_qmap(self,hf,scan,num_threads,output_file_path,pyfaiponi,ivqbins,qmapbins=0):
        self.load_curve_values(scan)
        
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        if self.setup=='DCD':
            tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        else:
            tthdirect=0
            
        self.two_theta_start=self.gammadata-tthdirect
        qlimhor=self.calcqlim( 'hor')
        qlimver=self.calcqlim( 'vert')
        
        #calculate map bins if not specified using resolution of 0.01 degrees 
        
        if qmapbins==0:
            qstep=round(self.calcq(1.00,self.incident_wavelength)-\
                self.calcq(1.01,self.incident_wavelength),4)
            binshor=abs(round(((qlimhor[1]-qlimhor[0])/qstep)*1.05))
            binsver=abs(round(((qlimver[1]-qlimver[0])/qstep)*1.05))
            qmapbins=(binshor,binsver)
            
        scalegamma=1
        
        datacheck=('data' in list(scan.metadata.data_file.nx_detector))
        localpathcheck=('local_image_paths' in scan.metadata.data_file.__dict__.keys())
        if datacheck:
            scanlength=np.shape(scan.metadata.data_file.nx_detector.data[:,1,:])[0]
        elif localpathcheck:
            scanlength=len(scan.metadata.data_file.local_image_paths)
        else:
            scanlength=scan.metadata.data_file.scan_length
        
        print('starting process pool')
        all_maps=[]
        all_xlabels=[]
        all_ylabels=[]
        
        with Pool(processes=num_threads) as pool:
            
            print(f'started pool with num_threads={num_threads}')
            indices=np.arange(0,scanlength,scalegamma)
            input_list = [(self,index,scan,pyfaiponi,qmapbins,ivqbins) for index in indices]
            results=pool.map(self.pyfai_stat_qmap_wrapper,input_list)
            maps=[result[0] for result in results]
            xlabels=[result[1] for result in results]
            ylabels=[result[2] for result in results]
            all_maps.append(maps)
            all_xlabels.append(xlabels)
            all_ylabels.append(ylabels)
            
            print('finished preparing chunked data')
            
        signal_shape=np.shape(scan.metadata.data_file.default_signal)
        outlist=[all_maps[0],all_xlabels[0],all_ylabels[0]]
        if len(signal_shape)>1:
            outlist=[self.reshape_to_signalshape(arr, signal_shape) for arr in outlist]
        
        binset=hf.create_group("binoculars")
        binset.create_dataset("counts",data=outlist[0])
        binset.create_dataset("contributions",data=np.ones(np.shape(outlist[0])))
        axgroup=binset.create_group("axes",track_order=True)
        # print(np.shape(outlist[0]),outlist[1][0][0],outlist[1][0][-1])
        # print(np.shape(outlist[0]),outlist[2][0][0],outlist[2][0][-1])

        
        zlen=np.shape(outlist[0])[0]
        if zlen>1:
            axgroup.create_dataset("1_index",data=self.get_bin_axvals(np.arange(zlen),0),track_order=True)
        else:
            axgroup.create_dataset("1_index",data=[0.0,1.0,1.0,1.0,1.0,1.0],track_order=True)
        
        axgroup.create_dataset("2_q_perp",data=self.get_bin_axvals(outlist[1][0],1),track_order=True)    
        axgroup.create_dataset("3_q_para",data=self.get_bin_axvals(outlist[2][0],2),track_order=True)
        
        
        dset=hf.create_group("qpara_qperp")
        dset["qpara_qperp_image"]=h5py.SoftLink('/binoculars/counts')
        dset.create_dataset("map_para",data=outlist[1])
        dset.create_dataset("map_perp",data=outlist[2])
        dset.create_dataset("map_perp_indices",data = [0,1,2])
        dset.create_dataset("map_para_indices",data = [0,1,3])
        
        
                        
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)    
        if self.savetiffs==True:
            self.do_savetiffs(hf, outlist[0],outlist[1], outlist[2])
    
    def get_bin_axvals(self,data_in,ind):
        #print(data_in,type(data_in[0]))
        single_list=[np.int64,np.float64,int,float]
        if type(data_in[0]) in single_list:
            data=data_in
        else:
            data=data_in[0]
        startval=data[0]
        stopval=data[-1]
        stepval=data[1]-data[0]
        startind=int(np.floor(startval/stepval))
        stopind=int(startind+len(data)-1)
        return [ind,startval,stopval,stepval,float(startind),float(stopind)]
        
        

    def pyfai_static_ivsq(self,hf,scan,num_threads,output_file_path,pyfaiponi,ivqbins,qmapbins=0):
        self.load_curve_values(scan)
        
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        if self.setup=='DCD':
            tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        else:
            tthdirect=0
            
        self.two_theta_start=self.gammadata-tthdirect
        qlimhor=self.calcqlim( 'hor')
        qlimver=self.calcqlim( 'vert')
        
        #calculate map bins if not specified using resolution of 0.01 degrees 
        
        if qmapbins==0:
            qstep=round(self.calcq(1.00,self.incident_wavelength)-\
                self.calcq(1.01,self.incident_wavelength),4)
            binshor=abs(round(((qlimhor[1]-qlimhor[0])/qstep)*1.05))
            binsver=abs(round(((qlimver[1]-qlimver[0])/qstep)*1.05))
            qmapbins=(binshor,binsver)
                        
                
                    
        scalegamma=1
        
        datacheck=('data' in list(scan.metadata.data_file.nx_detector))
        localpathcheck=('local_image_paths' in scan.metadata.data_file.__dict__.keys())
        if datacheck:
            scanlength=np.shape(scan.metadata.data_file.nx_detector.data[:,1,:])[0]
        elif localpathcheck:
            scanlength=len(scan.metadata.data_file.local_image_paths)
        else:
            scanlength=scan.metadata.data_file.scan_length
            
                
        print('starting process pool')
        all_ints=[]
        all_two_ths=[]
        all_Qs=[]
        
        with Pool(processes=num_threads) as pool:
            
            print(f'started pool with num_threads={num_threads}')
            indices=np.arange(0,scanlength,scalegamma)
            input_list = [(self,index,scan,pyfaiponi,qmapbins,ivqbins) for index in indices]
            results=pool.map(self.pyfai_stat_ivsq_wrapper,input_list)
            intensities=[result[0] for result in results]
            two_th_vals=[result[1] for result in results]
            Q_vals=[result[2] for result in results]
            all_ints.append(intensities)
            all_two_ths.append(two_th_vals)
            all_Qs.append(Q_vals)
            
            print('finished preparing chunked data')
            
        signal_shape=np.shape(scan.metadata.data_file.default_signal)
        outlist=[all_ints[0],all_Qs[0],all_two_ths[0]]
        if len(signal_shape)>1:
            outlist=[self.reshape_to_signalshape(arr, signal_shape) for arr in outlist]
                            
        dset=hf.create_group("integrations")
        dset.create_dataset("Intensity",data=outlist[0])
        dset.create_dataset("Q_angstrom^-1",data=outlist[1])
        dset.create_dataset("2thetas",data=outlist[2])    
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)    
        if self.savedats==True:
            self.do_savedats(hf,outlist[0],outlist[1],outlist[2])
                
    def pyfai_moving_ivsq(self,hf,scan,num_threads,output_file_path,pyfaiponi,radrange,radstepval,qmapbins=0):
        
        
        self.load_curve_values(scan)
        
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        if self.setup=='DCD':
            tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        else:
            tthdirect=0
        self.two_theta_start=self.gammadata-tthdirect
        qlimhor=self.calcqlim( 'hor')
        qlimver=self.calcqlim( 'vert')
        #calculate map bins if not specified using resolution of 0.01 degrees 
        if qmapbins==0:
            qstep=round(self.calcq(1.00,self.incident_wavelength)-\
                self.calcq(1.01,self.incident_wavelength),4)
            binshor=abs(round(((qlimhor[1]-qlimhor[0])/qstep)*1.05))
            binsver=abs(round(((qlimver[1]-qlimver[0])/qstep)*1.05))
            qmapbins=(binshor,binsver)
        async_results = []
            # Make a pool on which we'll carry out the processing.
        locks = [Lock() for _ in range(num_threads)]
        start_time = time()
        scalegamma=1
        datacheck=('data' in list(scan.metadata.data_file.nx_detector))
        localpathcheck=('local_image_paths' in scan.metadata.data_file.__dict__.keys())
        if datacheck:
            scanlength=np.shape(scan.metadata.data_file.nx_detector.data[:,1,:])[0]
        elif localpathcheck:
            scanlength=len(scan.metadata.data_file.local_image_paths)
        else:
            scanlength=scan.metadata.data_file.scan_length
        nqbins=int(np.ceil((radrange[1]-radrange[0])/radstepval))
        shapeqi=(2,3,nqbins)
            
            
        shapecake=(2, 2, 2)
        shapeqpqp=(2,qmapbins[1],qmapbins[0])
        output_path=fr"{output_file_path}"
        cake_arrays=0
        qpqp_arrays=0
        print('starting process pool')
        with Pool(
            processes=num_threads,  # The size of our pool.
            initializer=init_pyfai_process_pool,  # Our pool's initializer.
            initargs=(locks,  # The initializer makes this lock global.
                  num_threads,  # Initializer makes num_threads global.
                  self.scans[0].metadata,
                  shapeqi,
                  #(shapeqi[0],int(shapeqi[1]/50)),
                  shapecake,
                  shapeqpqp,
                  output_path)
        ) as pool:
                    
            print(f'started pool with num_threads={num_threads}')
            for indices in chunk(np.arange(0,scanlength,scalegamma), num_threads):
                async_results.append(pool.apply_async(
                    pyfai_move_ivsq,
                    (self,indices,scan,shapecake,shapeqi,shapeqpqp,self.two_theta_start,pyfaiponi,radrange,radstepval,qmapbins)))
                #print(f'done  {indices[0]  - indices[1]} with {num_threads}\n')
            print('finished preparing chunked data')
            pyfai_qi_names=[]
            cake_names=[]
            qpqpmap_names=[]
            mapaxisinfo=[1,2,3,4]
            for result in async_results:
                shared_pyfai_qi_nameval=result.get()#
                if shared_pyfai_qi_nameval not in pyfai_qi_names:
                    pyfai_qi_names.append(shared_pyfai_qi_nameval)
                    
                if not result.successful():
                    raise ValueError(
                    "Could not carry out map for an unknown reason. "
                    "Probably one of the threads segfaulted, or something.")
                            
                            
            qi_mem=[SharedMemory(x) for x in pyfai_qi_names ]
            totalqi_arrays=np.array([np.ndarray(shape=shapeqi, dtype=np.float32, buffer=y.buf)[0]
                for y in qi_mem])
            totalcount_arrays=np.array([np.ndarray(shape=shapeqi, dtype=np.float32, buffer=y.buf)[1]
                for y in qi_mem])            
                        
            new_totalcount_arrays=np.sum(totalcount_arrays,axis=0)
            new_totalqi_arrays =np.sum(totalqi_arrays,axis=0)
            qi_array=np.divide(new_totalqi_arrays,new_totalcount_arrays, out=np.copy(new_totalqi_arrays), where=new_totalcount_arrays !=0.0)
                            
            pool.close()
            pool.join()
            for shared_mem in qi_mem:
                shared_mem.close()
                try:
                    shared_mem.unlink()
                except:
                    pass
                            
        end_time=time()
        #datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        
        dset=hf.create_group("integrations")
        dset.create_dataset("Intensity",data=qi_array[0])
        dset.create_dataset("Q_angstrom^-1",data=qi_array[1])
        dset.create_dataset("2thetas",data=qi_array[2])
                        
        if self.savedats==True:
            self.do_savedats(hf,qi_array[0],qi_array[1],qi_array[2])
        minutes=(end_time-start_time)/60
        print(f'total calculation took {minutes}  minutes')           
    
    def pyfai_moving_exitangles(self,hf,scan,num_threads,output_file_path,pyfaiponi,radrange,radstepval,qmapbins=0):
        self.load_curve_values(scan)
        
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        if self.setup=='DCD':
            tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        else:
            tthdirect=0
            
        self.two_theta_start=self.gammadata-tthdirect
        if qmapbins==0:
            qmapbins=[800,800]
        async_results = []
            # Make a pool on which we'll carry out the processing.
        locks = [Lock() for _ in range(num_threads)]
        start_time = time()
        
        scalegamma=1
        
        datacheck=('data' in list(scan.metadata.data_file.nx_detector))
        localpathcheck=('local_image_paths' in scan.metadata.data_file.__dict__.keys())
        if datacheck:
            scanlength=np.shape(scan.metadata.data_file.nx_detector.data[:,1,:])[0]
        elif localpathcheck:
            scanlength=len(scan.metadata.data_file.local_image_paths)
        else:
            scanlength=scan.metadata.data_file.scan_length
        nqbins=int(np.ceil((radrange[1]-radrange[0])/radstepval))
        shapeqi=(2,3,nqbins)
         
         
        shapecake=(2, 2, 2)
        shapeexhexv=(2,qmapbins[1],qmapbins[0])
        output_path=fr"{output_file_path}"
        exhexv_arrays=0
        print('starting process pool')
        with Pool(
            processes=num_threads,  # The size of our pool.
            initializer=init_pyfai_process_pool,  # Our pool's initializer.
            initargs=(locks,  # The initializer makes this lock global.
                  num_threads,  # Initializer makes num_threads global.
                  self.scans[0].metadata,
                  shapeqi,
                  #(shapeqi[0],int(shapeqi[1]/50)),
                  shapecake,
                  shapeexhexv,
                  output_path)
        ) as pool:
            
            print(f'started pool with num_threads={num_threads}')
            for indices in chunk(np.arange(0,scanlength,scalegamma), num_threads):
                async_results.append(pool.apply_async(
                    pyfai_move_exitangles,
                    (self,indices,scan,shapecake,shapeqi,shapeexhexv,self.two_theta_start,pyfaiponi,radrange,radstepval,qmapbins)))
                #print(f'done  {indices[0]  - indices[1]} with {num_threads}\n')
            print('finished preparing chunked data')        
            exhexvmap_names=[]
            mapaxisinfo=[1,2]
            for result in async_results:
                shared_exhexvmap_nameval,mapaxisinfo=result.get()#

                if shared_exhexvmap_nameval not in exhexvmap_names:
                    exhexvmap_names.append(shared_exhexvmap_nameval)
                    # Make sure that no error was thrown while mapping.
                if not result.successful():
                    raise ValueError(
                    "Could not carry out map for an unknown reason. "
                    "The threads may have segfaulted.")
            
            
            exhexv_mem=[SharedMemory(x) for x in exhexvmap_names]
            new_exhexv_arrays=np.array([np.ndarray(shape=shapeexhexv, dtype=np.float32, buffer=y.buf)[0]
                for y in exhexv_mem])
            new_exhexv_counts=np.array([np.ndarray(shape=shapeexhexv, dtype=np.float32, buffer=y.buf)[1]
                for y in exhexv_mem])
            
            exhexv_arrays=np.sum(new_exhexv_arrays,axis=0)
            exhexv_counts=np.sum(new_exhexv_counts,axis=0)
            #print(f'shape of exhexv_arrays = {np.shape(exhexv_arrays)}')
            exhexv_array=np.divide(exhexv_arrays,exhexv_counts, out=np.copy(exhexv_arrays), where=exhexv_counts !=0.0)
            #print(f'shape of exhexv_array  = {np.shape(exhexv_array)}')
                       
            pool.close()
            pool.join()

            for shared_mem in exhexv_mem:
                shared_mem.close()
                try:
                    shared_mem.unlink()
                except:
                    pass
    
        end_time=time()
        #datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        
        dset3=hf.create_group("horiz_vert_exit")
        dset3.create_dataset("exit_angle_image",data=exhexv_array)
        dset3.create_dataset("exit_para",data=mapaxisinfo[1])
        dset3.create_dataset("exit_para_unit",data=mapaxisinfo[3])
        dset3.create_dataset("exit_perp",data=-1*mapaxisinfo[0])#list(reversed(mapaxisinfo[0])))
        dset3.create_dataset("exit_perp_unit",data=mapaxisinfo[2]) 
        dset3.create_dataset("exit_perp_indices",data = [0,1,2])
        dset3.create_dataset("exit_para_indices",data = [0,1,3])
        
        if self.savetiffs==True:
            self.do_savetiffs(hf, exhexv_array,mapaxisinfo[1], mapaxisinfo[0])

        minutes=(end_time-start_time)/60
        print(f'total exit angle map calculation took {minutes}  minutes')
        return mapaxisinfo             
        
    def pyfai_moving_qmap(self,hf,scanlist,num_threads,output_file_path,pyfaiponi,radrange,radstepval,qmapbins=0):            
        
        qlimitsout=[0,0,0,0]
        lowerinds=[0,3]
        upperinds=[1,2]
        qpqp_array_total=0
        qpqp_counts_total=0
        if type(scanlist)==Scan:
            scanlistnew=[scanlist]
        else:
            scanlistnew=scanlist

        for scan in scanlistnew:
            self.load_curve_values(scan)
            dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
            if self.setup=='DCD':
                tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
            else:
                tthdirect=0
            print(f'gamma shape = {np.shape(self.gammadata)}')
            print(f'tth shape = {np.shape(tthdirect)}')
            self.two_theta_start=self.gammadata-tthdirect
            qlimhor=self.calcqlim( 'hor',vertsetup=(self.setup=='vertical'))
            qlimver=self.calcqlim( 'vert',vertsetup=(self.setup=='vertical'))
            qlimits = [qlimhor[0], qlimhor[1],qlimver[0], qlimver[1]]
            for i,val in enumerate(qlimits):
                if (i in lowerinds)&(val<qlimitsout[i]):
                    qlimitsout[i]=val
                if (i in upperinds)&(val>qlimitsout[i]):
                    qlimitsout[i]=val
            print(f'new qlimits = {qlimitsout}')


        for scan in scanlistnew:
            self.load_curve_values(scan)
            dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
            if self.setup=='DCD':
                tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
            else:
                tthdirect=0
                
            self.two_theta_start=self.gammadata-tthdirect
            # qlimhor=self.calcqlim( 'hor')
            # qlimver=self.calcqlim( 'vert')
                    
            #calculate map bins if not specified using resolution of 0.01 degrees 
                
            if qmapbins==0:
                qstep=round(self.calcq(1.00,self.incident_wavelength)-\
                    self.calcq(1.01,self.incident_wavelength),4)
                binshor=abs(round(((qlimhor[1]-qlimhor[0])/qstep)*1.05))
                binsver=abs(round(((qlimver[1]-qlimver[0])/qstep)*1.05))
                qmapbins=(binshor,binsver)
                            
                
            async_results = []
                # Make a pool on which we'll carry out the processing.
            locks = [Lock() for _ in range(num_threads)]
            start_time = time()
            
            scalegamma=1
            
            datacheck=('data' in list(scan.metadata.data_file.nx_detector))
            localpathcheck=('local_image_paths' in scan.metadata.data_file.__dict__.keys())
            if datacheck:
                scanlength=np.shape(scan.metadata.data_file.nx_detector.data[:,1,:])[0]
            elif localpathcheck:
                scanlength=len(scan.metadata.data_file.local_image_paths)
            else:
                scanlength=scan.metadata.data_file.scan_length
                
            nqbins=int(np.ceil((radrange[1]-radrange[0])/radstepval))
            shapeqi=(2,3,nqbins)


            shapecake=(2, 2, 2)
            shapeqpqp=(2,qmapbins[1],qmapbins[0])
            output_path=fr"{output_file_path}"
            qpqp_arrays=0
            print('starting process pool')
            with Pool(
                processes=num_threads,  # The size of our pool.
                initializer=init_pyfai_process_pool,  # Our pool's initializer.
                initargs=(locks,  # The initializer makes this lock global.
                    num_threads,  # Initializer makes num_threads global.
                    self.scans[0].metadata,
                    shapeqi,
                    #(shapeqi[0],int(shapeqi[1]/50)),
                    shapecake,
                    shapeqpqp,
                    output_path)) as pool:
                
                print(f'started pool with num_threads={num_threads}')
                for indices in chunk(np.arange(0,scanlength,scalegamma), num_threads):
                    async_results.append(pool.apply_async(
                        pyfai_move_qmap,
                        (self,indices,scan,shapecake,shapeqi,shapeqpqp,self.two_theta_start,pyfaiponi,radrange,radstepval,qmapbins,qlimitsout)))
                    #print(f'done  {indices[0]  - indices[1]} with {num_threads}\n')
                print('finished preparing chunked data')

                qpqpmap_names=[]
                mapaxisinfo=[1,2]
                for result in async_results:
                    shared_qpqpmap_nameval,mapaxisinfo=result.get()#

                    if shared_qpqpmap_nameval not in qpqpmap_names:
                        qpqpmap_names.append(shared_qpqpmap_nameval)
                        # Make sure that no error was thrown while mapping.
                    if not result.successful():
                        raise ValueError(
                        "Could not carry out map for an unknown reason. "
                        "The threads may have segfaulted.")
                
                
                qpqp_mem=[SharedMemory(x) for x in qpqpmap_names]
                new_qpqp_arrays=np.array([np.ndarray(shape=shapeqpqp, dtype=np.float32, buffer=y.buf)[0]
                    for y in qpqp_mem])
                new_qpqp_counts=np.array([np.ndarray(shape=shapeqpqp, dtype=np.float32, buffer=y.buf)[1]
                    for y in qpqp_mem])
                
                
                qpqp_arrays=np.sum(new_qpqp_arrays,axis=0)
                qpqp_counts=np.sum(new_qpqp_counts,axis=0)
                #print(f'shape of qpqp_arrays = {np.shape(qpqp_arrays)}')
                qpqp_array=np.divide(qpqp_arrays,qpqp_counts, out=np.copy(qpqp_arrays), where=qpqp_counts !=0.0)
                #print(f'shape of qpqp_array  = {np.shape(qpqp_array)}')
                        
                pool.close()
                pool.join()

                for shared_mem in qpqp_mem:
                    shared_mem.close()
                    try:
                        shared_mem.unlink()
                    except:
                        pass
            print(f'new max count={qpqp_counts.max()}')
            qpqp_array_total+=qpqp_arrays
            qpqp_counts_total+=qpqp_counts
            print(f'total max count = {qpqp_counts_total.max()}')
        
        qpqp_map_norm=np.divide(qpqp_array_total,qpqp_counts_total, out=np.copy(qpqp_array_total), where=qpqp_counts_total !=0.0)
        end_time=time()
        #datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        
        dset3=hf.create_group("qpara_qperp")
        dset3.create_dataset("qpara_qperp_sum",data=qpqp_array_total)
        dset3.create_dataset("qpara_qperp_counts",data=qpqp_counts_total)
        dset3.create_dataset("qpara_qperp_image",data=qpqp_map_norm)
        dset3.create_dataset("map_para",data=mapaxisinfo[1])
        dset3.create_dataset("map_para_unit",data=mapaxisinfo[3])
        dset3.create_dataset("map_perp",data=-1*mapaxisinfo[0])#list(reversed(mapaxisinfo[0])))
        dset3.create_dataset("map_perp_unit",data=mapaxisinfo[2]) 
        dset3.create_dataset("map_perp_indices",data = [0,1,2])
        dset3.create_dataset("map_para_indices",data = [0,1,3])
        
        if self.savetiffs==True:
            self.do_savetiffs(hf, qpqp_array,mapaxisinfo[1], mapaxisinfo[0])

        minutes=(end_time-start_time)/60
        print(f'total calculation took {minutes}  minutes')
        return mapaxisinfo        
    
    
    
    def gamdel2rots(self,gamma,delta):
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
        rotdelta=R.from_euler('y', -delta, degrees=True)
        rotgamma=R.from_euler('z',gamma,degrees=True)
        totalrot=rotgamma*rotdelta
        fullrot=np.identity(4)
        fullrot[0:3,0:3]=totalrot.as_matrix()
        vals=tf.euler_from_matrix(fullrot,'rxyz')
        rots=vals[2],-vals[1],vals[0]
        return rots
        
    def load_curve_values(self,scan):
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
        p2mnames=['pil2stats','p2r','pil2roi']
        self.pixel_size=scan.metadata.diffractometer.data_file.pixel_size
        self.entry=scan.metadata.data_file.nx_entry
        
        self.detector_distance=scan.metadata.get_detector_distance(0)
        # else:
        #     self.detector_distance=scan.metadata.diffractometer.data_file.detector_distance
        self.incident_wavelength= 1e-10*scan.metadata.incident_wavelength
        try:
            self.gammadata=np.array( self.entry.instrument.diff1gamma.value_set).ravel()
        except:
            self.gammadata=np.array( self.entry.instrument.diff1gamma.value).ravel()
        #self.deltadata=np.array( self.entry.instrument.diff1delta.value)
        try:
            self.deltadata=np.array( self.entry.instrument.diff1delta.value_set).ravel()
        except:
            self.deltadata=np.array( self.entry.instrument.diff1delta.value).ravel()
            
        if self.setup=='DCD':
            self.dcdrad=np.array( self.entry.instrument.dcdc2rad.value)
            self.dcdomega=np.array( self.entry.instrument.dcdomega.value)
            self.projectionx=1e-3* self.dcdrad*np.cos(np.radians(self.dcdomega))
            self.projectiony=1e-3* self.dcdrad*np.sin(np.radians(self.dcdomega))
            dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance[0]
            self.dcd_incdeg=np.degrees(np.arctan(self.projectiony/(np.sqrt(np.square(self.projectionx)+np.square(dcd_sample_dist)))))
            self.incident_angle=self.dcd_incdeg
            self.deltadata+=self.dcd_incdeg
        elif (scan.metadata.data_file.is_eh1)&(self.setup!='DCD'):
            self.incident_angle=scan.metadata.data_file.chi
        elif (scan.metadata.data_file.is_eh2):
            self.incident_angle=scan.metadata.data_file.alpha
        else:
            self.incident_angle=[0]
        if scan.metadata.data_file.detector_name in p2mnames:
            self.deltadata=0

        self.imshape=scan.metadata.data_file.image_shape
        self.beam_centre=scan.metadata.beam_centre
        self.rotval=round(scan.metadata.data_file.det_rot)

    def createponi(self,outpath,image2dshape,beam_centre=0,offset=0):
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
        ponioutpath=fr'{outpath}/fast_rsm_{datetime_str}.poni'
        f=open(ponioutpath,'w')
        f.write('# PONI file created by fast_rsm\n#\n')
        f.write('poni_version: 2\n')
        f.write('Detector: Detector\n')
        f.write('Detector_config: {"pixel1":')
        f.write(f'{self.pixel_size}, "pixel2": {self.pixel_size}, "max_shape": [{image2dshape[0]}, {image2dshape[1]}]') 
        f.write('}\n')
        f.write(f'Distance: {self.detector_distance}\n')
        if beam_centre==0:
            poni1=(image2dshape[0]-offset)*self.pixel_size
            poni2=image2dshape[1]*self.pixel_size
        elif (offset==0):#&(self.setup!='vertical'):
            poni1=(beam_centre[0])*self.pixel_size
            poni2=beam_centre[1]*self.pixel_size
        # elif (offset==0)&(self.setup=='vertical'):
        #     poni1=beam_centre[1]*self.pixel_size
        #     poni2=(image2dshape[0]-beam_centre[0])*self.pixel_size
            
        f.write(f'Poni1: {poni1}\n')
        f.write(f'Poni2: {poni2}\n')
        f.write('Rot1: 0.0\n')
        f.write('Rot2: 0.0\n')
        f.write('Rot3: 0.0\n')
        f.write(f'Wavelength: {self.incident_wavelength}')
        f.close()
        return ponioutpath
    
    def save_projection(self,hf,projected2d,twothetas,Qangs,intensities,config):
        dset=hf.create_group("projection")
        dset.create_dataset("projection_2d",data=projected2d[0])
        dset.create_dataset("config",data=str(config))

        dset=hf.create_group("integrations")
        dset.create_dataset("2thetas",data=twothetas)
        dset.create_dataset("Q_angstrom^-1",data=Qangs)
        dset.create_dataset("Intensity",data=intensities)

    def save_integration(self,hf,twothetas,Qangs,intensities,configs,scan=0):
        dset=hf.create_group("integrations")
        dset.create_dataset("configs",data=str(configs))
        dset.create_dataset("2thetas",data=twothetas)
        dset.create_dataset("Q_angstrom^-1",data=Qangs)
        dset.create_dataset("Intensity",data=intensities)
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)
        if self.savedats==True:
            self.do_savedats(hf,intensities,Qangs,twothetas)
    
    def save_qperp_qpara(self,hf,qperp_qpara_map,scan=0):
        dset=hf.create_group("qperp_qpara")
        dset.create_dataset("images",data=qperp_qpara_map[0])
        dset.create_dataset("qpararanges",data=qperp_qpara_map[1])
        dset.create_dataset("qperpranges",data=qperp_qpara_map[2])
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan) 
                
        if self.savetiffs==True:
            self.do_savetiffs(hf, qperp_qpara_map[0],qperp_qpara_map[1], qperp_qpara_map[2])

    def save_config_variables(self,hf,joblines,pythonlocation,globalvals):
        config_group=hf.create_group('i07configuration')
        configlist=['setup','experimental_hutch', 'using_dps','beam_centre','detector_distance','dpsx_central_pixel','dpsy_central_pixel','dpsz_central_pixel',\
                    'local_data_path','local_output_path','output_file_size','save_binoculars_h5','map_per_image','volume_start','volume_step','volume_stop',\
                    'load_from_dat', 'edfmaskfile','specific_pixels','mask_regions','process_outputs','scan_numbers']
        for name in configlist :
            if name in globalvals:
                outval=globalvals[f'{name}']
                if outval==None:
                    outval='None'
                config_group.create_dataset(f"{name}",data=outval)
        if 'ubinfo' in globalvals:
            for i,coll in enumerate(globalvals['ubinfo']):
                ubgroup=config_group.create_group(f'ubinfo_{i+1}')
                ubgroup.create_dataset(f'lattice_{i+1}',data=coll['diffcalc_lattice'])
                ubgroup.create_dataset(f'u_{i+1}',data=coll['diffcalc_u'])
                ubgroup.create_dataset(f'ub_{i+1}',data=coll['diffcalc_ub'])
                
        config_group.create_dataset('joblines',data=joblines)
        config_group.create_dataset('python_location',data=pythonlocation)
    
    def reshape_to_signalshape(self,arr,signal_shape):
        testsize=int(np.prod(signal_shape))-np.shape(arr)[0]

        fullshape=signal_shape+np.shape(arr)[1:]
        if testsize==0:
            return np.reshape(arr,fullshape)
        else:
            extradata=np.zeros((testsize,)+(np.shape(arr)[1:]))
            outarr=np.concatenate((arr,extradata))
            return np.reshape(outarr,fullshape)
       
    def save_scan_field_values(self,hf,scan):
        try:
            rank=scan.metadata.data_file.diamond_scan.scan_rank.nxdata
            fields=scan.metadata.data_file.diamond_scan.scan_fields
            scanned=[x.decode('utf-8').split('.')[0] for x in fields[:rank].nxdata]
            scannedvalues=[np.unique(scan.metadata.data_file.nx_instrument[field].value )for field in scanned]
            scannedvaluesout=[scannedvals[~np.isnan(scannedvals)] for scannedvals in scannedvalues]
        except:
            scanned,scannedvaluesout=None,None
            
        dset=hf.create_group("scanfields")
        if scan!=0:
            if scanned!=None:
                for i, field in enumerate(scanned):
                    dset.create_dataset(f"dim{i}_{field}",data=scannedvaluesout[i])
                    
                    
    #============================================
    #================================================
    #DEPRECATED functions no longer in use due to move over to pyFAI
    def projectimage(self, scan,imnum,im1gammas):

        """
        DEPRECATED old function used to calculate  projected image before move over to using pyFAI

        Parameters
        ----------
        scan : TYPE
            DESCRIPTION.
        imnum : TYPE
            DESCRIPTION.
        im1gammas : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        data=scan.load_image(imnum).data
        
        #cropshape,cropdata,cropbc,gamshifts,delshifts=calc_cropshifts(xylimits,data,self.degperpix,two_theta_start,originalbc)
        if imnum==0:
            imgammas=im1gammas
    
        else:
            imgammas=im1gammas+(imnum*self.gammastep)
         
            
        self.imcounts=np.zeros(self.projshape)
        for j in np.arange(self.imshape[1]):
            gamma=imgammas[0,j]
            if j==0:
                gammaprevious=0
            else:
                gammaprevious=imgammas[0,j-1]
            horlow,horupp=self.calcupplow(gamma)
            avgamma=(gamma +gammaprevious)/2
            distpixhor=abs(j-self.beam_centre[1])*self.pixel_size
            dist_to_pix=np.sqrt(np.square(distpixhor)+np.square(self.detector_distance))
            dist2D=self.detector_distance/np.cos(np.radians(avgamma)) 
            ratiodist=dist2D/dist_to_pix
            pixvals=data[:,j]
            self.nonenans=np.where(~np.isnan(pixvals))[0]
            for ind in self.nonenans:
                
                deltaind=(ind-self.beam_centre[0])*-1
                vertstart=int(np.floor(deltaind*(ratiodist)))+self.vertoffset
                vertend=int(np.ceil((deltaind+1)*(ratiodist)))+self.vertoffset
                totalpixels=(1+vertend-vertstart)*(1+horupp-horlow)
                self.project2d[self.projshape[0]-vertend:self.projshape[0]-vertstart,-horupp:-horlow]+=data[ind][j]/totalpixels
                self.imcounts[self.projshape[0]-vertend:self.projshape[0]-vertstart,-horupp:-horlow]+=1

        self.counts+=self.imcounts
    def pyfai_static_qmap_qvsI_wrapper(self,args):
        """
        DEPRECATED - wrapper used for calculation of both q perp Vs qpara map and I Vs Q plot

        Parameters
        ----------
        args : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        current_experiment, index, scan, pyfaiponi, qmapbins,ivqbins = args
        return pyfai_qmap_qvsI(current_experiment,index, scan, current_experiment.two_theta_start, pyfaiponi, qmapbins,ivqbins)
    #@profile
    def pyfaidiffractometer(self,hf,scan,num_threads,output_file_path,pyfaiponi,radrange,radstepval,qmapbins=0):
        """
        DEPRECATED - old projection method before using pyFAI functions

        Parameters
        ----------
        hf : TYPE
            DESCRIPTION.
        scan : TYPE
            DESCRIPTION.
        num_threads : TYPE
            DESCRIPTION.
        output_file_path : TYPE
            DESCRIPTION.
        pyfaiponi : TYPE
            DESCRIPTION.
        radrange : TYPE
            DESCRIPTION.
        radstepval : TYPE
            DESCRIPTION.
        qmapbins : TYPE, optional
            DESCRIPTION. The default is 0.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        mapaxisinfo : TYPE
            DESCRIPTION.

        """
        self.load_curve_values(scan)
        
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        if self.setup=='DCD':
            tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        else:
            tthdirect=0
        self.two_theta_start=self.gammadata-tthdirect
        qlimhor=self.calcqlim( 'hor')
        qlimver=self.calcqlim( 'vert')
        #calculate map bins if not specified using resolution of 0.01 degrees 
        if qmapbins==0:
            qstep=round(self.calcq(1.00,self.incident_wavelength)-\
                self.calcq(1.01,self.incident_wavelength),4)
            binshor=abs(round(((qlimhor[1]-qlimhor[0])/qstep)*1.05))
            binsver=abs(round(((qlimver[1]-qlimver[0])/qstep)*1.05))
            qmapbins=(binshor,binsver)
        async_results = []
            # Make a pool on which we'll carry out the processing.
        locks = [Lock() for _ in range(num_threads)]
        start_time = time()
        scalegamma=1
        try:
            scanlength=len(scan.metadata.data_file.local_image_paths)
        except:
            scanlength=scan.metadata.data_file.scan_length
        nqbins=int(np.ceil((radrange[1]-radrange[0])/radstepval))
        shapeqi=(2,3,nqbins)


        shapecake=(2, 360, 1800)
        shapeqpqp=(2,qmapbins[1],qmapbins[0])
        output_path=fr"{output_file_path}"
        cake_arrays=0
        qpqp_arrays=0
        print('starting process pool')
        with Pool(
            processes=num_threads,  # The size of our pool.
            initializer=init_pyfai_process_pool,  # Our pool's initializer.
            initargs=(locks,  # The initializer makes this lock global.
                  num_threads,  # Initializer makes num_threads global.
                  self.scans[0].metadata,
                  shapeqi,
                  #(shapeqi[0],int(shapeqi[1]/50)),
                  shapecake,
                  shapeqpqp,
                  output_path)
        ) as pool:
            
            print(f'started pool with num_threads={num_threads}')
            for indices in chunk(np.arange(0,scanlength,scalegamma), num_threads):
                async_results.append(pool.apply_async(
                    pyfaicalcint,
                    (self,indices,scan,shapecake,shapeqi,shapeqpqp,self.two_theta_start,pyfaiponi,radrange,radstepval,qmapbins)))
                #print(f'done  {indices[0]  - indices[1]} with {num_threads}\n')
            print('finished preparing chunked data')
            pyfai_qi_names=[]
            cake_names=[]
            qpqpmap_names=[]
            mapaxisinfo=[1,2,3,4]
            for result in async_results:
                shared_pyfai_qi_nameval, shared_cake_nameval,shared_qpqpmap_nameval,cakeaxisinfo,mapaxisinfo=result.get()#
                if shared_pyfai_qi_nameval not in pyfai_qi_names:
                    pyfai_qi_names.append(shared_pyfai_qi_nameval)
                if shared_cake_nameval not in cake_names:
                    cake_names.append(shared_cake_nameval)
                if shared_qpqpmap_nameval not in qpqpmap_names:
                    qpqpmap_names.append(shared_qpqpmap_nameval)
                    # Make sure that no error was thrown while mapping.
                if not result.successful():
                    raise ValueError(
                    "Could not carry out map for an unknown reason. "
                    "Probably one of the threads segfaulted, or something.")
            
            
            qi_mem=[SharedMemory(x) for x in pyfai_qi_names ]
            totalqi_arrays=np.array([np.ndarray(shape=shapeqi, dtype=np.float32, buffer=y.buf)[0]
                for y in qi_mem])
            totalcount_arrays=np.array([np.ndarray(shape=shapeqi, dtype=np.float32, buffer=y.buf)[1]
                for y in qi_mem])            
            
            cake_mem=[SharedMemory(x) for x in cake_names]
            new_cake_arrays = np.array([np.ndarray(shape=shapecake, dtype=np.float32, buffer=y.buf)[0]
                for y in cake_mem])
            new_count_arrays = np.array([np.ndarray(shape=shapecake, dtype=np.float32, buffer=y.buf)[1]
                for y in cake_mem])
            
            qpqp_mem=[SharedMemory(x) for x in qpqpmap_names]
            new_qpqp_arrays=np.array([np.ndarray(shape=shapeqpqp, dtype=np.float32, buffer=y.buf)[0]
                for y in qpqp_mem])
            new_qpqp_counts=np.array([np.ndarray(shape=shapeqpqp, dtype=np.float32, buffer=y.buf)[1]
                for y in qpqp_mem])
            
            #print(f'shape of new_qpqp_array = {np.shape(new_qpqp_arrays)}')
            count_arrays=np.sum(new_count_arrays,axis=0)
            cake_arrays =np.sum(new_cake_arrays,axis=0)
            cake_array=np.divide(cake_arrays,count_arrays, out=np.copy(cake_arrays), where=count_arrays !=0.0)
            
            qpqp_arrays=np.sum(new_qpqp_arrays,axis=0)
            qpqp_counts=np.sum(new_qpqp_counts,axis=0)
            #print(f'shape of qpqp_arrays = {np.shape(qpqp_arrays)}')
            qpqp_array=np.divide(qpqp_arrays,qpqp_counts, out=np.copy(qpqp_arrays), where=qpqp_counts !=0.0)
            #print(f'shape of qpqp_array  = {np.shape(qpqp_array)}')
            
            new_totalcount_arrays=np.sum(totalcount_arrays,axis=0)
            new_totalqi_arrays =np.sum(totalqi_arrays,axis=0)
            qi_array=np.divide(new_totalqi_arrays,new_totalcount_arrays, out=np.copy(new_totalqi_arrays), where=new_totalcount_arrays !=0.0)
            
            pool.close()
            pool.join()
            for shared_mem in qi_mem:
                shared_mem.close()
                try:
                    shared_mem.unlink()
                except:
                    pass
            for shared_mem in cake_mem:
                shared_mem.close()
                try:
                    shared_mem.unlink()
                except:
                    pass
            for shared_mem in qpqp_mem:
                shared_mem.close()
                try:
                    shared_mem.unlink()
                except:
                    pass
    
        end_time=time()
        #datetime_str = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

        dset=hf.create_group("integrations")
        dset.create_dataset("Intensity",data=qi_array[0])
        dset.create_dataset("Q_angstrom^-1",data=qi_array[1])
        dset.create_dataset("2thetas",data=qi_array[2])

        
        dset2=hf.create_group("caked image")
        dset2.create_dataset("cakeimage",data=cake_array)
        dset2.create_dataset("cake_azimuthal",data=cakeaxisinfo[0])
        dset2.create_dataset("cake_azimuthal_unit",data=cakeaxisinfo[2])
        dset2.create_dataset("cake_radial",data=cakeaxisinfo[1])
        dset2.create_dataset("cake_radial_unit",data=cakeaxisinfo[3])  
        
        dset3=hf.create_group("qpara_qperp")
        dset3.create_dataset("qpara_qperp_image",data=qpqp_array)
        dset3.create_dataset("map_para",data=mapaxisinfo[1])
        dset3.create_dataset("map_para_unit",data=mapaxisinfo[3])
        dset3.create_dataset("map_perp",data=-1*mapaxisinfo[0])
        dset3.create_dataset("map_perp_unit",data=mapaxisinfo[2]) 
        
        if self.savetiffs==True:
            self.do_savetiffs(hf, qpqp_array,mapaxisinfo[1], mapaxisinfo[0])
        if self.savedats==True:
            self.do_savedats(hf,qi_array[0],qi_array[1],qi_array[2])
        minutes=(end_time-start_time)/60
        print(f'total calculation took {minutes}  minutes')
        return mapaxisinfo
    def curved_to_2d(self,scan):
        """
        DEPRECATED - old method used before using pyFAI functions

        Parameters
        ----------
        scan : TYPE
            DESCRIPTION.

        Returns
        -------
        projected_data : TYPE
            DESCRIPTION.

        """
        self.load_curve_values(scan)
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        if self.setup=='DCD':
            gammadirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        else:
            gammadirect=0
        self.two_theta_start=self.gammadata-gammadirect
        self.twothetastart=self.two_theta_start
        self.projshape=self.calc_projected_size(self.two_theta_start)
        self.degperpix=np.degrees(np.arctan(self.pixel_size/(self.detector_distance)))
    
        self.gammastep=(self.two_theta_start[-1]-self.two_theta_start[0])/(len(self.two_theta_start)-1)

        self.gamma2d=np.degrees(np.arctan((np.arange(self.projshape[1])*self.pixel_size)/self.detector_distance))
        

        self.vertoffset=int((self.imshape[0]-self.beam_centre[0])*self.maxratiodist)

        self.project2d=np.zeros(self.projshape)
        self.counts=np.zeros(self.projshape)
        try:
            scanlength=len(scan.metadata.data_file.local_image_paths)
        except:
            scanlength=scan.metadata.data_file.scan_length
        
        gamshifts=-1*(np.arange(self.imshape[1])-self.beam_centre[1])
        delshifts=np.abs(np.arange(self.imshape[0])-self.beam_centre[1])
        im1gammas=np.zeros(self.imshape)
        for col in np.arange(np.shape(im1gammas)[1]):
            O=gamshifts[col]*self.pixel_size
            tantheta=O/self.detector_distance
            im1gammas[:,col]=self.two_theta_start[0]+(np.degrees(np.arctan(tantheta)))
        #self.imgamma=im1gammas
        print(f'projecting {scanlength} images   completed images:  ')
        imstep=1#int(np.floor(scanlength/50))
        print(scanlength,imstep)
        #for imnum in np.arange(0,scanlength,imstep):
        for imnum in np.arange(10,12,1):
            self.projectimage(scan, imnum,im1gammas)
            #if (imnum+1)%10==0:
               # print('\n')
            #print(fr'{imnum+1}','\r', end='')
            #print(f' projected image {outstring}','\r', end='')
        norm2d=np.divide(self.project2d,self.counts,where=self.counts!=0)
        projected_data=[norm2d,self.counts,self.vertoffset]
        return projected_data
    
    def pyfai1D(self,imagespath,maskpath,ponipath,outpath,scan,projected2d=None,gammastep=0.005):
        """
        DEPRECATED

        Parameters
        ----------
        imagespath : TYPE
            DESCRIPTION.
        maskpath : TYPE
            DESCRIPTION.
        ponipath : TYPE
            DESCRIPTION.
        outpath : TYPE
            DESCRIPTION.
        scan : TYPE
            DESCRIPTION.
        projected2d : TYPE, optional
            DESCRIPTION. The default is None.
        gammastep : TYPE, optional
            DESCRIPTION. The default is 0.005.

        Returns
        -------
        outlist : TYPE
            DESCRIPTION.

        """
        
        #images=scan.metadata.data_file.local_image_paths
        if projected2d==None:
            try:
                scanlength=len(scan.metadata.data_file.local_image_paths)
            except:
                scanlength=scan.metadata.data_file.scan_length
        else:
            scanlength=1
        #tiflist=[file.split(f'{imagespath}')[-1] for file in images]
        twothetas=[]
        intensities=[]
        Qangs=[]
        configs=[]
        try:
            gammavalues=np.array(scan.metadata.diffractometer.data_file.nx_instrument.diff1gamma.value_set)
        except:
            gammavalues=np.array(scan.metadata.diffractometer.data_file.nx_instrument.diff1gamma.value)
        gammarange=gammavalues.max()-gammavalues.min()
        if gammarange<3:
            bins=1000
        else:
            bins=gammarange/gammastep 
        
        chivals=[0]
        hryvals=[0]
        signal_shape=np.shape(scan.metadata.data_file.default_signal)

        if 'diff1chi' in scan.metadata.data_file.nx_instrument.keys():
            chivals=scan.metadata.data_file.nx_instrument.diff1chi.value.nxdata
            if len(np.shape(chivals))>1:
                chivals=np.reshape(chivals,np.prod(np.shape(chivals)))
            #-\
            #scan.metadata.data_file.nx_instrument.diff1chioffset.value.nxdata
        if 'hry' in scan.metadata.data_file.nx_instrument.keys():
            hryvals=scan.metadata.data_file.nx_instrument.hry.value.nxdata
            if len(np.shape(hryvals))>1:
                hryvals=np.reshape(hryvals,np.prod(np.shape(hryvals)))
        for i in np.arange(scanlength):

            if (projected2d==None) or (projected2d==1):  
               
                # fname=scan.
                # img = fabio.open(fr'{imagespath}/{fname}')
                img_array = scan.load_image(i).data
                maskimg = fabio.open(maskpath)
                mask = maskimg.data
                
    
            else:
                img_array=projected2d[0]
                mask=np.less_equal(projected2d[1],0)
            
            ai = pyFAI.load(ponipath)
            hryval=0
            chival=0
            if np.isscalar(hryvals):
                hryval=hryvals
            elif len(hryvals)>1:
                hryval=hryvals[i]
            if np.isscalar(chivals):
                chival=chivals
            elif len(chivals)>1:
                chival=chivals[i]
            ai.rot2=np.radians(-hryval)
            
            tth,I = ai.integrate1d_ng(img_array,
                                    bins,
                                    mask=mask,
                                    unit="2th_deg",polarization_factor=1)
            

            Q,I = ai.integrate1d_ng(img_array,
                                bins,
                                mask=mask,
                               unit="q_A^-1",polarization_factor=1)



            #outdata={'2theta':tth,'Q_angstrom^-1':Q,'Intensity':I}
            Qangs.append(Q)
            intensities.append(I)
            twothetas.append(tth)
            #outdatas.append(outdata)
            configs.append(ai.get_config())
        if len(signal_shape)>1:
            out_qangs=self.reshape_to_signalshape(Qangs,signal_shape)
            out_twothetas=self.reshape_to_signalshape(twothetas, signal_shape)
            out_intensities=self.reshape_to_signalshape(intensities,signal_shape)
            out_configs=self.reshape_to_signalshape(configs, signal_shape)
            outlist=out_twothetas,out_qangs,out_intensities,out_configs
        else:
            outlist=twothetas,Qangs,intensities,configs

        return outlist
    
    def calc_mapnorm_ranges(self,pdata,qvals,mapnorms,rangeqparas,rangeqperps,mapints):
        """
        DEPRECATED

        Parameters
        ----------
        pdata : TYPE
            DESCRIPTION.
        qvals : TYPE
            DESCRIPTION.
        mapnorms : TYPE
            DESCRIPTION.
        rangeqparas : TYPE
            DESCRIPTION.
        rangeqperps : TYPE
            DESCRIPTION.
        mapints : TYPE
            DESCRIPTION.

        Returns
        -------
        mapnorms : TYPE
            DESCRIPTION.
        rangeqparas : TYPE
            DESCRIPTION.
        rangeqperps : TYPE
            DESCRIPTION.

        """
        qxvalues=np.reshape(qvals[:,:,0],np.size(pdata))
        qyvalues=np.reshape(qvals[:,:,1],np.size(pdata))
        qzvalues=np.reshape(qvals[:,:,2],np.size(pdata))
        
        qperp=qzvalues
        qpara=np.sqrt(np.square(qxvalues)+np.square(qyvalues))*np.copysign(1,np.sign(qxvalues))
        
        perpstep=0.005
        parastep=0.005

        rangeqperp=np.linspace(qperp.min(),qperp.max(),int((qperp.max()-qperp.min())/perpstep))
        rangeqpara=np.linspace(qpara.min(),qpara.max(),int((qpara.max()-qpara.min())/parastep))
        qperpbin=[int(val)-1 for val in (qperp-qperp.min())/perpstep]
        qparabin=[int(val)-1 for val in (qpara-qpara.min())/parastep]
        
        
        qmap=np.zeros([len(rangeqperp),len(rangeqpara)])
        counts=np.zeros([len(rangeqperp),len(rangeqpara)])
        for i in np.arange(len(qparabin)):
            qpara_save,qperp_save=qparabin[i],qperpbin[i]
            intval=mapints[i]
            try:
                qmap[qperp_save,qpara_save]+=intval
                counts[qperp_save,qpara_save]+=1
            except:
                print(f'failed on {qpara}  {qperp}  {intval} ')
        mapnorm=np.zeros([len(rangeqperp),len(rangeqpara)])
        np.divide(qmap,counts,out=mapnorm,where=counts!=0)
        mapnorms.append(mapnorm)
        rangeqparas.append(rangeqpara)
        rangeqperps.append(rangeqperp)
        return mapnorms,rangeqparas,rangeqperps
        
    
    def calc_qpara_qper(self,scan,oop,frame: Frame,proj2d):
        """
        DEPRECATED -

        Parameters
        ----------
        scan : TYPE
            DESCRIPTION.
        oop : TYPE
            DESCRIPTION.
        frame : Frame
            DESCRIPTION.
        proj2d : TYPE
            DESCRIPTION.

        Returns
        -------
        outlist : TYPE
            DESCRIPTION.

        """
        mapnorms=[]
        rangeqparas=[]
        rangeqperps=[]
        
        if proj2d==None:
            try:
                number_images=len(scan.metadata.data_file.local_image_paths)
            except:
                number_images=scan.metadata.data_file.scan_length
            #scan.metadata.data_file.scan_length
            for imnum in np.arange(number_images):
                pdata=scan.load_image(imnum).data
                allints=np.reshape(pdata,np.size(pdata))
                mapints=allints
                qvals=scan.load_image(imnum).q_vectors(frame=frame,oop=oop)
                mapnorms, rangeqparas, rangeqperps =self.calc_mapnorm_ranges(pdata,qvals,mapnorms,rangeqparas,rangeqperps,mapints)
        else:
            pdata=proj2d[0]
            allints=np.reshape(pdata,np.size(pdata))
            mapints=allints
            kin = scan.metadata.k_incident_length
            qvals=self.projected2dQvecs(pdata,self.vertoffset,oop,kin)
            mapnorms, rangeqparas, rangeqperps =self.calc_mapnorm_ranges(pdata,qvals,mapnorms,rangeqparas,rangeqperps,mapints)
        signal_shape=np.shape(scan.metadata.data_file.default_signal)
        if len(signal_shape)>1:
            out_mapnorms=self.reshape_to_signalshape(mapnorms,signal_shape)
            out_rangeqparas=self.reshape_to_signalshape(rangeqparas, signal_shape)
            out_rangeqperps=self.reshape_to_signalshape(rangeqperps,signal_shape)
            outlist=out_mapnorms,out_rangeqparas, out_rangeqperps

        else:
            outlist=mapnorms,rangeqparas,rangeqperps
        
        
        return outlist
    
    
    def projected2dQvecs(self,projectedimage,vertoffset,oop,kin,horoffset=0):
        """
        DEPRECATED - old function for manually projecting image used before move to pyFAI
        Parameters
        ----------
        projectedimage : TYPE
            DESCRIPTION.
        vertoffset : TYPE
            DESCRIPTION.
        oop : TYPE
            DESCRIPTION.
        kin : TYPE
            DESCRIPTION.
        horoffset : TYPE, optional
            DESCRIPTION. The default is 0.
            
        Returns
        -------
        k_out_array : TYPE
            DESCRIPTION.
            
        """
        
        i = slice(None)
        j = slice(None)
        desired_shape = tuple(list(np.shape(projectedimage)) + [3])
        # if self.metadata.data_file.is_rotated:
        #     desired_shape= tuple(list([self._raw_data.shape[1],self._raw_data.shape[0]]) + [3])
        
        # Don't bother initializing this.
        k_out_array = np.ndarray(desired_shape, np.float32)
        
        # Get a unit vector pointing towards the detector.
        # Get a unit vector pointing towards the detector.
        det_displacement = [0,0,1]
        det_vertical=[0,1,0]
        det_horizontal=[1,0,0]
        detector_distance=self.detector_distance
        
        vertical_pixel_offsets = np.zeros(np.shape(projectedimage), np.float32)
        vertoffset1d= np.arange(np.shape(projectedimage)[0],0,-1) -vertoffset    
        for ival, pixel_offset in enumerate(vertoffset1d):
             vertical_pixel_offsets[ival, :] = pixel_offset
             
        horizontal_pixel_offsets = np.zeros(np.shape(projectedimage),np.float32)
        horoffset1d= np.arange(np.shape(projectedimage)[1],0,-1) -horoffset    
        for ival, pixel_offset in enumerate(horoffset1d):
             horizontal_pixel_offsets[:, ival] = pixel_offset
                 
             
        vertical = vertical_pixel_offsets*self.pixel_size
        horizontal = horizontal_pixel_offsets*self.pixel_size
        
        k_out_array[i, j, 0] = (
            det_displacement[0]*detector_distance +
            det_vertical[0]*vertical[i, j] +
            det_horizontal[0]*horizontal[i, j])
        k_out_array[i, j, 1] = (
            det_displacement[1]*detector_distance +
            det_vertical[1]*vertical[i, j] +
            det_horizontal[1]*horizontal[i, j])
        k_out_array[i, j, 2] = (
            det_displacement[2]*detector_distance +
            det_vertical[2]*vertical[i, j] +
            det_horizontal[2]*horizontal[i, j])
        
        # addition is an order of magnitude faster than using sum(.., axis=-1)
        k_out_squares = np.square(k_out_array[i, j, :])
        
        # k_out_sqares' shape depends on what i and j are. Handle all 3 cases.
        if len(k_out_squares.shape) == 1:
            norms = np.sum(k_out_squares)
        elif len(k_out_squares.shape) == 2:
            norms = (k_out_squares[:, 0] +
                     k_out_squares[:, 1] +
                     k_out_squares[:, 2])
        elif len(k_out_squares.shape) == 3:
            norms = (k_out_squares[:, :, 0] +
                     k_out_squares[:, :, 1] +
                     k_out_squares[:, :, 2])
        norms = np.sqrt(norms)
        
        # Right now, k_out_array[a, b] has units of meters for all a, b. We want
        # k_out_array[a, b] to be normalized (elastic scattering). This can be
        # done now that norms has been created because it has the right shape.
        k_out_array[i, j, 0] /= norms
        k_out_array[i, j, 1] /= norms
        k_out_array[i, j, 2] /= norms
        
        # For performance reasons these should also be float32.
        incident_beam_arr = [0,0,1]
        k_incident_len=kin
            
        
        # Note that this is an order of magnitude faster than:
        # k_out_array -= incident_beam_arr
        k_out_array[i, j, 0] -= incident_beam_arr[0]
        k_out_array[i, j, 1] -= incident_beam_arr[1]
        k_out_array[i, j, 2] -= incident_beam_arr[2]
        
        # It turns out that diffcalc assumes that k has an extra factor of
        # 2π. I would never in my life have realised this had it not been
        # for an offhand comment by my colleague Dean. Thanks, Dean.
        k_out_array[i, j, :] *= k_incident_len * 2*np.pi
        
        ub_mat = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            
        # Finally, we make it so that (001) will end up OOP.
        if oop == 'y':
            coord_change_mat = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
        elif oop == 'x':
            coord_change_mat = np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ])
        elif oop == 'z':
            coord_change_mat = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            
        ub_mat = np.matmul(ub_mat, coord_change_mat)
        ub_mat = ub_mat.astype(np.float32)
        
        k_out_array = k_out_array.reshape(
            (desired_shape[0]*desired_shape[1], 3))
        # This takes CPU time: mapping every vector.
        mapper_c_utils.linear_map(k_out_array, ub_mat)
        # Reshape the k_out_array to have the same shape as the image.
        k_out_array = k_out_array.reshape(desired_shape)
        return k_out_array
    
    
    def pyfai_static_diff(self,hf,scan,num_threads,output_file_path,pyfaiponi,ivqbins,qmapbins=0):      

        """
        DEPRECATED  - calculation of both q perp Vs qpara map and I Vs Q plot

        Parameters
        ----------
        hf : TYPE
            DESCRIPTION.
        scan : TYPE
            DESCRIPTION.
        num_threads : TYPE
            DESCRIPTION.
        output_file_path : TYPE
            DESCRIPTION.
        pyfaiponi : TYPE
            DESCRIPTION.
        ivqbins : TYPE
            DESCRIPTION.
        qmapbins : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        self.load_curve_values(scan)
        
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        if self.setup=='DCD':
            tthdirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        else:
            tthdirect=0
            
        self.two_theta_start=self.gammadata-tthdirect
        qlimhor=self.calcqlim( 'hor')
        qlimver=self.calcqlim( 'vert')
            
        #calculate map bins if not specified using resolution of 0.01 degrees 
            
        if qmapbins==0:
            qstep=round(self.calcq(1.00,self.incident_wavelength)-\
                self.calcq(1.01,self.incident_wavelength),4)
            binshor=abs(round(((qlimhor[1]-qlimhor[0])/qstep)*1.05))
            binsver=abs(round(((qlimver[1]-qlimver[0])/qstep)*1.05))
            qmapbins=(binshor,binsver)
                
        
        
        scalegamma=1
        
        try:
            scanlength=len(scan.metadata.data_file.local_image_paths)
        except:
            scanlength=scan.metadata.data_file.scan_length
                
                        
        print('starting process pool')
        all_maps=[]
        all_xlabels=[]
        all_ylabels=[]
        all_ints=[]
        all_two_ths=[]
        all_Qs=[]
                
        with Pool(processes=num_threads) as pool:
            
            print(f'started pool with num_threads={num_threads}')
            indices=np.arange(0,scanlength,scalegamma)
            input_list = [(self,index,scan,pyfaiponi,qmapbins,ivqbins) for index in indices]
            results=pool.map(self.pyfai_static_qmap_qvsI_wrapper,input_list)
            maps=[result[0] for result in results]
            xlabels=[result[1] for result in results]
            ylabels=[result[2] for result in results]
            intensities=[result[3] for result in results]
            two_th_vals=[result[4] for result in results]
            Q_vals=[result[5] for result in results]
            all_maps.append(maps)
            all_xlabels.append(xlabels)
            all_ylabels.append(ylabels)
            all_ints.append(intensities)
            all_two_ths.append(two_th_vals)
            all_Qs.append(Q_vals)
                
            print('finished preparing chunked data')
                
        signal_shape=np.shape(scan.metadata.data_file.default_signal)
        outlist=[all_maps[0],all_xlabels[0],all_ylabels[0],all_ints[0],all_Qs[0],all_two_ths[0]]
        if len(signal_shape)>1:
            outlist=[self.reshape_to_signalshape(arr, signal_shape) for arr in outlist]
                    
                
                    
        dset=hf.create_group("qpara_qperp")
        dset.create_dataset("qpara_qperp_image",data=outlist[0])
        dset.create_dataset("map_para",data=outlist[1])
        dset.create_dataset("map_perp",data=outlist[2])
                        
        dset=hf.create_group("integrations")
        dset.create_dataset("Intensity",data=outlist[3])
        dset.create_dataset("Q_angstrom^-1",data=outlist[4])
        dset.create_dataset("2thetas",data=outlist[5])    
        if "scanfields" not in hf.keys():
            self.save_scan_field_values(hf, scan)    
        if self.savetiffs==True:
            self.do_savetiffs(hf, outlist[0],outlist[1], outlist[2])
        if self.savedats==True:
            self.do_savedats(hf,outlist[3],outlist[4],outlist[5])

    def calcimagegammarange(self):
        """
        DEPRECATED - 
        Returns
        -------
        totaltheta : TYPE
            DESCRIPTION.
        thetahigh : TYPE
            DESCRIPTION.
        thetalow : TYPE
            DESCRIPTION.
        """
        imshape=self.imshape
        beamcentre=self.beam_centre
        pixsize=self.pixel_size
        dd=self.detector_distance
        leftside=beamcentre[1]*pixsize
        rightside=(imshape[1]-beamcentre[1])*pixsize
        thetahigh=np.degrees(np.arctan(leftside/dd))
        thetalow=np.degrees(np.arctan(rightside/dd))
        totaltheta=thetahigh+thetalow
        return totaltheta,thetahigh,thetalow
    


        
    def calcupplow(self, gamma):
        """
        DEPRECATED - old function used to calculate upper and lower gamma values, before move over to pyFAI

        Parameters
        ----------
        gamma : TYPE
            DESCRIPTION.

        Returns
        -------
        low : TYPE
            DESCRIPTION.
        upp : TYPE
            DESCRIPTION.

        """
        nearlow=np.abs(self.gamma2d-gamma).argmin()
        if self.gamma2d[nearlow]>gamma:
            low=nearlow-1
        else:
            low=nearlow
        gammaupp=gamma+self.degperpix
        nearupp=np.abs(self.gamma2d-(gammaupp)).argmin()
        if self.gamma2d[nearupp]>(gammaupp):
            upp=nearupp
        else:
            upp=nearupp+1
        return low,upp
    

    def calc_projected_size(self, two_theta_start):
        """
        DEPRECATED old function used to calculate size of projected image before move over to using pyFAI

        Parameters
        ----------
        two_theta_start : TYPE
            DESCRIPTION.

        Returns
        -------
        projshape : TYPE
            DESCRIPTION.

        """
        
        if self.rotval==0:
            extrahorizontal=self.beam_centre[1]*self.pixel_size
            startheight=self.imshape[0]*self.pixel_size
        else:
            extrahorizontal=self.beam_centre[0]*self.pixel_size
            startheight=self.imshape[1]*self.pixel_size

        extratwotheta=np.degrees(np.arctan(extrahorizontal/self.detector_distance))

        try: 
            self.maxdist2D=self.detector_distance/np.cos(np.radians(two_theta_start[-1]+extratwotheta))
        except:
            self.maxdist2D=self.detector_distance/np.cos(np.radians(two_theta_start+extratwotheta))
        maxdistdiff=self.maxdist2D-self.detector_distance

        # if self.rotval==0:
        #     startheight=self.imshape[0]*self.pixel_size
        # else:
        startheight=self.imshape[0]*self.pixel_size
        maxdist=startheight+(maxdistdiff*(startheight/self.detector_distance))
        maxheight=np.ceil(maxdist/self.pixel_size)
        self.maxratiodist=self.maxdist2D/self.detector_distance
        
        #calculate the maximum value for the projected width measured in the final image
        try:
            maxwidth=np.ceil(self.detector_distance*np.tan(np.radians(two_theta_start[-1]+extratwotheta))/self.pixel_size)
        except:
            maxwidth=np.ceil(self.detector_distance*np.tan(np.radians(two_theta_start+extratwotheta))/self.pixel_size)
        
        
        # #account for pixels after beam centre
        # if self.rotval==0:
        #     maxwidth+=self.beam_centre[0]
        # else:
        #     maxwidth+=self.beam_centre[1] 
        
               
        projshape=(int(maxheight),int(maxwidth))
        return projshape    



    
    
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
        from time import time
        t1 = time()
        # Make sure that we have a list, in case we just received a single path.
        if isinstance(nexus_paths, (str, Path)):
            nexus_paths = [nexus_paths]

        # Instantiate all of the scans.
        scans = [
            io.from_i07(x, beam_centre, detector_distance,
                        setup, path_to_data, using_dps,experimental_hutch)
            for x in nexus_paths]
        print(f"Took {time() - t1}s to load all nexus files.")
        return cls(scans)
    

        

def _match_start_stop_to_step(
                step,
                user_bounds,
                auto_bounds,
                eps = 1e-5):
    warning_str = ("User provided bounds (volume_start, volume_stop) do not "
                   "match the step size volume_step. Bounds will be adjusted "
                   "automatically. If you want to avoid this warning, make "
                   "that the bounds match the step size, i.e. volume_bound = "
                   "volume_step * integer.")
    
    if user_bounds == (None, None):
        # use auto bounds and expand both ways
        return (np.floor(auto_bounds[0]/step)*step,
                np.ceil(auto_bounds[1]/step)*step)
    elif user_bounds[0] is None:
        # keep user value and expand to rightdone image {i+1}/{totalimages}
        stop = np.ceil(user_bounds[1]/step)*step
        checkstop=np.sum(np.any(abs(stop - user_bounds[1]) > eps))
        if checkstop>0:
            print(warning_str)
        return np.floor(auto_bounds[0]/step)*step, stop
    elif user_bounds[1] is None:
        # keep user value and expand to left
        start = np.floor(user_bounds[0]/step)*step
        checkstart=np.sum(abs(user_bounds[0] - start) > eps)
        if checkstart>0:
            print(warning_str)
        return start, np.ceil(auto_bounds[1]/step)*step
    else:
        start, stop = (np.floor(user_bounds[0]/step)*step,
                       np.ceil(user_bounds[1]/step)*step)
        checkstart=np.sum(abs(user_bounds[0] - start) > eps)
        checkstop=np.sum(np.any(abs(stop - user_bounds[1]) > eps))
        checkboth=checkstart+checkstop
        if checkboth>0:
            print(warning_str)
        return start, stop
    
    
