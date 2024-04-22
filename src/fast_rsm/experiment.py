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

import numpy as np
from diffraction_utils import Frame, Region
from scipy.constants import physical_constants

from . import io
from .binning import weighted_bin_1d, finite_diff_shape
from .meta_analysis import get_step_from_filesize
from .scan import Scan, init_process_pool, bin_maps_with_indices, chunk
from .writing import linear_bin_to_vtk
import pandas as pd
import pyFAI,fabio
#from datetime import datetime
import h5py

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

        is going to be used at some point.

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
                pool.close()
                pool.join()
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
    
    

    
    def calcupplow(self, gamma):
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
        
        if self.rotval==0:
            extrahorizontal=self.beam_centre[1]*self.pixel_size
            startheight=self.imshape[0]*self.pixel_size
        else:
            extrahorizontal=self.beam_centre[0]*self.pixel_size
            startheight=self.imshape[1]*self.pixel_size
            
        extratwotheta=np.degrees(np.arctan(extrahorizontal/self.detector_distance))
        self.maxdist2D=self.detector_distance/np.cos(np.radians(two_theta_start[-1]+extratwotheta))
        maxdistdiff=self.maxdist2D-self.detector_distance
        if self.rotval==0:
            startheight=self.imshape[0]*self.pixel_size
        else:
            startheight=self.imshape[1]*self.pixel_size
        maxdist=startheight+(maxdistdiff*(startheight/self.detector_distance))
        maxheight=np.ceil(maxdist/self.pixel_size)
        self.maxratiodist=self.maxdist2D/self.detector_distance
        
        #calculate the maximum value for the projected width measured in the final image
        maxwidth=np.ceil(self.detector_distance*np.tan(np.radians(two_theta_start[-1]+extratwotheta))/self.pixel_size)
        
        
        # #account for pixels after beam centre
        # if self.rotval==0:
        #     maxwidth+=self.beam_centre[0]
        # else:
        #     maxwidth+=self.beam_centre[1] 
        
               
        projshape=(int(maxheight),int(maxwidth))
        return projshape    


    def projectimage(self, scan,imnum,im1gammas):
        
        data=scan.load_image(imnum).data
        
        #cropshape,cropdata,cropbc,gamshifts,delshifts=calc_cropshifts(xylimits,data,self.degperpix,two_theta_start,originalbc)
        if imnum==0:
            imgammas=im1gammas
    
        else:
            imgammas=im1gammas+(imnum*self.gammastep)
            
        
        for j in np.arange(self.imshape[1]):
            gamma=imgammas[0,j]
            horlow,horupp=self.calcupplow(gamma)
            avgamma=(gamma +gamma+self.degperpix)/2
            dist2D=self.detector_distance/np.cos(np.radians(avgamma)) 
            ratiodist=dist2D/self.detector_distance
            pixvals=data[:,j]
            self.nonenans=np.where(~np.isnan(pixvals))[0]
            for ind in self.nonenans:            
                deltaind=(ind-self.beam_centre[0])*-1
                vertstart=int(np.floor(deltaind*(ratiodist)))+self.vertoffset
                vertend=int(np.ceil((deltaind+1)*(ratiodist)))+self.vertoffset
                self.project2d[self.projshape[0]-vertend:self.projshape[0]-vertstart,-horupp:-horlow]+=data[ind][j]
                self.counts[self.projshape[0]-vertend:self.projshape[0]-vertstart,-horupp:-horlow]+=1

        
                
 
            
    def load_curve_values(self,scan):
        self.pixel_size=scan.metadata.diffractometer.data_file.pixel_size
        self.entry=scan.metadata.data_file.nx_entry
        self.detector_distance=scan.metadata.diffractometer.data_file.detector_distance
        self.incident_wavelength= 1e-10*scan.metadata.incident_wavelength
        self.gammadata=np.array( self.entry.instrument.diff1gamma.value)
        self.deltadata=np.array( self.entry.instrument.diff1delta.value)
        self.dcdrad=np.array( self.entry.instrument.dcdc2rad.value)
        self.dcdomega=np.array( self.entry.instrument.dcdomega.value)
        self.projectionx=1e-3* self.dcdrad*np.cos(np.radians(self.dcdomega))
        self.imshape=scan.metadata.data_file.image_shape
        self.beam_centre=scan.metadata.beam_centre
        self.rotval=round(scan.metadata.data_file.det_rot)

        
    def curved_to_2d(self,scan):
        self.load_curve_values(scan)
        dcd_sample_dist=1e-3*scan.metadata.diffractometer._dcd_sample_distance
        gammadirect=-1*np.degrees(np.arctan(self.projectionx/dcd_sample_dist))
        two_theta_start=self.gammadata-gammadirect
        self.twothetastart=two_theta_start
        projshape=self.calc_projected_size(two_theta_start)
        self.degperpix=np.degrees(np.arctan(self.pixel_size/(self.detector_distance)))
    
        self.gammastep=(two_theta_start[-1]-two_theta_start[0])/(len(two_theta_start)-1)

        self.gamma2d=np.degrees(np.arctan((np.arange(projshape[1])*self.pixel_size)/self.detector_distance))
        self.projshape=self.calc_projected_size(two_theta_start)

        self.vertoffset=int((self.imshape[0]-self.beam_centre[0])*self.maxratiodist)

        self.project2d=np.zeros(projshape)
        self.counts=np.zeros(projshape)
        scanlength=scan.metadata.data_file.scan_length
        
        gamshifts=-1*(np.arange(self.imshape[1])-self.beam_centre[1])
        delshifts=np.abs(np.arange(self.imshape[0])-self.beam_centre[1])
        im1gammas=np.zeros(self.imshape)
        for col in np.arange(np.shape(im1gammas)[1]):
            O=gamshifts[col]*self.pixel_size
            tantheta=O/self.detector_distance
            im1gammas[:,col]=two_theta_start[0]+(np.degrees(np.arctan(tantheta)))
        self.imgamma=im1gammas
        print(f'projecting {scanlength} images   completed images:  ')
        for imnum in np.arange(scanlength):
            self.projectimage(scan, imnum,im1gammas)
            if (imnum+1)%10==0:
                print('\n')
            print(fr'{imnum+1}','\r', end='')
            #print(f' projected image {outstring}','\r', end='')
        norm2d=np.divide(self.project2d,self.counts,where=self.counts!=0)
        projected_data=[norm2d,self.counts,self.vertoffset]
        return projected_data
    
    
    def createponi(self,outpath,image2dshape,offset=0):
        f=open(fr'{outpath}/fast_rsm.poni','w')
        f.write('# PONI file created by fast_rsm\n#\n')
        f.write('poni_version: 2\n')
        f.write('Detector: Detector\n')
        f.write('Detector_config: {"pixel1":')
        f.write(f'{self.pixel_size}, "pixel2": {self.pixel_size}, "max_shape": [{image2dshape[0]}, {image2dshape[1]}]') 
        f.write('}\n')
        f.write(f'Distance: {self.detector_distance}\n')
        poni1=(image2dshape[0]-offset)*self.pixel_size
        poni2=image2dshape[1]*self.pixel_size
        f.write(f'Poni1: {poni1}\n')
        f.write(f'Poni2: {poni2}\n')
        f.write('Rot1: 0.0\n')
        f.write('Rot2: 0.0\n')
        f.write('Rot3: 0.0\n')
        f.write(f'Wavelength: {self.incident_wavelength}')
        f.close()
        return fr'{outpath}/fast_rsm.poni'
    
    
    def save_projection(self,hf,projected2d,twothetas,Qangs,intensities,config):
        
        #hf=h5py.File(f'{local_output_path}/{projected_name}.hdf5',"w")
        dset=hf.create_group("projection")
        dset.create_dataset("projection_2d",data=projected2d[0])
        dset.create_dataset("config",data=str(config))

        dset=hf.create_group("integrations")
        dset.create_dataset("2thetas",data=twothetas)
        dset.create_dataset("Q_angstrom^-1",data=Qangs)
        dset.create_dataset("Intensity",data=intensities)
        #hf.close()
    
    def save_integration(self,hf,twothetas,Qangs,intensities,configs):
       # hf=h5py.File(f'{local_output_path}/{integrated_name}.hdf5',"w")
        dset=hf.create_group("integrations")
        
        dset.create_dataset("configs",data=str(configs))
        dset.create_dataset("2thetas",data=twothetas)
        dset.create_dataset("Q_angstrom^-1",data=Qangs)
        dset.create_dataset("Intensity",data=intensities)
        #hf.close()
    
    def save_qperp_qpara(self,hf,qperp_qpara_map):
        #hf=h5py.File(f'{local_output_path}/{out_name}.hdf5',"w")
        dset=hf.create_group("qperp_qpara")
        dset.create_dataset("images",data=qperp_qpara_map[0])
        dset.create_dataset("qpararanges",data=qperp_qpara_map[1])
        dset.create_dataset("qperpranges",data=qperp_qpara_map[2])
        #hf.close()

    def pyfai1D(self,imagespath,maskpath,ponipath,outpath,scan,projected2d=None,bins=1000):
        #images=scan.metadata.data_file.local_image_paths
        if projected2d==None:
            scanlength=scan.metadata.data_file.scan_length
        else:
            scanlength=1
        #tiflist=[file.split(f'{imagespath}')[-1] for file in images]
        twothetas=[]
        intensities=[]
        Qangs=[]
        configs=[]
        for i in np.arange(scanlength):

            if projected2d==None:  
               
                # fname=scan.
                # img = fabio.open(fr'{imagespath}/{fname}')
                img_array = scan.load_image(i).data
                maskimg = fabio.open(maskpath)
                mask = maskimg.data
    
            else:
                img_array=projected2d[0]
                mask=np.less_equal(projected2d[1],0)
            
            ai = pyFAI.load(ponipath)       
            # print("\nIntegrator: \n", ai)
        
            
            # print("img_array:", type(img_array), img_array.shape, img_array.dtype)
            
            #GIVE PATH TO MASK FILE
    
            
            
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
        
        return twothetas,Qangs,intensities,configs

    
    def calc_qpara_qper(self,scan,oop,frame: Frame):
        number_images=scan.metadata.data_file.scan_length
        mapnorms=[]
        rangeqparas=[]
        rangeqperps=[]
        for imnum in np.arange(number_images):
            pdata=scan.load_image(imnum).data
            allints=np.reshape(pdata,np.size(pdata))
            mapints=allints
            qvals=scan.load_image(imnum).q_vectors(frame=frame,oop=oop)
    
            
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
                qpara,qperp=qparabin[i],qperpbin[i]
                intval=mapints[i]
                try:
                    qmap[qperp,qpara]+=intval
                    counts[qperp,qpara]+=1
                except:
                    print(f'failed on {qpara}  {qperp}  {intval} ')
            mapnorm=np.zeros([len(rangeqperp),len(rangeqpara)])
            np.divide(qmap,counts,out=mapnorm,where=counts!=0)
            mapnorms.append(mapnorm)
            rangeqparas.append(rangeqpara)
            rangeqperps.append(rangeqperp)
        return mapnorms,rangeqparas,rangeqperps

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
    
    
