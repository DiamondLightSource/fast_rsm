"""
This file contains the Experiment class, which contains all of the information
relating to your experiment.
"""
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Pool, Lock
from ast import literal_eval
from types import SimpleNamespace
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
import yaml
# from datetime import datetime
import h5py
import tifffile
import fast_rsm.io as io
from fast_rsm.binning import finite_diff_shape
from fast_rsm.meta_analysis import get_step_from_filesize
from fast_rsm.scan import Scan, chunk, \
    rsm_init_worker, bin_maps_with_indices_smm
from fast_rsm.writing import linear_bin_to_vtk
from fast_rsm.pyfai_interface import createponi as new_createponi, \
    pyfai_moving_exitangles_smm as new_pyfai_moving_exitangles_smm, \
    pyfai_moving_qmap_smm as new_pyfai_moving_qmap_smm

logger = logging.getLogger("fastrsm")

# from memory_profiler import profile


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
            if self.scans[0].metadata.data_file.is_rotated:
                mask = np.rot90(np.flip(mask, axis=0), 1)
        else:
            mask = None

        for scan in self.scans:
            scan.metadata.edfmask = mask

    def q_bounds(self, frame: Frame, spherical_bragg_vec: np.ndarray,
                 oop: str = 'y') -> Tuple[np.ndarray]:
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
                frame, spherical_bragg_vec, oop=oop)
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

        if self.scans[0].metadata.data_file.is_rotated:
            rot_option = 'rot'
        else:
            rot_option = 'norot'

        chosen_setup = f'{self.setup}_{rot_option}'

        index_scales = {'vertical_rot': ['hor0', 'thvert', -1, 1],
                        'vertical_norot': ['hor0', 'thvert', -1, -1],
                        'DCD_rot': ['vert0', 'delvert', -1, -1],
                        'DCD_norot': ['vert0', 'delvert', -1, -1],
                        'horizontal_rot': ['vert0', 'delvert', -1, 1],
                        'horizontal_norot': ['vert0', 'delvert', -1, -1],
                        }

        chosen_ind_scales = index_scales[chosen_setup]
        [horindex, vertindex] = horvert_indices[chosen_ind_scales[0]]
        [vertangles, horangles] = horvert_angles[chosen_ind_scales[1]]
        verscale = chosen_ind_scales[2]
        horscale = chosen_ind_scales[3]

        # if (vertsetup is True) & (self.scans[0].metadata.data_file.is_rotated):
        #     # GOOD
        #     [horindex, vertindex] = horvert_indices['hor0']
        #     [vertangles, horangles] = horvert_angles['thvert']
        #     verscale = -1
        #     horscale = 1

        # elif vertsetup is True:
        #     # GOOD
        #     [horindex, vertindex] = horvert_indices['hor0']
        #     [vertangles, horangles] = horvert_angles['thvert']
        #     verscale = -1
        #     horscale = -1

        # elif (self.setup == 'DCD') & (self.scans[0].metadata.data_file.is_rotated):
        #     [horindex, vertindex] = horvert_indices['vert0']
        #     [vertangles, horangles] = horvert_angles['delvert']
        #     verscale = -1
        #     horscale = -1

        # elif self.setup == 'DCD':
        #     [horindex, vertindex] = horvert_indices['vert0']
        #     [vertangles, horangles] = horvert_angles['delvert']
        #     verscale = -1
        #     horscale = -1

        # elif (vertsetup is False) & (self.scans[0].metadata.data_file.is_rotated):
        #     [horindex, vertindex] = horvert_indices['vert0']
        #     [vertangles, horangles] = horvert_angles['delvert']
        #     verscale = -1
        #     horscale = 1

        # else:
        #     # GOOD
        #     [horindex, vertindex] = horvert_indices['vert0']
        #     [vertangles, horangles] = horvert_angles['delvert']
        #     verscale = -1
        #     horscale = -1

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
        outlist = ['horindex', 'vertindex', 'vertangles', 'horangles',
                   'verscale', 'horscale', 'pixhigh', 'pixlow', 'outscale',
                   'pixscale', 'highsign', 'lowsign', 'highsection', 'lowsection']
        outdict = {}
        for name in outlist:
            outdict[name] = locals().get(name, None)
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


# ==============full reciprocal space mapping process


    def binned_reciprocal_space_map_smm(self,
                                        num_threads: int,
                                        map_frame: Frame,
                                        process_config: SimpleNamespace,
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
        cfg = process_config
        if map_frame.frame_name == Frame.qpar_qperp:
            map_frame.frame_name = Frame.lab
          # Compute the optimal finite differences volume.

        if volume_step is None:
            # Overwrite whichever of these we were given explicitly.
            if (volume_start is not None) & (volume_stop is not None):
                _start = np.array(volume_start)
                _stop = np.array(volume_stop)
            else:
                _start, _stop = self.q_bounds(map_frame, np.ndarray(cfg.spherical_bragg_vec),
                                              oop)

            step = get_step_from_filesize(_start, _stop, output_file_size)

        else:
            step = np.array(volume_step)
            _start, _stop = self.q_bounds(map_frame, np.ndarray(cfg.spherical_bragg_vec),
                                          oop)

        if map_frame.coordinates == Frame.sphericalpolar:
            step = np.array([0.02, np.pi / 180, np.pi / 180])

        # Make sure start and stop match the step as required by binoviewer.
        start, stop = _match_start_stop_to_step(
            step=step,
            user_bounds=(volume_start, volume_stop),
            auto_bounds=(_start, _stop))

        # locks = [Lock() for _ in range(num_threads)]
        shapersm = finite_diff_shape(start, stop, step)

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
                     np.ndarray(cfg.spherical_bragg_vec),
                     map_each_image,
                     images_so_far) for indices in chunk(
                        list(
                            range(
                                scan.metadata.data_file.scan_length)),
                        num_threads)]

                with Pool(num_threads, initializer=rsm_init_worker,
                          initargs=(l, shm_rsm.name, shm_counts.name, shapersm, scan.metadata,
                                    new_metadata, new_motors, num_threads, map_frame, output_file_name)) as pool:
                    print(f'started pool with num_threads={num_threads}')
                    pool.starmap(bin_maps_with_indices_smm, bin_args)

                print(
                    f'finished process pool for scan {scanind+1}/{len(self.scans)}')
                images_so_far += scan.metadata.data_file.scan_length

        normalised_map = np.divide(
            rsm_arr,
            counts_arr,
            out=np.copy(rsm_arr),
            where=counts_arr != 0.0)

        # Only save the vtk/npy files if we've been asked to.
        if cfg.save_vtk:
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
        if cfg.save_npy:
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

# =======refactored functions now in fast_rsm.pyfai_interface
    def createponi(self, outpath, image2dshape, beam_centre=0, offset=0):
        return new_createponi(self, outpath)

    def pyfai_moving_exitangles_SMM(self, hf, scanlist, num_threads, output_file_path,
                                    pyfaiponi, radrange, radstepval, qmapbins=[800, 800], slitratios=None):

        return new_pyfai_moving_exitangles_smm(self, hf, scanlist, num_threads, output_file_path,
                                               pyfaiponi, radrange, radstepval, qmapbins=[800, 800], slitratios=None)

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
