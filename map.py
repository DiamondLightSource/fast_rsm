#!/usr/bin/env python
# coding: utf-8

# # How to use this notebook
# 
# This notebook is converted into a .py file, which provides the instructions for the calculation to be run on the cluster. 
# 
# steps are:
# - Edit the cells which require specific input. This will mostly be just the 'Experiment settings' section, but potentially the 'Altering metadata' section will need edits as well. 
# 
# - Once these edits have been made, save the notebook (ctrl+s) and then run the first cell in the 'Cluster submission' section (shortcut for running individual cell is shift+enter). 
# 
# - This should output something similar to:
# 
#     <i>"[NbConvertApp] Converting notebook map.ipynb to script <br>
#     [NbConvertApp] Writing 29470 bytes to map.py<br>
#     Submitted batch job 52137"</i>
#     <br>
#     
# - If submission was successful (i.e. a batch job number was given), you can optionally run the following cell in the 'Cluster submission' section to view a  list of current jobs being run on the cluster. However fast_rsm is usually completed quite quickly so may not be in list. 
# 
# 

# # import packages

# In[ ]:


"""
**IGNORE**

First we need to import some stuff. Feel free to ignore this cell.

If you're interested, each import has an associated comment that explains why
the import is useful/necessary.
"""

# Needed for various filesystem tasks (os.path.exists etc.)
import os
# Used for checking how many cores are available for processing.
import multiprocessing
# Used for constructing paths.
from pathlib import Path

# Essential for all mathematical operations we'll be carrying out.
import numpy as np

# diffraction_utils is a library developed at Diamond by Richard Brearton
# (richard.brearton@diamond.ac.uk) to ease the task of parsing data files and
# carrying out some common calculations. Here, we'll be using it to define
# frames of reference, and parse nexus files.
# We also use diffraction_utils' Region object to specify regions of interest/
# background regions.
from diffraction_utils import Frame, Region

# The following imports are required for the core of the calculation code, also
# written by Richard Brearton (richard.brearton@diamond.ac.uk).
# This is the central Experiment object, which stores all the logic related to
# mapping the experiment.
from fast_rsm.experiment import Experiment


# # Experiment settings

# In[ ]:


"""
**ESSENTIAL**

This cell requires action! Make sure you set all of the variables defined here.
"""

# How was your sample mounted? Options are 'horizontal', 'vertical' and 'DCD'.
setup = 'vertical'

# Set local_data_path if your data isn't stored on the diamond system any more
# (for example if it's on a memory stick or scratch drive).
local_data_path = None
# Set this if you want to save the output somewhere other than the processing
# folder. Be warned, this could take up a lot of space.
local_output_path = None

# If you're processing on the cluster, you need to populate the next few fields.
# The experiment number, used to work out where your data is stored.
experiment_number = 'si31429-1'

# The sub-directory containing your experimental data. Leave as None if unused.
# Otherwise, if the data was stored in a subdirectory called "day_1", e.g.
#   /dls/i07/data/2022/si32333-1/day_1/
# then you should use:
#   data_sub_directory = "day_1"
data_sub_directory = "sample1/"

# The year the experiment took place.
year = 2022

# The scan numbers of the scans that we want to use to produce this reciprocal
# space map. For example, the default value of scan_numbers shows how to specify
# every scan between number 421772 and 421778 inclusive, but skipping scan
# number 421776.
scan_numbers = [446509]

# Uncomment the following to set scan_numbers equal to every scan number between
# scan_start and scan_stop (inclusive of scan_stop):
# scan_start = 439168
# scan_stop = 439176
# scan_numbers = list(range(scan_start, scan_stop + 1))

# The beam centre, as can be read out from GDA, in pixel_x, pixel_y. If your
# map looks wacky, you probably cocked this up.
beam_centre = (243, 92)

# The distance between the sample and the detector (or, if using the DCD, the
# distance between the receiving slit and the detector). Units of meters.
detector_distance = 930e-3

# The frame/coordinate system you want the map to be carried out in.
# Options for frame_name argument are:
#     Frame.hkl     (map into hkl space - requires UB matrix in nexus file)
#     Frame.sample_holder   (standard map into 1/Å)
#     Frame.lab     (map into frame attached to lab.)
#
# Options for coordinates argument are:
#     Frame.cartesian   (normal cartesian coords: hkl, Qx Qy Qz, etc.)
#     Frame.polar       (cylindrical polar with cylinder axis set by the
#                        cylinder_axis variable)
#
# Frame.polar will give an output like a more general version of PyFAI.
# Frame.cartesian is for hkl maps and Qx/Qy/Qz. Any combination of frame_name
# and coordinates will work, so try them out; get a feel for them.
# Note that if you want something like a q_parallel, q_perpendicular projection,
# you should choose Frame.lab with cartesian coordinates. From this data, your
# projection can be easily computed.
frame_name = Frame.hkl
coordinates = Frame.cartesian

# Ignore this unless you selected Frame.polar.
# This sets the axis about which your polar coordinates will be generated.
# Options are 'x', 'y' and 'z'. These are the synchrotron coordinates, rotated
# according to your requested frame_name. For instance, if you select
# Frame.lab, then 'x', 'y' and 'z' will correspond exactly to the synchrotron
# coordinate system (z along beam, y up). If you select frame.sample_holder and
# rotate your sample by an azimuthal angle µ, then 'y' will still be vertically
# up, but 'x' and 'z' will have been rotated about 'y' by the angle µ.
# Leave this as "None" if you aren't using cylindrical coordinates.
cylinder_axis = None

# Set this to True if you would like each image to be mapped independently.
# If this is False, all images in all scans will be combined into one large
# reciprocal space map.
map_per_image = False

# How large would you like your output file to be, in MB? 100MB normally gives
# very good resolution without sacrificing performance. If you want something
# higher resolution, feel free, but be aware that the performance of the map and
# the analysis will start to suffer above around 1GB.
# Max file size is 2GB (2048MB).
output_file_size = 50

# This is for loading into binoculars. If set to false, .npy and .vtr files
# will be saved, for manual analysis and paraview, respectively.
save_binoculars_h5 = True

# Are you using the DPS system?
using_dps = False

# The DPS central pixel locations are not typically recorded in the nexus file.
# NOTE THAT THIS SHOULD BE THE CENTRAL PIXEL FOR THE UNDEFLECTED BEAM.
# UNITS OF METERS, PLEASE (everything is S.I., except energy in eV).
dpsx_central_pixel = 0
dpsy_central_pixel = 0
dpsz_central_pixel = 0

# Note: THESE CAN BE HAPPILY AUTO CALCULATED.
# These take the form:
# volume_start = [h_start, k_start, l_start]
# volume_stop = [h_stop, k_stop, l_stop]
# volume_step = [h_step, k_step, l_step]
# Leave as None if you don't want to specify them. You can specify whichever
# you like (e.g. you can specify step and allow start/stop to be auto
# calculated)
volume_start = None
volume_stop = None
volume_step = None

# Only use this if you need to load your data from a .dat file.
load_from_dat = False


# In[ ]:


"""
**MASKING**

This cell contains details on how to mask pixels. You can either mask a series
of individual pixels, mask rectangular regions of pixels, or dynamically mask
pixels based on their intensity (not recommended).
"""

# If you have a small number of hot pixels to mask, specify them one at a time
# in a list. In other words, it should look like:
# specific_pixels = [(pixel_x1, pixel_y1), (pixel_x2, pixel_y2)]
# Or, an exact example, where we want to mask pixel (233, 83) and pixel 
# (234, 83), where pixel coordinates are (x, y):
# 
# specific_pixels = [
#     (233, 83),
#     (234, 83)
# ]
# 
# Leave specific pixels as None if you dont want to mask any specific pixels.
# For this dataset we need to mask pixel (x=233, y=83)
specific_pixels = None

# If you want to specify an entire region of pixels to mask, do so here.
# This is done using a "Region" object. To make a Region, give it start_x, 
# stop_x, start_y, start_y, as follows:
# 
mask_1 = Region(0, 180, 0, 30)
mask_2 = Region(320, -1, 165, -1)
mask_3 = Region(0, -1, 0, 30)
mask_4 = Region(0, -1, 165, -1)
 
# Where my_mask_region runs in x from pixel 3 to 6 inclusive, and runs in y from
# pixel 84 to 120 inclusive. You can make as many mask regions as you like, just
# make sure that you put them in the mask_regions list, as follows:
# mask_regions = [my_mask_region, Region(1, 2, 3, 4)]
# 
# If you don't want to use any mask regions, just leave mask_regions equal to
# None.
# mask_regions = [mask_1, mask_2, mask_3, mask_4]
mask_regions = None

# Ignore pixels with an intensity below this value. If you don't want to ignore
# any pixels, then set min_intensity = None. This is useful for dynamically
# creating masks (which is really useful for generating masks from -ve numbers).
min_intensity = 0.


# # calculation preparation

# In[ ]:


"""
**IGNORE**
This cell prepares the calculation. You probably shouldn't change anything here
unless you know what you're doing.
"""

# Warn if dps offsets are silly.
if ((dpsx_central_pixel > 10) or (dpsy_central_pixel > 10) or 
    (dpsz_central_pixel > 10)):
    raise ValueError("DPS central pixel units should be meters. Detected "
                     "values greater than 10m")


# Which synchrotron axis should become the out-of-plane (001) direction.
# Defaults to 'y'; can be 'x', 'y' or 'z'.
if setup == 'vertical':
    oop = 'x'
elif setup == 'horizontal':
    oop = 'y'
elif setup == 'DCD':
    oop = 'y'
else:
    raise ValueError(
        "Setup not recognised. Must be 'vertical', 'horizontal' or 'DCD.")

# Overwrite the above oop value depending on requested cylinder axis for polar
# coords.
if cylinder_axis is not None:
    oop = cylinder_axis

if output_file_size > 2000:
    raise ValueError("output_file_size must not exceed 2000. "
                     f"Value received was {output_file_size}.")

# Max number of cores available for processing.
num_threads = multiprocessing.cpu_count()

# Work out where the data is.
if local_data_path is None:
    data_dir = Path(f"/dls/i07/data/{year}/{experiment_number}/")
else:
    data_dir = Path(local_data_path)
# data_dir = Path(f"/Users/richard/Data/i07/{experiment_number}/")

# Store this for later.
if local_output_path is None:
    processing_dir = data_dir / "processing"
else:
    processing_dir = Path(local_output_path)
if data_sub_directory is not None:
    data_dir /= Path(data_sub_directory)

# Here we calculate a sensible file name that hasn't been taken.
i = 0

save_file_name = f"mapped_scan_{scan_numbers[0]}_{i}"
save_path = processing_dir / save_file_name
# Make sure that this name hasn't been used in the past.
while (os.path.exists(str(save_path) + ".npy") or
       os.path.exists(str(save_path) + ".vtk") or
       os.path.exists(str(save_path) + "_l.txt") or
       os.path.exists(str(save_path) + "_tth.txt") or
       os.path.exists(str(save_path) + "_Q.txt") or
       os.path.exists(save_path)):
    i += 1
    save_file_name = f"mapped_scan_{scan_numbers[0]}_{i}"
    save_path = processing_dir / save_file_name

    if i > 1e7:
        raise ValueError(
            "Either you tried to save this file 10000000 times, or something "
            "went wrong. I'm going with the latter, but exiting out anyway.")


# Work out the paths to each of the nexus files. Store as pathlib.Path objects.
nxs_paths = [data_dir / f"i07-{x}.nxs" for x in scan_numbers]

# Construct the Frame object from the user's preferred frame/coords.
map_frame = Frame(frame_name=frame_name, coordinates=coordinates)

# Prepare the pixel mask. First, deal with any specific pixels that we have.
# Note that these are defined (x, y) and we need (y, x) which are the
# (slow, fast) axes. So: first we need to deal with that!
if specific_pixels is not None:
    specific_pixels = specific_pixels[1], specific_pixels[0]

# Now deal with any regions that may have been defined.
# First make sure we have a list of regions.
if isinstance(mask_regions, Region):
    mask_regions = [mask_regions]

# Now swap (x, y) for each of the regions.
if mask_regions is not None:
    for region in mask_regions:
        region.x_start, region.y_start = region.y_start, region.x_start
        region.x_end, region.y_end = region.y_end, region.x_end

# Finally, instantiate the Experiment object.
experiment = Experiment.from_i07_nxs(
    nxs_paths, beam_centre, detector_distance, setup, 
    using_dps=using_dps)

experiment.mask_pixels(specific_pixels)
experiment.mask_regions(mask_regions)


# # Altering metadata

# In[ ]:


"""
**POTENTIALLY REQUIRED**

This cell is for changing metadata that is stored in, or inferred from, the
nexus file. This is generally for more nonstandard stuff.
"""

total_images = 0
for i, scan in enumerate(experiment.scans):
    total_images += scan.metadata.data_file.scan_length
    # Deal with the dps offsets.
    if scan.metadata.data_file.using_dps:
        if scan.metadata.data_file.setup == 'DCD':
            # If we're using the DCD and the DPS, our offset calculation is
            # somewhat involved. If you're confused about this and would like to
            # see a derivation, contact Richard Brearton.

            # Work out the in-plane and out-of-plane incident light angles.
            # To do this, first grab a unit vector pointing along the beam.
            lab_frame = Frame(Frame.lab, scan.metadata.diffractometer, 
                              coordinates=Frame.cartesian)
            beam_direction = scan.metadata.diffractometer.get_incident_beam(
                lab_frame).array

            # Now do some basic handling of spherical polar coordinates.
            out_of_plane_theta = np.sin(beam_direction[1])
            cos_theta_in_plane = beam_direction[2]/np.cos(out_of_plane_theta)
            in_plane_theta = np.arccos(cos_theta_in_plane)

            # Work out the total displacement from the undeflected beam of the
            # central pixel, in the x and y directions (we know z already).
            # Note that dx, dy are being calculated with signs consistent with
            # synchrotron coordinates.
            total_dx = -detector_distance * np.tan(in_plane_theta)
            total_dy = detector_distance * np.tan(out_of_plane_theta)

            # From these values we can compute true DPS offsets.
            dps_off_x = total_dx - dpsx_central_pixel
            dps_off_y = total_dy - dpsy_central_pixel

            scan.metadata.data_file.dpsx += dps_off_x
            scan.metadata.data_file.dpsy += dps_off_y
            scan.metadata.data_file.dpsz -= dpsz_central_pixel
        else:
            # If we aren't using the DCD, our life is much simpler.
            scan.metadata.data_file.dpsx -= dpsx_central_pixel
            scan.metadata.data_file.dpsy -= dpsy_central_pixel
            scan.metadata.data_file.dpsz -= dpsz_central_pixel

        # Load from .dat files if we've been asked.
        if load_from_dat:
            dat_path = data_dir / f"{scan_numbers[i]}.dat"
            scan.metadata.data_file.populate_data_from_dat(dat_path)

    # This is where you might want to overwrite some data that was recorded
    # badly in the nexus file. See (commented out) examples below.
    # scan.metadata.data_file.probe_energy = 12500
    # scan.metadata.data_file.transmission = 0.4
    # scan.metadata.data_file.using_dps = True
    # scan.metadata.data_file.ub_matrix = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])

    # Would you like to skip any images in any scans? Do so here!
    # This shows how to skip the 9th in the 3rd scan (note the zero counting).
    # if i == 2:
    #     scan.skip_images.append(8)



# In[ ]:


"""
**IGNORE**

This cell contains all of the logic for running the calculation. You shouldn't
run this on your local computer, it'll either raise an exception or take
forever.
"""
from fast_rsm.diamond_utils import save_binoculars_hdf5
from time import time


if __name__ == "__main__":
    start_time = time()
    # Calculate and save a binned reciprocal space map, if requested.
    experiment.binned_reciprocal_space_map(
        num_threads, map_frame, output_file_size=output_file_size, oop=oop,
        min_intensity_mask=min_intensity,
        output_file_name=save_path, 
        volume_start=volume_start, volume_stop=volume_stop,
        volume_step=volume_step,
        map_each_image=map_per_image)

    if save_binoculars_h5:
        save_binoculars_hdf5(str(save_path) + ".npy", str(save_path) + '.hdf5')
        print(f"\nSaved BINoculars file to {save_path}.hdf5.\n")

    # Finally, print that it's finished We'll use this to work out when the
    # processing is done.
    total_time = time() - start_time
    print(f"\nProcessing took {total_time}s")
    print(f"This corresponds to {total_time*1000/total_images}ms per image.\n")

    print("PROCESSING FINISHED.")

class DontContinue(Exception):
    """Raise to stop processing on the cluster at this cell"""

raise DontContinue("Processing complete!!\n"
                   "This is intentionally raised to stop the processing. "
                   "Never worry about the presence of this 'error'.")


# # Cluster submission

# In[ ]:


"""

**ESSENTIAL**

This is the cell that you should execute to run this notebook on the cluster.

DO NOT EXECUTE THIS MULTIPLE TIMES. IT WILL SUBMIT MULTIPLE JOBS TO THE CLUSTER.
PLEASE BE RESPONSIBLE.
"""
# We need this to grab the current working directory.
import os

# We'll need this to run the program that will submit the cluster job.
# This module isn't needed for the calculation itself, which is why it is
# imported here.
import subprocess

# First, we save this as "map.py". Make sure it doesn't already exist.
try:
    os.remove("map.py")
except OSError:
    pass

# Convert this notebook to a python script in our home directory.
get_ipython().system('jupyter nbconvert --to script map.ipynb')

#new command to submit job to cluster using SLURM
subprocess.run(["ssh","wilson","cd fast_rsm \nsbatch mapscript.sh"])


# In[ ]:


#new command to check progress on cluster using SLURM
subprocess.run(["ssh","wilson","squeue"])


# In[ ]:


###


# # analysis cells
# The following cells are only needed for 2D/1D analysis of data.
# 

# In[ ]:


###


# In[ ]:


"""
This cell can be used to calculate I(Q) exactly, when the map_per_image option
has been selected.
"""

import plotly.express as px
from fast_rsm.diamond_utils import intensity_vs_q_exact, load_exact_map

# Replace these with paths to your files.
q_path = '/path/to/data/<name>_q.npy'
intensity_path = '/path/to/data/<name>_uncorrected_intensities.npy'

# Load all unbinned q vectors and intensities.
all_qs, all_intensities = load_exact_map(q_path, intensity_path)

# Defaulting to 1000 is going to be sensible in almost all use cases.
number_of_bins = 1000

# Calculate I(Q).
I, Q = intensity_vs_q_exact(all_qs, all_intensities, number_of_bins)

# Plot I(Q).
px.line(x=Q, y=I).show()


# In[ ]:


"""
This cell can be used to calculate exact Qxy Qz projections per image, when
the map_per_image option has been selected.
"""

import plotly.express as px
from fast_rsm.diamond_utils import qxy_qz_exact, load_exact_map

# Replace these with paths to your files.
q_path = '/path/to/data/<name>_q.npy'
intensity_path = '/path/to/data/<name>_uncorrected_intensities.npy'

# Load all unbinned q vectors and intensities.
all_qs, all_intensities = load_exact_map(q_path, intensity_path)

# You should set this to be slightly less than the horizontal resolution of your
# detector.
qxy_bins = 1000

# You should set this to be slightly less than the vertical resolution of your
# detector.
qz_bins = 1000

qxy_qz_intensities, q_vecs = qxy_qz_exact(
    all_qs, all_intensities, qxy_bins, qz_bins)

px.imshow(qxy_qz_intensities).show()


# In[ ]:


"""
The following cells are included for instructional purposes. They contain
examples of how to manipulate data output by fast_rsm.
"""


# In[ ]:


"""
EXAMPLE 1: I(Q)

Here we show how to compute intensity as a function of |Q| from scratch, using
exact, unbinned data obtained via the map_per_image option.
"""

# We're going to need the numpy library to do arithmetic and it'll help with
# loading in the data, too.
import numpy as np

# Replace these with paths to your files.
q_path = '/path/to/data/<name>_q.npy'
intensity_path = '/path/to/data/<name>_uncorrected_intensities.npy'

# Now let's load in that data.
intensities = np.load(intensity_path)
q_vectors = np.load(q_path)

# Imagine that we had N pixels in our detector that was used to capture these
# images. The intensities array is an array with N elements. Each element of the
# intensities array contains, as you would expect, the intensity measured at a
# particular pixel.
# 
# Similarly, the q_vectors array has 3N elements. The first three elements of
# the q_vectors array describe the coordinates of reciprocal space at which the
# first element of intensities was measured, and so on.
#
# Lets write some code to prove some of these statements, and start to get the
# hang of manipulating the output of fast_rsm.

# First lets check the "shape" of the two arrays we've loaded.
print("The shape of the intensities array is:", intensities.shape)
print("The shape of the q_vectors array is:", q_vectors.shape)


# In[ ]:


# Okay, now lets check the first intensity and look at its corresponding
# q_vector. Note that there's a high probability that the first intensity
# is np.nan (meaning not a number). This is fine! A good fraction of pixels are
# masked for various reasons, and NaN is used to represent masked pixels.
print("The first intensity recorded is:", intensities[0])
print("That intensity's corresponding scattering vector is:", q_vectors[0:3])

# Now let's look at, say, the 50,000th intensity and its corresponding
# scattering vector.
print("The 50,000th intensity is:", intensities[50000])
print("The 50,000th scattering vector is:", q_vectors[3*50000:(3*50000+3)])


# In[ ]:


# So, because there are 3 q-vectors per intensity, we have to do some kinda
# annoying arithmetic all the time to access q_vectors. We can make our lives
# somewhat easier by changing the shape of the q_vectors array. Lets do that.
q_vectors = q_vectors.reshape((intensities.shape[0], 3))

# Now lets look at the shape of our q_vectors.
print("The new shape of our q_vectors array is:", q_vectors.shape)

# Let's once again print the 50,000th scattering vector.
print("The 50,000th scattering vector is:", q_vectors[50000])

# ...Much easier!

# Finally, let's see how to access the individual components of each scattering
# vector.
print("The 2nd component of the 50,000th scattering vector is:", q_vectors[50000, 1])


# In[ ]:


# Now that we hopefully have some intuition as to how the data is laid out, lets
# start trying to calculate I(Q).

# Start by completely ignoring all NaN intensity values.
q_vectors = q_vectors[~np.isnan(intensities)]
intensities = intensities[~np.isnan(intensities)]

# Now create a new array that contains the lengths of all the q vectors.
q_lengths = np.sqrt(q_vectors[:, 0]**2 + q_vectors[:, 1]**2 + q_vectors[:, 2]**2)

# Note that x**2 computes "x squared", which is fairly common syntax.


# In[ ]:


# Okay, so now we have an array of intensities and an array of corresponding
# values of |Q|. We could plot this as a scatter plot, but it will probably be
# easier to handle this data if we histogram it. Then we'll have nice,
# regularly spaced data. (Not to mention that your favourite plotting library
# will probably complain at you when you try to plot 2million points!)
import fast_histogram

# The beginning and end of the range over which we'll histogram.
start = float(np.min(q_lengths))
stop = float(np.max(q_lengths))

# The number of histogram bins to use. 1000 is usually reasonable.
num_bins = 1000

# Work out the binned intensities. Note that this isn't normalised by the
# number of times each bin is binned to; we have to do that manually.
final_intensities = fast_histogram.histogram1d(
    x=q_lengths,
    bins=num_bins,
    range=[start, stop],
    weights=intensities
)

# Work out how many times each bin was binned to for normalisation.
final_intensity_counts = fast_histogram.histogram1d(
    x=q_lengths,
    bins=num_bins,
    range=[start, stop],
)

# Carry out the normalisation.
final_intensities /= final_intensity_counts

# Compute an array of corresponding q vectors for plotting.
binned_qs = np.linspace(start, stop, num_bins)


# In[ ]:


import plotly.express as px

# Now we can plot intensity vs |Q|!
px.line(x=binned_qs, y=final_intensities).show()


# In[ ]:


# We can very easily save this data to a file as follows.
save_path = '/path/to/file.txt'

np.savetxt(save_path, np.c_[binned_qs, final_intensities])


# In[ ]:


"""
EXAMPLE 2: I(qxy, qz)

Here we carry out a qxy qz projection. There's a lot less fluff in this example
than the previous example - for a more basic introduction, first work through
example 1.
"""

# We're going to need the numpy library to do arithmetic and it'll help with
# loading in the data, too.
import numpy as np

# Replace these with paths to your files.
q_path = '/path/to/data/<name>_q.npy'
intensity_path = '/path/to/data/<name>_uncorrected_intensities.npy'

# Now let's load in that data.
intensities = np.load(intensity_path)
q_vectors = np.load(q_path)

# As before, let's reshape our q_vectors array to make it easier to work with.
q_vectors = q_vectors.reshape((intensities.shape[0], 3))

# Completely ignore all NaN intensity values and their corresponding q-vectors.
q_vectors = q_vectors[~np.isnan(intensities)]
intensities = intensities[~np.isnan(intensities)]

# Calculate qxy and qz from our q_vectors array.
qxy = np.sqrt(q_vectors[:, 0]**2 + q_vectors[:, 1]**2)
qz = q_vectors[:, 2]


# In[ ]:


# Now we have an array of intensities and an array of corresponding
# qxy and qz positions. Our life is probably going to be substantially simpler
# if we histogram this data so that it is regularly spaced in qxy and qz.
# As before, we can use the fast_histogram library to do this. As before, we
# need to calculate the smallest and largest qxy and qz values, so that we know
# between which bounds we're doing the binning. As before, we need to say how
# many bins we would like (but this time, we need to specify a number in each
# direction).
# 
# This may seem like a lot of information, but hopefully the code is self
# explanitory.
import fast_histogram

# Compute the min/max values of qxy and qz.
min_qxy = np.min(qxy)
max_qxy = np.max(qxy)
min_qz = np.min(qz)
max_qz = np.max(qz)

# Make a 1100x1000 qxy-qz map; I found 1000x1000 to give reasonable results.
num_qxy_bins = 1000
num_qz_bins = 1100

# Now run the binning. This is not normalised.
qxy_qz_intensities = fast_histogram.histogram2d(
    x=qxy,
    y=qz,
    bins=(num_qxy_bins, num_qz_bins),
    range=[[min_qxy, max_qxy], [min_qz, max_qz]],
    weights=intensities
)

# Work out how many times we binned into each pixel in the above routine.
qxy_qz_intensity_counts = fast_histogram.histogram2d(
    x=qxy,
    y=qz,
    bins=(num_qxy_bins, num_qz_bins),
    range=[[min_qxy, max_qxy], [min_qz, max_qz]]
)

# Normalise by the number of times we binned into each pixel.
qxy_qz_intensities /= qxy_qz_intensity_counts

# Also compute the corresponding Q values at all pixels.
# This has the shape (num_qxy_bins, ).
binned_qxy = np.linspace(min_qxy, max_qxy, num_qxy_bins)
# This has the shape (num_qz_bins, )
binned_qz = np.linspace(min_qz, max_qz, num_qz_bins)
# This has the shape (num_qxy_bins, num_qz_bins)
qxy_qz_q_vals = np.meshgrid(binned_qxy, binned_qz, indexing='ij')


# In[ ]:


import plotly.express as px

# Finally, make a plot of our computed qxy qz projection.
px.imshow(
    # Plot on a log scale.
    np.log(qxy_qz_intensities),
    x=binned_qxy, y=np.flip(binned_qz),
    labels=dict(x='Qxy', y='Qz'),
    origin='lower'
).show()

