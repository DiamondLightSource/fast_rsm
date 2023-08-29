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

"""
**ESSENTIAL**

This cell requires action! Make sure you set all of the variables defined here.
"""

# How was your sample mounted? Options are 'horizontal', 'vertical' and 'DCD'.
setup = 'horizontal'

# Set local_data_path if your data isn't stored on the diamond system any more
# (for example if it's on a memory stick or scratch drive).
local_data_path = None
# Set this if you want to save the output somewhere other than the processing
# folder. Be warned, this could take up a lot of space.
local_output_path = None

# If you're processing on the cluster, you need to populate the next few fields.
# The experiment number, used to work out where your data is stored.
experiment_number = 'cm33856-2'

# The sub-directory containing your experimental data. Leave as None if unused.
# Otherwise, if the data was stored in a subdirectory called "day_1", e.g.
#   /dls/i07/data/2022/si32333-1/day_1/
# then you should use:
#   data_sub_directory = "day_1"
data_sub_directory = None

# The year the experiment took place.
year = 2023



# The beam centre, as can be read out from GDA, in pixel_x, pixel_y. If your
# map looks wacky, you probably cocked this up.
beam_centre = (955, 300)

# The distance between the sample and the detector (or, if using the DCD, the
# distance between the receiving slit and the detector). Units of meters.
detector_distance = 898e-3

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
cylinder_axis = 'None'

# Set this to True if you would like each image to be mapped independently.
# If this is False, all images in all scans will be combined into one large
# reciprocal space map.
map_per_image = False

# How large would you like your output file to be, in MB? 100MB normally gives
# very good resolution without sacrificing performance. If you want something
# higher resolution, feel free, but be aware that the performance of the map and
# the analysis will start to suffer above around 1GB.
# Max file size is 2GB (2048MB).
output_file_size = 200

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


# The scan numbers of the scans that you want to use to produce this reciprocal
# space map. 
