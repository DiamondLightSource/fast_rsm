"""
First we need to import some stuff. 
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
This section requires action! Make sure you set all of the variables defined here.
"""

# How was your sample mounted? Options are 'horizontal', 'vertical' and 'DCD'.
setup = 'horizontal'

#which experimental hutch was used 0 = unsure, 1= experimental hutch 1, 2=experimental hutch 2
experimental_hutch=1
# Set local_data_path if your data isn't stored on the diamond system any more
# (for example if it's on a memory stick or scratch drive).
local_data_path = '/dls/i07/data/2024/si32266-3/QIchunfilms'
# Set this if you want to save the output somewhere other than the processing
# folder. Be warned, this could take up a lot of space.
local_output_path =  '/dls/i07/data/2024/cm37245-1/PhilMousley_testing/output'


# The beam centre, as can be read out from GDA, in pixel_x, pixel_y. If your
# map looks wacky, you probably cocked this up.
beam_centre = (119,1564)

# The distance between the sample and the detector (or, if using the DCD, the
# distance between the receiving slit and the detector). Units of meters.
detector_distance = 0.18

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



"""
**MASKING**

This section contains details on how to mask pixels. You can either mask a series
of individual pixels, mask rectangular regions of pixels, or dynamically mask
pixels based on their intensity (not recommended).
"""

# If you have a small number of hot pixels to mask, specify them one at a time
# in a list. In other words, it should look like:
# specific_pixels = [(pixel_x1, pixel_y1), (pixel_x2, pixel_y2)]
# Or, an exact example, where we want to mask pixel (233, 83) and pixel 
# (234, 83), where pixel coordinates are (x, y):
# 
specific_pixels = [(233, 234),(83, 83)]
# 
# Leave specific pixels as None if you dont want to mask any specific pixels.
# For this dataset we need to mask pixel (x=233, y=83)
# specific_pixels = None

# If you want to specify an entire region of pixels to mask, do so here.
# This is done using a "Region" object. To make a Region, give it start_x, 
# stop_x, start_y, start_y, as follows:
# 
# mask_1 = Region(0, 75, 0, 194)
# mask_2 = Region(425, 485, 0, 194)
#mask_3 = Region(1050, 1135, 0, 80)
#mask_4 = Region(1050, 1135, 245, 515)

# 
# If you don't want to use any mask regions, just leave mask_regions equal to
# None.
#mmask_regions = [mask_1, mask_2]
mask_regions = None

# Ignore pixels with an intensity below this value. If you don't want to ignore
# any pixels, then set min_intensity = None. This is useful for dynamically
# creating masks (which is really useful for generating masks from -ve numbers).
min_intensity = 0.

#for pyfai integration of an image, PONI and mask files need to be created
#first using pyfai calibration. Add the path locations to poni and mask files here
edfmaskfile =  '/home/i07user/fast_rsm/example_mask_Qichun.edf' #'/home/rpy65944/test2dFromImages.edf'


#define what outputs you would like form the processing here, choose from:
# 'full_reciprocal_map' = calculates a full reciprocal space map combining all
#                           scans listed into a single volume
#
# 'curved_projection_2D' = projects a series of detector images into a single 2D,
#                           treating the images as if there were all from a curved detector.

# 'pyfai_1D' =  Does an azimuthal integration on an image using PONI and MASK 
#               settings described in corresponding files
#
# 'save_binoculars_h5'= saved a hdf5 format as well as other output
#
# 'qperp_qpara_map'  - project GIWAXS image into q_para,q_perp plot
process_outputs=['qperp_qpara_map','pyfai_1D']#]#'qperp_qpara_map' ,'pyfai_1D']#'qperp_qpara_map','pyfai_1D']#'qperp_qpara_map','full_reciprocal_map']#'curved_projection_2D']#'pyfai_1D']#]#



# The scan numbers of the scans that you want to use to produce this reciprocal
# space map. 