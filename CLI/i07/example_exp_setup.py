"""
First we need to import some stuff. 
If you're interested, each import has an associated comment that explains why
the import is useful/necessary.
"""

import os
import multiprocessing
from pathlib import Path
import numpy as np
from diffraction_utils import Frame, Region
from fast_rsm.experiment import Experiment

#=====================EXPERIMENTAL DETAILS============
# How was your sample mounted? Options are 'horizontal', 'vertical' and 'DCD'.
setup = 'horizontal'

#which experimental hutch was used 0 = unsure, 1= experimental hutch 1, 2=experimental hutch 2
experimental_hutch=1
# Set this to the directory path where your files are saved, note you will need to include any subdirectories in this path
local_data_path = 
# Set this to the path where you want the output from the data processing to be saved
local_output_path =  '/dls/i07/data/2024/cm37245-1/PhilMousley_testing/output'


# The beam centre, as can be read out from GDA, in pixel_x, pixel_y.
beam_centre = (119,1564)

# The distance between the sample and the detector (or, if using the DCD, the
# distance between the receiving slit and the detector). Units of meters.
detector_distance = 0.18


#========CALCULATION INFORMATION==========
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


#if calculating pyfai integration on scan with moving detector and large number of images need to 
# specify range of q or 2th so that number of bins can be calculated
radialrange=(0,60)
radialstepval=0.01

#===========MASKING=============
#add path to edfmaskfile created with pyFAI gui accessed via 'makemask' option in fast_rsm
edfmaskfile =  None


# alternatively specify masked regions with pixels and regions
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

# to allow saving masks to hdf5 file, creating regions was moved to calc_setup, 
# here give just start_x,  stop_x, start_y, start_y, as follows:
# 
mask_1 =(0, 75, 0, 194)
mask_2 = (425, 485, 0, 194)

# 
# If you don't want to use any mask regions, just leave mask_regions equal to
# None.
mmask_regions = [mask_1, mask_2]

# Ignore pixels with an intensity below this value. If you don't want to ignore
# any pixels, then set min_intensity = None. This is useful for dynamically
# creating masks (which is really useful for generating masks from -ve numbers).
min_intensity = 0.

skipscans=[1235,\
           1237]

skipimages=[[4,12],\
            [45,67,69]]


#============OUTPUTS==========
#define what outputs you would like form the processing here, choose from:
# 'full_reciprocal_map' = calculates a full reciprocal space map combining all
#                           scans listed into a single volume
#
# 'curved_projection_2D' = projects a series of detector images into a single 2D,
#                           treating the images as if there were all from a curved detector. 
#                           NOTE: Currently does not work for scans with 1000s of images

# 'pyfai_1D' =  Does an azimuthal integration on an image using PONI and MASK 
#               settings described in corresponding files
#
# 'qperp_qpara_map'  - project GIWAXS image into q_para,q_perp plot. 
#                          NOTE: Currently does not work for scans with 1000s of images
#
# 'large_moving_det' - utilise MultiGeometry option in pyFAI for scan with a moving detector and a 
#                       large number of images (~1000s), outputs: I, Q, two theta, caked image 
#
process_outputs=['large_moving_det']
qmapbins=(350,1050)

# The scan numbers of the scans that you want to use to produce this reciprocal
# space map. 