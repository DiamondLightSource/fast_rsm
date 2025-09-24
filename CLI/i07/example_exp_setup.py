"""
First we need to import some stuff.
"""

import os
import multiprocessing
from pathlib import Path
import numpy as np
from diffraction_utils import Frame, Region
from fast_rsm.experiment import Experiment

# =====================EXPERIMENTAL DETAILS============
# How was your sample mounted? Options are 'horizontal', 'vertical' and 'DCD'.
setup = 'horizontal'

# which experimental hutch was used 0 = unsure, 1= experimental hutch 1,
# 2=experimental hutch 2
experimental_hutch = 1
# Set this to the directory path where your files are saved, note you will
# need to include any subdirectories in this path
local_data_path = 'path'  # '/dls/i07/data/2024/##experiment-number##/##subfolder#
# Set this to the path where you want the output from the data processing
# to be saved
local_output_path = 'path'  # '/dls/i07/data/2024/##experiment-number##/processing'


# The beam centre, as can be read out from GDA, in pixel_x, pixel_y.
beam_centre = (119, 1564)

# The distance between the sample and the detector (or, if using the DCD, the
# distance between the receiving slit and the detector). Units of meters.
detector_distance = 0.18

# if not using sample slits leave both as None, if using slits set to
# slit-detector/sample-detector  e.g. 0.55/0.89
slitvertratio = 0.55 / 0.89  # None
slithorratio = None

# critical edge of sample in degrees
alphacritical = 0.08

# Are you using the DPS system?
using_dps = False
# The DPS central pixel locations are not typically recorded in the nexus file.
# NOTE THAT THIS SHOULD BE THE CENTRAL PIXEL FOR THE UNDEFLECTED BEAM.
# UNITS OF METERS, PLEASE (everything is S.I., except energy in eV).
dpsx_central_pixel = 0
dpsy_central_pixel = 0
dpsz_central_pixel = 0

# ========CALCULATION INFORMATION==========

# **************FULL RECIPROCAL VOLUME -  CRYSTAL TRUNCATION RODS, HKL MAPS etc
# How large would you like your output file to be, in MB? 100MB normally gives
# very good resolution without sacrificing performance. If you want something
# higher resolution, feel free, but be aware that the performance of the map and
# the analysis will start to suffer above around 1GB.
# Max file size is 2GB (2048MB).
output_file_size = 50

#Choose if you want a .vtk volume saved as well the hdf5, which can be used for loading into paraview
save_vtk = False

#Choose if you want a .npy file saved as well as the hdf5, for manual analysis
save_npy = False

# choose map co-ordinates for special mappings e.g. polar co-ordinates, if commented out defaults to co-ordinates='cartesian'
# coordinates='sphericalpolar'

# choose central point to calculate spherical polars around - if commented out defaults to [0,0,0]
# spherical_bragg_vec=[1.35,1.42,0.96] #519910 , 519528


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


# ********GIWAXS WITH MOVING DETECTOR AND LARGE NUMBER OF IMAGES TO BE COMBINED INTO ONE INTEGRATION
# if calculating pyfai integration on scan with moving detector and large number of images, need to
# specify range of q or 2th so that number of bins can be calculated
radialrange = (0, 60)
radialstepval = 0.01

# *********calculating qpara Vs qperp maps,
# set number of bins in the form (q_parallel, q_perpendicular)
qmapbins = (1200, 1200)

# *******calculating azimuthal integrations from single images to give I Vs Q plots
# number of bins in Q
ivqbins = 1000

# ===========MASKING=============
# add path to edfmaskfile created with pyFAI gui accessed via 'makemask'
# option in fast_rsm
edfmaskfile = None


# alternatively specify masked regions with pixels and regions
# If you have a small number of hot pixels to mask, specify them one at a time
# in a list. In other words, it should look like:
# specific_pixels = [(pixel_x1, pixel_y1), (pixel_x2, pixel_y2)]
# Or, an exact example, where we want to mask pixel (233, 83) and pixel
# (234, 83), where pixel coordinates are (x, y):
#
specific_pixels = None  # [(233, 234),(83, 83)]
#
# Leave specific pixels as None if you dont want to mask any specific pixels.
# For this dataset we need to mask pixel (x=233, y=83)
# specific_pixels = None

# to allow saving masks to hdf5 file, creating regions was moved to calc_setup,
# here give just start_x,  stop_x, start_y, start_y, as follows:
#
mask_1 = (0, 75, 0, 194)
mask_2 = (425, 485, 0, 194)

#
# If you don't want to use any mask regions, just leave mask_regions equal to
# None.
mask_regions = None

# Ignore pixels with an intensity below this value. If you don't want to ignore
# any pixels, then set min_intensity = None. This is useful for dynamically
# creating masks (which is really useful for generating masks from -ve
# numbers).
min_intensity = 0.

# =======OPTIONS FOR SKIPPING IMAGES IF ISSUES ARE PRESENT
# CHOOSE SCANS WHICH HAVE IMAGES TO SKIP, AND THEN SPECIFY WHICH IMAGES WITHIN THOSE SCANS NEED TO BE SKIPPED
# I.E. A LIST OF IMAGES TO SKIP FOR EACH SCAN VALUE IN SKIPSCANS
skipscans = []

skipimages = [[],
              []]

# ============OUTPUTS==========
# define what outputs you would like form the processing here, choose from:
# 'full_reciprocal_map' = calculates a full reciprocal space map combining all
#                           scans listed into a single volume
#
# 'pyfai_qmap' = calculates 2d q_parallel Vs q_perpendicular plots using pyFAI
#
# 'pyfai_ivsq' = calculates 1d Intensity Vs Q using pyFAI
#
# 'pyfai_exitangles' - calculates a map of vertical exit angle Vs horizontal exit angle

# 'pyfai_ivsq'  , 'pyfai_qmap','pyfai_exitangles' ,'full_reciprocal_map'
process_outputs = []


# Set this to True if you would like each image to be mapped independently.
# If this is False, all images in all scans will be combined into one large
# reciprocal space map.
map_per_image = False

# There will always be a .hdf5 file created. You can set the option for exporting additonal files with the savetiffs and savedats options below
# if you want to export '2d qpara Vs qperp maps' to extra .tiff images set
# savetiffs to True
savetiffs = False

# if you want to export '1d I Vs Q data' to extra .dat files set savedats
# to True
savedats = False

# The scan numbers of the scans that you want to use to produce this reciprocal
# space map.
