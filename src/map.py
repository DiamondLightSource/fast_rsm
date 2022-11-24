#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
First we need to import some stuff. Feel free to ignore this cell.

If you're interested, each import has an associated comment that explains why
the import is useful/necessary.
"""

# Used for constructing paths.
from pathlib import Path
# Used to work out how much time the processing took.
from time import time

# Essential for all mathematical operations we'll be carrying out.
import numpy as np

# diffraction_utils is a library developed at Diamond by Richard Brearton
# (richard.brearton@diamond.ac.uk) to ease the task of parsing data files and
# carrying out some common calculations. Here, we'll be using it to define
# frames of reference, and parse nexus files.
from diffraction_utils import Frame, I07Nexus

# The following imports are required for the core of the calculation code, also
# written by Richard Brearton (richard.brearton@diamond.ac.uk).
# These functions are used to calculate the region of reciprocal space that was
# sampled during the scan, and to calculate the step size we should use to
# achieve the file size requested.
from fast_rsm.meta_analysis import get_step_from_filesize
# This is the central Scan object, which stores all the logic related to
# individual scans.
from fast_rsm.scan import Scan
# This function will be used to save our data so that we can open it in
# Paraview.
from fast_rsm.writing import linear_bin_to_vtk


# In[ ]:


"""
This cell requires action! Make sure you set all of the variables defined here.
"""

# What was your scattering geometry/how was your sample mounted? Options are
# 'horizontal', 'vertical' and 'DCD'.
setup = 'horizontal'

# The experiment number, used to work out where your data is stored.
experiment_number = 'si32333-1'

# The sub-directory containing your experimental data. Leave as None if unused.
# Otherwise, if the data was stored in a subdirectory called "day_1", e.g. 
#   /dls/i07/data/2022/si32333-1/day_1/
# then you should use:
#   data_sub_directory = "day_1"
data_sub_directory = None

# The year the experiment took place.
year = 2022

# The scan numbers of the scans that we want to use to produce this reciprocal
# space map. For example, the default value of scan_numbers shows how to specify
# every scan between number 421772 and 421778 inclusive, but skipping scan
# number 421776.
scan_numbers = [421772, 421773, 421774, 421775, 421777, 421778]

# Uncomment the following to set scan_numbers equal to every scan number between
# scan_start and scan_stop:
# scan_start = 421772
# scan_stop = 421775
# scan_numbers = list(range(scan_start, scan_stop + 1))

# The beam centre, as can be read out from GDA, in pixel_x, pixel_y. If your
# map looks wacky, you probably cocked this up.
beam_centre = (769, 1330)

# The distance between the sample and the detector (or, if using the DCD, the
# distance between the receiving slit and the detector). Units of meters.
detector_distance = 485e-3

# The frame/coordinate system you want the map to be carried out in. 
# Options for frame_name argument are:
#     Frame.hkl     (map into hkl space - requires UB matrix in nexus file)
#     Frame.sample_holder   (standard map into 1/Å)
#     Frame.lab     (map into frame attached to lab. I dont think you want this)
# 
# Options for coordinates argument are:
#     Frame.cartesian   (normal cartesian coords: hkl, Qx Qy Qz, etc.)
#     Frame.polar       (cylindrical polar with cylinder axis along l/Qz)
# 
# Frame.polar will give an output like a more general version of PyFAI.
# Frame.cartesian is for hkl maps and Qx/Qy/Qz. Any combination of frame_name
# and coordinates will work, so try them out; get a feel for them.
frame_name = Frame.hkl
coordinates = Frame.cartesian

# Ignore pixels with an intensity below this value. If you don't want to ignore
# any pixels, then set min_intensity = None. This is ueful for dynamically
# creating masks.
min_intensity = None

# How large would you like your output file to be, in MB? 100MB normally gives
# very good resolution without sacrificing performance. If you want something
# higher resolution, feel free, but be aware that the performance of the map and
# the analysis will start to suffer above around 1GB.
output_file_size = 100


# In[ ]:


"""
This cell prepares the calculation. You probably shouldn't change anything here
unless you know what you're doing.
"""

# Max number of cores on a Hamilton cluster node.
num_threads = 40

# Work out where the data is. 
data_dir = Path(f"/dls/i07/data/{year}/{experiment_number}/")

# Store this for later.
processing_dir = data_dir / "processing"
if data_sub_directory is not None:
    data_dir /= Path(data_sub_directory)

# You can make this what you like, but note that same datetime data will be
# inserted to ensure that your output file has a unique name.
save_name = "mapped_scan_"

# Construct the Frame object from the user's preferred frame/coords.
map_frame = Frame(frame_name=frame_name, coordinates=coordinates)


# In[ ]:


"""
This cell contains all of the logic for running the calculation. You shouldn't
run this on your local computer, it'll either raise an exception or take
forever.
"""

# These keep track of where the processing output of individual scans will be
# saved.
data_file_names = []
normalisation_file_names = []

# First work out start/stop/step.
t1 = time()
print("Calculating q-bounds...\r", end='')
starts, stops = [], []
for scan_number in scan_numbers:
    print(f"Calculating q-bounds for scan number {scan_number}.\r", end='')
    path_to_nx = data_dir / f"{scan_number}.nxs"
    scan = Scan.from_i07(path_to_nx, beam_centre, detector_distance, setup,
                         path_to_nx)
    start, stop = scan.q_bounds(map_frame)
    starts.append(start)
    stops.append(stop)

starts, stops = np.array(starts), np.array(stops)
start, stop = np.min(starts, axis=0), np.max(stops, axis=0)
step = get_step_from_filesize(start, stop, output_file_size)

print(f"Binning reciprocal space from {start} to {stop} in steps of " +
      f"{step}.")

print(f"Took {(time() - t1)*1000} ms to calculate q bounds.")

# Now that we've worked out where we are in reciprocal space, actually do
# the mapping.
for scan_number in scan_numbers:
    t1 = time()
    path_to_nx = data_dir / f"{scan_number}.nxs"

    scan = Scan.from_i07(path_to_nx, beam_centre, detector_distance, setup,
                            [0, 1, 0], path_to_nx)

    i07nexus = I07Nexus(path_to_nx)
    print(f"Beginning mapping of the {i07nexus.scan_length} images in "
            f"scan number {scan_number} with {num_threads} cores.")
    rsmap, counts = scan.binned_reciprocal_space_map(
        map_frame, start, stop, step, min_intensity, num_threads)

    t2 = time()
    time_taken = t2-t1
    time_per_image = time_taken/scan.metadata.data_file.scan_length*1000

    print(f"Time taken per image: {time_per_image} ms")
    print(f"Time taken to process whole scan: {time_taken} s")

    # Save the map and the normalisation separately.
    save_data = save_name + str(scan_number)
    save_norm = save_name + f"normalisation_{scan_number}"
    data_file_names.append(save_data)
    normalisation_file_names.append(save_norm)
    np.save(save_data, rsmap)
    np.save(save_norm, counts)

print("Mapping completed. Commencing normalisation procedure...")

# Now iterate over all of the maps and normalise the total map.
total_map = np.zeros_like(rsmap, dtype=np.float32)
total_counts = np.zeros_like(counts, dtype=np.float32)
for i, map_name in enumerate(data_file_names):
    total_map += np.load(map_name + '.npy')
    total_counts += np.load(normalisation_file_names[i] + '.npy')

normalised_map = total_map / total_counts
vtk_name = data_file_names[0] + "_normalised"
linear_bin_to_vtk(normalised_map, data_file_names[0], start, stop, step)

print("Map normalised and saved to vtk. Exiting...")
exit()


# In[2]:


"""
This is the cell that you should execute to run this notebook on the cluster.
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

# Convert this notebook to a python script in the current dir.
get_ipython().system('jupyter nbconvert --to script map.ipynb')

# Copy this script to a central location.
move_from = "map.py"
# move_to = "/dls_sw/apps/fast_rsm/current/scripts/map.py"
move_to = "src/map.py"
subprocess.run(["cp", move_from, move_to])

# Submit the job, which in the shell (in the right directory) would be done as:
# qsub -pe smp 40 -l m_mem_free=1.6G -P i07 cluster_job.sh
subprocess.run(["qsub", "-pe", "smp", "40", "-l", 
                f"m_mem_free={output_file_size/300}G", "-P", "i07",
                "/dls_sw/apps/fast_rsm/current/scripts/cluster_job.sh"])


# In[ ]:


"""
The following cells contain tools for interacting with and visualising data.
"""


# In[ ]:




