
"""
This section prepares the calculation. You probably shouldn't change any'qperp_qpara_map',thing here
unless you know what you're doing.
"""

from diffraction_utils import Frame, Region
from fast_rsm.diamond_utils import run_process_list,\
        make_globals_compatible,create_standard_experiment

from fast_rsm.logging_config import configure_logging,get_frsm_logger
import sys
import h5py

make_globals_compatible(globals())
configure_logging(DEBUG_LOG)
logger=get_frsm_logger()


experiment,num_threads,total_images,slitratios,oop=create_standard_experiment(globals())
"""
This section is for changing metadata that is stored in, or inferred from, the
nexus file. This is generally for more nonstandard stuff.
uncommend loop over scans to adjust metadata within scans
"""
# for i, scan in enumerate(experiment.scans):
#     """
#     area for making special adjustments to metadata information
#     """
#      # This is where you might want to overwrite some data that was recorded
#     # badly in the nexus file. See (commented out) examples below.
#     # scan.metadata.data_file.probe_energy = 12500
#     # scan.metadata.data_file.transmission = 0.4
#     # scan.metadata.data_file.using_dps = True
#     # scan.metadata.data_file.ub_matrix = np.array([
#     #     [1, 0, 0],
#     #     [0, 1, 0],
#     #     [0, 0, 1]
#     # ])


# Get the full path of the current file
full_path = __file__

f = open(full_path)
joblines = f.readlines()
f.close()
pythonlocation = sys.executable

# grab ub information
ubinfo = [scan.metadata.data_file.nx_instrument.diffcalchdr for scan in experiment.scans]

# check for deprecated GIWAXS functions and print message if needed
deplist = [print(deprecation_msg(output)) for output in process_outputs]
input_args=  [map_per_image,scan_numbers,local_output_path,joblines,num_threads,\
        ivqbins,qmapbins,pythonlocation,radialrange,radialstepval,slitratios,frame_name,oop,\
            output_file_size,coordinates,volume_start,volume_stop,volume_step,min_intensity,\
                save_path,total_images,save_binoculars_h5]
run_process_list(experiment,process_outputs,input_args)

print("PROCESSING FINISHED.")
