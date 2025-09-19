
from fast_rsm.diamond_utils import run_process_list,\
        create_standard_experiment,experiment_config

# Get the full path of the current file
full_path = __file__

default_config=experiment_config(scan_numbers)
for key,val in default_config.items():
    if key in globals():
        setattr(default_config,key,globals()[key])
#create experiment object
experiment,process_config=create_standard_experiment(default_config)


"""
This section is for changing metadata that is stored in, or inferred from, the
nexus file. This is generally for more nonstandard stuff.
uncomment loop over scans to adjust metadata within each scan
"""
# for i, scan in enumerate(experiment.scans):
#     """
#     area for making special adjustments to metadata information
#     """
# #      This is where you might want to overwrite some data that was recorded
# #     badly in the nexus file. See (commented out) examples below.
#     scan.metadata.data_file.probe_energy = 12500
#     scan.metadata.data_file.transmission = 0.4
#     scan.metadata.data_file.using_dps = True
#     scan.metadata.data_file.ub_matrix = np.array([
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1]
#     ])

run_process_list(experiment,process_config)
print("PROCESSING FINISHED.")
