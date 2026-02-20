from fast_rsm.diamond_utils import run_process_list,\
        setup_processing

from fast_rsm.logging_config import start_frsm_loggers



err_logger,dbg_logger=start_frsm_loggers(version_path,debuglogging)


#create experiment object, process configuration and logger
experiment,process_config,debug_logger=\
setup_processing(exp_file,__file__,scan_numbers)
#=================================================================================
####============SPECIAL ADJUSTMENTS ==============================================
# #This section is for changing metadata that is stored in, or inferred from, the
# #nexus file. This is generally for more nonstandard stuff.
# #uncomment loop over scans to adjust metadata within each scan
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
#=================================================================================
#=================================================================================

#
def main():
    """
    run processing jobs with process configuration settings requested
    """


    run_process_list(experiment,process_config)
    print("PROCESSING FINISHED.")


if __name__ == "__main__":
    main()
 