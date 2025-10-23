"""
custom module to contain methods for creating custom a custom logger
"""
import shutil
import os
import logging
import logging.config
import getpass
import concurrent_log_handler



ERROR_LOG_DIR='/dls/science/groups/das/ExampleData/i07/fast_rsm_error_logs'
def start_frsm_loggers(version_path,debugflag: bool):
    """
    initiate loggers for fast_rsm - debug and error loggers
    """
    config_fname=f'{version_path}/fast_rsm/src/fast_rsm/logging.ini'
    logging.config.fileConfig(\
        config_fname,\
              disable_existing_loggers=False)
    if not debugflag:
        logging.getLogger('fastrsm_debug').setLevel(logging.CRITICAL)
    else:
        logging.getLogger('fastrsm_debug').setLevel(logging.DEBUG)
    return logging.getLogger('fastrsm_debug'),logging.getLogger('fastrsm_error')

def get_debug_logger():
    """
    return already initialised debug logger
    """
    return logging.getLogger('fastrsm_debug')

def get_error_logger():
    """
    return already initialised debug logger
    """
    return logging.getLogger('fastrsm_error')

def log_error_info(jobfile,slurmfile,error_logger):
    """
    pass error info to logger
    """
    copy_path=shutil.copy2(slurmfile,ERROR_LOG_DIR)
    os.chmod(copy_path, 0o777)
    error_logger.error(f"{getpass.getuser()}\t{jobfile}\t{slurmfile}")

def close_logging():
    '''
    closes all logging handlers and then shutsdown logging
    '''
    e_logger=get_error_logger()
    # Close all handlers
    for handler in e_logger.handlers[:]:
        handler.close()

    # Shutdown logging
    logging.shutdown()
