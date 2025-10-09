"""
custom module to contain methods for creating custom a custom logger
"""
import logging
import logging.config
import getpass
import concurrent_log_handler



def start_frsm_loggers(version_path,debugflag: bool):
    """
    initiate loggers for fast_rsm - debug and error loggers
    """
    logging.config.fileConfig(\
        f'{version_path}/fast_rsm/src/fast_rsm/logging.ini',\
              disable_existing_loggers=False)
    if not debugflag:
        logging.getLogger('fastrsm_debug').setLevel(logging.CRITICAL)
    else:
        logging.getLogger('fastrsm_debug').setLevel(logging.DEBUG)


def get_debug_logger():
    """
    return already initialised debug logger
    """
    return logging.getLogger('fastrsm_debug')

def log_error_info(jobfile,slurmfile,error_logger):
    """
    pass error info to logger
    """
    error_logger.error(f"{getpass.getuser()}\t{jobfile}\t{slurmfile}")
