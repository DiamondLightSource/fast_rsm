"""
custom module to contain methods for creating custom a custom logger
"""
import logging
import logging.config
import getpass

_LOGGING_ENABLED = False


def configure_logging(enabled: bool):
    """
    set logger either enabled or disabled
    """
    global _LOGGING_ENABLED
    print(f"change logging to {enabled}")
    _LOGGING_ENABLED = enabled


def start_frsm_loggers(version_path):
    """
    initiate loggers for fast_rsm - debug and error loggers
    """
    logging.config.fileConfig(\
        f'{version_path}/fast_rsm/src/fast_rsm/logging.ini',\
              disable_existing_loggers=False)
    debug_logger=logging.getLogger('fastsm_debug')
    error_logger=logging.getLogger('fastsm_error')
    if not _LOGGING_ENABLED:
        debug_logger.setLevel(logging.CRITICAL)
    return debug_logger,error_logger

def get_debug_logger():
    """
    return already initialised debug logger
    """
    return logging.getLogger('fastsm_debug')

def log_error_info(jobfile,slurmfile,error_logger):
    """
    pass error info to logger
    """
    error_logger.error(f"{getpass.getuser()}\t{jobfile}\t{slurmfile}")