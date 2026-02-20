"""
custom module to contain methods for creating custom a custom logger
"""
import shutil
import os
import sys
import traceback
import logging
import logging.config
import getpass
import concurrent_log_handler
import multiprocessing
import time as toptime
from logging.handlers import QueueListener, QueueHandler
import multiprocessing as mp


LOGGER_DEBUG = 'fastrsm_debug'
LOGGER_ERROR = 'fastrsm_error'


ERROR_LOG_DIR='/dls/science/groups/das/ExampleData/i07/fast_rsm_error_logs'


def do_time_check(outstring, queue=None,logn=None):
    
    pid = os.getpid()
    allowed = None
    if hasattr(os, "sched_getaffinity"):
        try:
            allowed = sorted(os.sched_getaffinity(0))  # e.g., [0] or [0,1,...,31]
        except BaseException:
            allowed = None

    cur = None
    if hasattr(os, "sched_getcpu"):
        try:
            cur = os.sched_getcpu()  # current CPU core ID
        except BaseException:
            cur = None

    t_wall = toptime.time()
    t_cpu  = toptime.process_time()
    return f"{outstring} pid={pid} allowed={allowed if allowed is not None else 'n/a'} current_cpu={cur} wall={t_wall:.6f} cpu={t_cpu:.6f}"

def listener_process(queue,configurer,log_name):
    """ Listener process is a target for a multiprocess process
    that runs and listens to a queue for logging events.

    Arguments:
        queue (multiprocessing.manager.Queue): queue to monitor
        configurer (func): configures loggers
        log_name (str): name of the log to use

    Returns:
        None
    """
    logger = configurer(log_name)

    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            print('Failure in listener_process', file=sys.stderr)
            traceback.print_last(limit=1, file=sys.stderr)

def start_frsm_loggers(version_path,debugflag: bool):
    """
    initiate loggers for fast_rsm - debug and error loggers
    """

    # Load your INI configuration
    config_fname = f'{version_path}/fast_rsm/src/fast_rsm/logging.ini'
    logging.config.fileConfig(config_fname, disable_existing_loggers=False)

    # Adjust debug level according to debugflag
    dbg_logger = logging.getLogger(LOGGER_DEBUG)
    err_logger = logging.getLogger(LOGGER_ERROR)
    dbg_logger.setLevel(logging.DEBUG if debugflag else logging.CRITICAL)

    # Create a multiprocessing Queue for log records
    log_queue = mp.Queue()

    # Collect handlers configured by your INI for both loggers
    # (these are the "final sinks" you already defined: ConcurrentRotatingFileHandler, etc.)
    target_handlers = []
    target_handlers.extend(dbg_logger.handlers)
    target_handlers.extend(err_logger.handlers)

    # # Start a QueueListener that forwards records from the queue to those handlers
    # listener = QueueListener(log_queue, *target_handlers)
    # listener.start()

    # manager = multiprocessing.Manager()
    # log_queue = manager.Queue()
    # listener = multiprocessing.Process(target=listener_process,
    #                                     args=(log_queue, get_logger, LOGGER_DEBUG))
    # listener.start()
    return err_logger,dbg_logger 
#,log_queue, listener


def get_logger(log_name):
    return logging.getLogger(log_name)

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
    error_logger.debug(f"{getpass.getuser()}\t{jobfile}\t{slurmfile}")

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
