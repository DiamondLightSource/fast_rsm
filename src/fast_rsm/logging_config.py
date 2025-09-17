"""
custom module to contain methods for creating custom a custom logger
"""

import logging
import logging.handlers
import os



# Internal flag (not global across modules)
_logging_enabled = 0

def configure_logging(enabled: bool):
    global _logging_enabled
    _logging_enabled = enabled


def get_frsm_logger(name: str):
    """
    create custom logger for debugging fast_rsm
    """

    # Create fast_rsm logger
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.WARNING)
    if _logging_enabled ==1:
        log_path = os.path.join('/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data', 'debug.log')
        #Set root logger to WARNING to suppress third-party debug/info logs
        logger.setLevel(logging.INFO)
        # Add a rotating file handler to fast_rsm logger only
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=500000, backupCount=1 )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        print(f'logging at {log_path}')
        logger.info("test info line")
        logger.debug("test debug line")
    else:
        print("logging disabled")
        logger.disabled=True


    return logger
