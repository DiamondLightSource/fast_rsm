"""
custom module to contain methods for creating custom a custom logger
"""
import logging
import logging.handlers
import os

# Centralized logging flag
_logging_enabled = False
_log_path = os.path.join('/dls/science/groups/das/ExampleData/i07/fast_rsm_example_data', 'debug.log')

def configure_logging(enabled: bool):
    global _logging_enabled
    _logging_enabled = enabled

def get_frsm_logger(name: str):
    """
    Create or retrieve a logger for fast_rsm with consistent file logging.
    """
    logger = logging.getLogger(name)

    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        logger.setLevel(logging.DEBUG if _logging_enabled else logging.CRITICAL)
        file_handler = logging.handlers.RotatingFileHandler(
            _log_path, maxBytes=500000, backupCount=1
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if _logging_enabled:
        print(f'Logging enabled at {_log_path}')
        logger.info("Logger initialized")
    else:
        print("Logging disabled")

    return logger
