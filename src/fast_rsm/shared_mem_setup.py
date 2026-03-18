# shared_mem_setup.py

from multiprocessing.shared_memory import SharedMemory

import numpy as np

UNIT_IP = None
UNIT_OOP = None
# Synchronization & config
LOCK = None
CFG = None
LOGQ = None


# def init_units(qip_name, qoop_name, sample_orientation):
#     global UNIT_IP, UNIT_OOP
#     UNIT_IP, UNIT_OOP = setup_ip_oop_units(
#         qip_name,
#         qoop_name,
#         sample_orientation,
#     )


def init_worker(global_cfg, global_log_queue):
    """
    Runs once per worker process. Sets the global CFG and LOGQ in this module.
    Since this function lives in shared_mem_setup.py, no imports are needed.
    """
    global CFG, LOGQ
    CFG = global_cfg
    LOGQ = global_log_queue


def pyfai_init_worker(lock, shm_intensities_name, shm_counts_name, shmshape):
    """
    intialiser for pyfai mappings
    """
    global LOCK
    global SHM_INTENSITY
    global INTENSITY_ARRAY
    global SHM_COUNT
    global COUNT_ARRAY

    SHM_INTENSITY = SharedMemory(name=shm_intensities_name)
    SHM_COUNT = SharedMemory(name=shm_counts_name)
    INTENSITY_ARRAY = np.ndarray(
        shape=shmshape, dtype=np.float32, buffer=SHM_INTENSITY.buf
    )
    COUNT_ARRAY = np.ndarray(shape=shmshape, dtype=np.float32, buffer=SHM_COUNT.buf)
    LOCK = lock


def combined_initializer(
    lock,
    shm_intensities_name,
    shm_counts_name,
    shmshape,
):
    """
    Pool initializer that sets both config and shared memory attachments
    in worker processes.
    Order matters: config first, then main SHM, then axis SHM.
    """

    try:
        # init_worker(global_cfg, global_log_queue)

        # init_units(
        #     global_cfg.unit_qip_name,
        #     global_cfg.unit_qoop_name,
        #     global_cfg.sample_orientation,
        # )
        pyfai_init_worker(lock, shm_intensities_name, shm_counts_name, shmshape)

    except Exception as e:
        print(f"combined initialised failed: {e}")
        raise
