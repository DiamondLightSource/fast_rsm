# shared_mem_setup.py

from pyFAI import units

UNIT_IP = None
UNIT_OOP = None
# Synchronization & config
LOCK = None
CFG = None
LOGQ = None


def init_units(qip_name, qoop_name, sample_orientation):
    global UNIT_IP, UNIT_OOP
    UNIT_IP, UNIT_OOP = setup_ip_oop_units(
        qip_name,
        qoop_name,
        sample_orientation,
    )


def setup_ip_oop_units(qip_name, qoop_name, sample_orientation, inc_angle_out=0):
    unit_ip = units.get_unit_fiber(
        qip_name,
        sample_orientation=sample_orientation,
        incident_angle=inc_angle_out,
    )
    unit_oop = units.get_unit_fiber(
        qoop_name,
        sample_orientation=sample_orientation,
        incident_angle=inc_angle_out,
    )
    return unit_ip, unit_oop


def init_worker(global_cfg, global_log_queue):
    """
    Runs once per worker process. Sets the global CFG and LOGQ in this module.
    Since this function lives in shared_mem_setup.py, no imports are needed.
    """
    global CFG, LOGQ
    CFG = global_cfg
    LOGQ = global_log_queue


def combined_initializer(
    global_cfg,
    global_log_queue,
):
    """
    Pool initializer that sets both config and shared memory attachments
    in worker processes.
    Order matters: config first, then main SHM, then axis SHM.
    """

    try:
        init_worker(global_cfg, global_log_queue)

        init_units(
            global_cfg.unit_qip_name,
            global_cfg.unit_qoop_name,
            global_cfg.sample_orientation,
        )

    except Exception as e:
        print(f"combined initialised failed: {e}")
        raise
