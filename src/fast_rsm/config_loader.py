"""
module to load in process configurations and check against preset schemas
"""
import os
from schema import Or, And, Schema, SchemaError

import yaml
import numpy as np

valid_outs = [
    "full_reciprocal_map",
    "pyfai_ivsq",
    "pyfai_qmap",
    "pyfai_exitangles"]
giwaxsdeplist = [
    'curved_projection_2D',
    'pyfai_1D',
    'qperp_qpara_map',
    'large_moving_det',
    'pyfai_2dqmap_IvsQ']
# ====process functions


def invalid_msg(output):
    """
    return message for invalid process_output request
    """
    return f"requested output '{output}' not in valid output format.\
    \nValid formats are {valid_outs}"


def deprecation_msg(output):
    """
    give message for a deprecated process_output request
    """
    return f"option {output} has been deprecated. \
                GIWAXS mapping calculations now use pyFAI. \
            Please use process outputs 'pyfai_ivsq'  , 'pyfai_qmap'\
                    and 'pyfai_exitangles'"


def validate_outputs(outputs):
    """
    check list of deprecated functions, and print out warning message if needed
    """
    for output in outputs:
        if output in giwaxsdeplist:
            raise SchemaError(deprecation_msg(output))
        if output not in valid_outs:
            raise SchemaError(invalid_msg(output))
    return True


config_schema = Schema({
    "setup": str,
    "experimental_hutch": And(int, lambda n: n in (1, 2),
                              error="experimental hutch number needs to be 1 or 2"),
    "local_data_path": str,
    "local_output_path": str,
    "beam_centre": And(tuple, lambda n: np.shape(n) == (2,),
                       error="beam_centre not correct shape"),
    "detector_distance": float,
    "slitvertratio": Or(None, float),
    "slithorratio": Or(None, float),
    "alphacritical": float,
    "using_dps": bool,
    "dpsx_central_pixel": int,
    "dpsy_central_pixel": int,
    "dpsz_central_pixel": int,
    "output_file_size": int,
    "save_vtk": bool,
    "save_npy": bool,
    "volume_start": Or(None, list),
    "volume_stop": Or(None, list),
    "volume_step": Or(None, list),
    "load_from_dat": bool,
    "radialrange": tuple,
    "radialstepval": float,
    "qmapbins": tuple,
    "ivqbins": int,
    "edfmaskfile": Or(None, str),
    "specific_pixels": Or(None, list),
    "mask_regions": Or(None, list),
    "min_intensity": float,
    "skipscans": list,
    "skipimages": list,
    "process_outputs": And(list, lambda n: validate_outputs(n)),
    "map_per_image": bool,
    "savetiffs": bool,
    "savedats": bool,
    "frame_name": str,
    "coordinates": str,
    "cylinder_axis": bool,
    "DEBUG_LOG": int,
    "spherical_bragg_vec": And(list, lambda n: np.shape(n) == (3,),
                               error="bragg vector needs to be size 3 in the for [h,k,l]"),
    "default_config_path": str,
    "scan_numbers": list,
    "full_path": str,
})


def check_config_schema(input_config: dict):
    """
    check a process configuration against define scheme
    """
    try:
        config_schema.validate(input_config)
        print("config values loaded")
        return True
    except SchemaError as se:
        raise se


def experiment_config(scans, global_vals: SimpleNamespace):
    """
    create an Experiment instance using the default settings
    """
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the YAML file
    config_path = os.path.join(base_dir, "default_config.yaml")

    # Load the YAML file
    with open(config_path, "r",encoding='utf-8') as f:
        config_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
    config_dict['scan_numbers'] = scans
    config_dict['default_config_path'] = config_path

    return config_dict
