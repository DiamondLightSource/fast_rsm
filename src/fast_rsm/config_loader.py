"""
module to load in process configurations and check against preset schemas
"""
import os
import ast
import re
from pathlib import Path
from schema import Or, And, Schema, SchemaError, Optional
from simpleeval import SimpleEval
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

def validate_beam_centre(bc):
    """
    validate beamcentre is correct shape
    """
    if np.shape(bc)!=(2,):
        raise ValueError (f"beam_centre not correct shape. \
                          Got shape {np.shape(bc)} instead of (2,)")
    return True

def validate_spherical_bragg(bragg):
    """
    validate spherical bragg is correct shape
    """
    if np.shape(bragg)!=(3,):
        raise ValueError(f"bragg vector not correct shape. \
                         Got shape {np.shape(bragg)} instead of (3,)")
    return True

config_schema = Schema({
    "setup": str,
    "experimental_hutch": And(int, lambda n: n in (1, 2),
                              error="experimental hutch number needs to be 1 or 2"),
    "local_data_path": str,
    "local_output_path": str,
    "beam_centre": And(tuple, validate_beam_centre),
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
    "radialrange": Or(None,tuple),
    "radialstepval": Or(None,float),
    "qmapbins": tuple,
    "ivqbins": Or(None,int),
    "edfmaskfile": Or(None, str),
    "specific_pixels": Or(None, list),
    "mask_regions": Or(None, list, tuple),
    "min_intensity": float,
    "skipscans": list,
    "skipimages": list,
    "process_outputs": And(list, validate_outputs),
    "map_per_image": bool,
    "savetiffs": bool,
    "savedats": bool,
    "frame_name": str,
    "cylinder_axis": bool,
    "DEBUG_LOG": int,
    "spherical_bragg_vec": And(list, validate_spherical_bragg),
    "default_config_path": str,
    "scan_numbers": list,
    "full_path": str,
    "map_frame":Or(None,str),
    "coordinates":Or(None,str),
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

def check_node_list(nodevalue,assignments):
    """
    convert correctly list or tuple items read in from config
    """
    simp_eval=SimpleEval(names=assignments)
    remove_strings={ast.Tuple:"()",ast.List:"[]"}
    items=ast.unparse(nodevalue)
    try:
        outvals=ast.literal_eval(items)
        return outvals
    except ValueError:
        stripstring=remove_strings[type(nodevalue)]
        items_stripped=items.strip(stripstring).split(',')
        if all([len(item.strip())==0 for item in items_stripped]):
            return None
        outvals=[simp_eval.eval(item.strip())  if (len(item)>0) else None for item in items_stripped]
        if isinstance(nodevalue,ast.Tuple):
            return tuple(outvals)
        return outvals




def parse_setup_file(file_path: Path):
    """
    parse in settings from users exp_setup file
    """
    with open(file_path, 'r',encoding='utf-8') as f:
        tree = ast.parse(f.read())
    assignments = {}
    value_nodelist=[ast.Constant]
    #unparse_nodelist=[ast.Tuple,ast.List]
    deprecated_list=['save_binoculars_h5']

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        var_name = node.targets[0].id


        if var_name in deprecated_list:
            print(f'Warning: variable {var_name} is deprecated and will be ignored.\nPlease check https://fast-rsm.readthedocs.io/ for the latest exp_setup format')
            continue
        if any( isinstance(node.value,inst) for inst in value_nodelist):
            out_value = node.value.value  # int, float, str preserved
        else:
            out_value = check_node_list(node.value,assignments)

        assignments[var_name] = out_value
    #ignoring any individual mask, because it is unknown apriori how many the user has defined, and all used ones will be in mask_regions
    out_assignments={k:v for k,v in assignments.items() if len(re.findall(r"^mask_\d+$",k))==0}
    return out_assignments

def experiment_config(scans):
    """
    create an Experiment instance using the default settings
    """
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the YAML file
    config_path = os.path.join(base_dir, "default_config.yaml")

    # Load the YAML file
    with open(config_path, "r",encoding='utf-8') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dict['scan_numbers'] = scans
    config_dict['default_config_path'] = config_path

    return config_dict
