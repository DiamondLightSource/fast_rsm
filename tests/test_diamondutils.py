from types import SimpleNamespace
#from fast_rsm.pyfai_interface import *
from fast_rsm.diamond_utils import initial_value_checks,colour_text

def test_default_scans(test_default_config: dict):
    assert test_default_config['scan_numbers']==[1234]


def test_colour_text():
    assert colour_text('blue','hello')=='\033[34mhello\033[0m'
    assert colour_text('green','hello')=='\033[32mhello\033[0m'
    assert colour_text('red','hello')=='\033[31mhello\033[0m'

def test_initial_checks(test_default_config: dict):
    outlist=['cylinder_axis','setup','output_file_size']
    cfg=SimpleNamespace(**test_default_config)
    dps_centres= [cfg.dpsx_central_pixel,cfg.dpsy_central_pixel,cfg.dpsz_central_pixel]
    assert initial_value_checks(dps_centres,cfg.cylinder_axis,cfg.setup,cfg.output_file_size)=='y'

# def test_get_functions(test_default_config: dict):
#     static_functions={'pyfai_qmap':[pyfai_static_qmap,"Qmap","2d Qmap"],\
#                       'pyfai_exitangles':[pyfai_static_exitangles,"exitmap", "2d exit angle map"],\
#                         'pyfai_ivsq':[pyfai_static_ivsq,"IvsQ","1d integration "]}
    
#     moving_functions={'pyfai_qmap':[pyfai_moving_qmap_smm,"Qmap","2d Qmap"],\
#                       'pyfai_exitangles':[pyfai_moving_exitangles_smm,"exitmap", "2d exit angle map"],\
#                         'pyfai_ivsq':[pyfai_moving_ivsq_smm,"IvsQ","1d integration "]}
#     cfg=SimpleNamespace(**test_default_config)
#     cfg.map_per_image=False
#     assert get_run_functions(cfg)[0]==moving_functions
#     assert get_run_functions(cfg)[1]==run_scanlist_combined
#     cfg.map_per_image=True
#     assert get_run_functions(cfg)[0]==static_functions
#     assert get_run_functions(cfg)[1]==run_scanlist_loop
    


    

