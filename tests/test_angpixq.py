
import pytest

from fast_rsm.angle_pixel_q import get_geometry_indices,get_vert_pixang_limits,get_hor_pixang_limits,pix_to_addminus

import numpy as np


geometries=[['vertical',True,['hor0', 'thvert']],
            ['vertical',False,['hor0', 'thvert']],
            ['DCD',True,['vert0', 'delvert']],
            ['DCD',False,['vert0', 'delvert']],
            ['horizontal',True,['vert0', 'delvert']],
            ['horizontal',False, ['vert0', 'delvert']]
            ]
geometry_test_params=[pytest.param(*setting,id=f'geometry_{n}') for n,setting in enumerate(geometries)]

@pytest.mark.parametrize("setup,rot,checkvals",geometry_test_params)
def test_geometry_indices(setup,rot,checkvals):
    assert get_geometry_indices(setup,rot)==checkvals


vertpixang_settings=[[(200,100),[25,60],0,np.arange(10),(175,25,0,9)]
                 ]
vertpixang_test_params=[pytest.param(*setting,id=f'vertpixang_{n}') for n,setting in enumerate(vertpixang_settings)]
@pytest.mark.parametrize("imshape,beam_centre,index,angles,checkvals",vertpixang_test_params)
def test_vertpixang_limits(imshape,beam_centre,index,angles,checkvals):
    assert get_vert_pixang_limits(imshape,beam_centre,index,angles)==checkvals



horpixang_settings=[[(200,100),[25,60],1,np.arange(10),'horizonal',False,(40,60,0,9)],
                    [(200,100),[25,60],1,np.arange(10),'vertical',False,(40,60,0,9)],
                    [(200,100),[25,60],1,np.arange(10),'vertical',True,(60,40,0,9)]
                 ]
horpixang_test_params=[pytest.param(*setting,id=f'horpixang_{n}') for n,setting in enumerate(horpixang_settings)]
@pytest.mark.parametrize("imshape,beam_centre,index,angles,setup,rotation,checkvals",horpixang_test_params)
def test_horpixang_limits(imshape,beam_centre,index,angles,setup,rotation,checkvals):
    assert get_hor_pixang_limits(imshape,beam_centre,index,angles,setup,rotation)==checkvals


addminus_settings=[
    [60,40,172e-6,0.89,0,'vert',(0.6643437593515902,0.442906866534868)]
]
addminus_test_params=[pytest.param(*setting,id=f'addminus_{n}') for n,setting in enumerate(addminus_settings)]
@pytest.mark.parametrize("pixhigh,pixlow,pixscale,detector_dist,vertangles,axis,checkvals",
                         addminus_test_params)
def test_pix_to_addminus(pixhigh,pixlow,pixscale,detector_dist,vertangles,axis,checkvals):
    assert pix_to_addminus(pixhigh,pixlow,pixscale,detector_dist,vertangles,axis)==checkvals
