
import pytest
from fast_rsm.angle_pixel_q import get_geometry_indices,get_vert_pixang_limits,get_hor_pixang_limits,\
    pix_to_addminus,gamdel2rots,calcq,calctheta,calcqstep,sohqcalc,get_correction_scales,get_pix_scale,\
        calc_qupplow_vert,calc_qupplow_hor,calc_kout_array
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_allclose


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


#add in option for 'hor' with vertangle - double check calculations
addminus_settings=[
    [60,40,172e-6,0.89,0,'vert',(0.6643437593515902,0.442906866534868)],
    [60,40,172e-6,0.89,0,'hor',(0.6643437593515902,0.442906866534868)]
]
addminus_test_params=[pytest.param(*setting,id=f'addminus_{n}') for n,setting in enumerate(addminus_settings)]
@pytest.mark.parametrize("pixhigh,pixlow,pixscale,detector_dist,vertangles,axis,checkvals",
                         addminus_test_params)
def test_pix_to_addminus(pixhigh,pixlow,pixscale,detector_dist,vertangles,axis,checkvals):
    assert pix_to_addminus(pixhigh,pixlow,pixscale,detector_dist,vertangles,axis)==checkvals

getcorr_settings=[
    ['vertical',True,{'vert': -1, 'hor': -1}],
    ['vertical',False,{'vert': -1, 'hor': 1}],
    ['DCD',True,{'vert': -1, 'hor': -1}],
    ['DCD',False,{'vert': -1, 'hor': -1}],
    ['horizontal',True,{'vert': -1, 'hor': -1}],
    ['horizontal',False,{'vert': -1, 'hor': -1}],

]
getcorr_params=[pytest.param(*setting,id=f'getcorr_{n}') for n,setting in enumerate(getcorr_settings)]
@pytest.mark.parametrize("setup,rotated,checkvals",getcorr_params)
def test_getcorrscales(setup,rotated,checkvals):
    assert get_correction_scales(setup,rotated)==checkvals


pixscale_settings=[
    [0.1,2,1,'vert',0.2],
    [0.1,1,2,'hor',0.2],
    [0.1,None,None,'hor',0.1]
]
pixscale_params=[pytest.param(*setting,id=f'pixscale_{n}') for n,setting in enumerate(pixscale_settings)]
@pytest.mark.parametrize("pixelsize,vertratio,horratio,axis,checkvals",pixscale_params)
def test_getpixscale(pixelsize,vertratio,horratio,axis,checkvals):
    assert get_pix_scale(pixelsize,vertratio,horratio,axis)==checkvals


def test_gamdel2rots():
    rots1=(0,np.radians(10),0)
    assert_allclose(gamdel2rots(0,10),rots1)
    rots2=(np.radians(10),0,0)
    assert_allclose(gamdel2rots(10,0),rots2)
    rots3=(0.17716979849676953, 0.17185475131873923, 0.030609295710743036)
    assert_allclose(gamdel2rots(10,10),rots3)


def test_calcq():
    assert calcq(3.00,0.6)==5.482487233160938e-11


def test_calctheta():
    assert_allclose(calctheta(5.482487233160938e-11,0.6),3.00)

def test_calcqstep():
    assert_allclose(calcqstep(0.05,3.00,0.6),1e-10*0.009135338570864203)

def test_sohqcalc():
    assert_allclose(sohqcalc(15,4),1.035276180410083e-10)

def test_calcqupplowvert():
    testvertvals1=(np.float64(1.9567012263483676e-14), np.float64(3.850050174880243e-10))
    assert_allclose(calc_qupplow_vert(25,0,2*np.pi/0.69,np.arange(-10,10),0.1),testvertvals1)

def test_calcquplowhor():
    testhorvals1=(np.float64(-7.981259908293286e-11), np.float64(3.917774960094374e-10))
    assert_allclose(calc_qupplow_hor(25,-5,2*np.pi/0.69,np.arange(-10,10),0,[25,50],172e-6,0.89),testhorvals1)


def test_calckout():
    imshape=(5,4)
    outshape=(imshape[0],imshape[1],3)
    detector_displacement=SimpleNamespace(array=[0,0,1])
    detector_distance=0.89
    detector_vertical=  SimpleNamespace(array=[0,1,0])
    detector_horizontal=  SimpleNamespace(array=[1,0,0])
    vert_pixels=np.tile([-1, 0, 1, 2], (5, 1))*172e-6
    hor_pixels=np.arange(5).reshape(5,1).repeat(4, axis=1)*172e-6
    detector_values=[detector_displacement,detector_distance,detector_vertical,detector_horizontal]
    pixel_arrays=[vert_pixels,hor_pixels]
    i,j=slice(None),slice(None)
    out_array=calc_kout_array(outshape,i,j,detector_values,pixel_arrays)
    x_vals=np.array([[0.      , 0.      , 0.      , 0.      ],
        [0.000172, 0.000172, 0.000172, 0.000172],
        [0.000344, 0.000344, 0.000344, 0.000344],
        [0.000516, 0.000516, 0.000516, 0.000516],
        [0.000688, 0.000688, 0.000688, 0.000688]])
    assert_allclose(out_array[:,:,0],x_vals)
    y_vals=np.array([[-0.000172,  0.      ,  0.000172,  0.000344],
        [-0.000172,  0.      ,  0.000172,  0.000344],
        [-0.000172,  0.      ,  0.000172,  0.000344],
        [-0.000172,  0.      ,  0.000172,  0.000344],
        [-0.000172,  0.      ,  0.000172,  0.000344]])
    assert_allclose(out_array[:,:,1],y_vals)
    z_vals=np.array([[0.89, 0.89, 0.89, 0.89],
        [0.89, 0.89, 0.89, 0.89],
        [0.89, 0.89, 0.89, 0.89],
        [0.89, 0.89, 0.89, 0.89],
        [0.89, 0.89, 0.89, 0.89]])
    assert_allclose(out_array[:,:,2],z_vals)