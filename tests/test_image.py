from fast_rsm.image import get_coordchange_matrix,do_mask_pixels,do_mask_regions,\
    correct_transmission,do_mask_detris,correct_counttime,do_edfmask
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_allclose

# def test_processing_steps():
#     fake_metadata = SimpleNamespace(
#         image_loader=lambda idx: np.ones((5,5)),
#         solid_angles=np.ones((5,5)),
#         get_count_time=lambda idx: 1,
#         data_file=SimpleNamespace(is_dectris=False),
#         edfmask=None,
#         mask_pixels=None,
#         mask_regions=None,
#         diffractometer=SimpleNamespace(data_file=SimpleNamespace(is_dectris=False,transmission=1))
#     )

#     img = Image(fake_metadata, 0)


def test_get_coordchange():
    yvals=np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
    xvals= np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
    zvals=np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    assert_allclose(get_coordchange_matrix('x'),xvals)
    assert_allclose(get_coordchange_matrix('y'),yvals)
    assert_allclose(get_coordchange_matrix('z'),zvals)
    assert get_coordchange_matrix('random') is None

def test_maskpixels():
    testarr=np.ones((4,4))
    maskedarr=testarr.copy()
    maskedarr[0,2]=np.nan
    maskedarr[2,1]=np.nan
    pixels=(0,2),(2,1)
    assert_allclose(do_mask_pixels(pixels,testarr),maskedarr)

def test_maskregions():
    testarr=np.ones((4,4))
    maskedarr=testarr.copy()
    maskedarr[2:0,0:2]=np.nan
    maskedarr[2:4,0:3]=np.nan
    region1=SimpleNamespace(slice=np.s_[2:0,0:2])
    region2=SimpleNamespace(slice=np.s_[2:4,0:3])
    region_list=[region1,region2]
    assert_allclose(do_mask_regions(region_list,testarr),maskedarr)

def test_transmission_correction():
    testarr=np.ones((4,4))*5
    normed_arr=np.ones((4,4))
    transmission1=np.array([5])
    assert_allclose(correct_transmission(transmission1,testarr,0),normed_arr)
    transmission2=np.array([1,1,5,1])
    assert_allclose(correct_transmission(transmission2,testarr,2),normed_arr)
    transmission3=[1,1,5,1]
    assert_allclose(correct_transmission(transmission3,testarr,2),normed_arr)
    transmission4=None
    assert_allclose(correct_transmission(transmission4,testarr,2),testarr)

def test_counttime_correction():
    testarr=np.ones((4,4))*5
    normed_arr=np.ones((4,4))
    counttime1=np.array([5])
    assert_allclose(correct_counttime(counttime1,testarr,0),normed_arr)
    counttime2=np.array([1,1,5,1])
    assert_allclose(correct_counttime(counttime2,testarr,2),normed_arr)
    counttime3=[1,1,5,1]
    assert_allclose(correct_counttime(counttime3,testarr,2),normed_arr)
    counttime4=None
    assert_allclose(correct_counttime(counttime4,testarr,2),testarr)

def test_mask_dectris():
    testarr=np.ones((4,4))
    maskedarr=testarr.copy()
    testarr[(2,0)]=4294967300.0
    testarr[(0,3)]=4294967300.0
    maskedarr[(2,0)]=np.nan
    maskedarr[(0,3)]=np.nan
    assert_allclose(do_mask_detris(testarr),maskedarr)


def test_mask_edf():
    testarr=np.ones((4,4))
    testarr*=5
    maskedarr=testarr.copy()
    maskedarr[2,0]=np.nan
    maskedarr[0,3]=np.nan
    mask=np.zeros((4,4))
    mask[2,0]=1
    mask[0,3]=1
    assert_allclose(do_edfmask(mask,testarr),maskedarr)



print('DONE')
