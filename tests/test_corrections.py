from fast_rsm.corrections import make_float32,get_corr_ints_kout,reshape_to_original
import numpy as np
from numpy.testing import assert_allclose

def test_make32():
    arr1=np.array([1,1.3,0,123])
    arr2=np.array([1.34,np.float64(1.2)])
    outarrs=make_float32([arr1,arr2])
    assert all([outarr.dtype==np.float32 for outarr in outarrs])


def test_getcorrints():
    intensities=np.ones((3,5))
    k_out=np.ones((3,5))
    out_ints,out_kouts,shapes=get_corr_ints_kout(intensities,k_out)
    assert_allclose(out_ints,np.ones((15)))
    assert_allclose(out_kouts,np.ones((5,3)))
    assert shapes==[(3,5),(3,5)]

def test_reshapeoriginal():
    shapes=[(3,5),(3,5)]
    intensities=np.ones(15)
    k_out=np.ones((5,3))
    out_ints,out_kouts=reshape_to_original(intensities,k_out,shapes)
    assert out_kouts.shape==(3,5)
    assert out_ints.shape==(3,5)
