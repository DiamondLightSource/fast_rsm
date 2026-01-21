import pytest
from schema import SchemaError
from fast_rsm.config_loader import validate_spherical_bragg,validate_beam_centre,validate_outputs,\
    experiment_config, check_config_schema, parse_setup_file


def test_spherical_bragg():
    bragg_vec1=[1,2,3]
    bragg_vec2=[1.2]
    assert validate_spherical_bragg(bragg_vec1) == True
    with pytest.raises(ValueError):
        validate_spherical_bragg(bragg_vec2)


def test_validate_beamcentre():
    bc1=[100,20]
    bc2=[12]
    assert validate_beam_centre(bc1)==True
    with pytest.raises(ValueError):
        validate_beam_centre(bc2)


def test_validateoutputs():
    outputs1=[    "full_reciprocal_map",
    "pyfai_ivsq",
    "pyfai_qmap",
    "pyfai_exitangles"]
    output2='pyfai_1D'
    output3='Nonesens'
    with pytest.raises(SchemaError):
        validate_outputs(output3)
    assert validate_outputs(outputs1)==True
    with pytest.raises(SchemaError):
        validate_outputs(output2)
    

def test_experimentconfig():
    scans=[123]
    testdict=experiment_config(scans)
    with pytest.raises(SchemaError):
        check_config_schema(testdict)
    testdict['beam_centre']=(0,0)
    with pytest.raises(SchemaError):
        check_config_schema(testdict)
    testdict['full_path']='path_for_testing'
    assert check_config_schema(testdict)== True


def test_parsesetup(path_to_resources):
    example1=path_to_resources / 'example_expsetup.py'
    testdict=parse_setup_file(example1)
    assert list(testdict.keys())==['val1', 'val2', 'mask_regions']
    assert testdict['val1']=='stringvalue'
    assert testdict['val2']==[1,2,3,4]
    assert testdict['mask_regions']==[(123, 456, 126, 457), (234, 456, 236, 458)]