"""
module containing logic for switching between or calculations 
involving angles, pixels, and q values

"""

import numpy as np
from scipy.constants import physical_constants
from scipy.spatial.transform import Rotation as R
import transformations as tf
from types import SimpleNamespace



def calc_kout_array(outshape,i,j,detector_values,pixel_arrays):
    """
    calculates k_out values given detector position and image size
    """
    detector_displacement,detector_distance,detector_vertical,detector_horizontal=detector_values
    vertical_pixels,horizontal_pixels=pixel_arrays
    k_out_array = np.ndarray(outshape, np.float32)
    k_out_array[i, j, 0] = (
        detector_displacement.array[0] * detector_distance +
        detector_vertical.array[0] * vertical_pixels[i, j] +
        detector_horizontal.array[0] * horizontal_pixels[i, j])
    k_out_array[i, j, 1] = (
        detector_displacement.array[1] * detector_distance +
        detector_vertical.array[1] * vertical_pixels[i, j] +
        detector_horizontal.array[1] * horizontal_pixels[i, j])
    k_out_array[i, j, 2] = (
        detector_displacement.array[2] * detector_distance +
        detector_vertical.array[2] * vertical_pixels[i, j] +
        detector_horizontal.array[2] * horizontal_pixels[i, j])
    return k_out_array


#pull out functions that do not need to be part of Experiment class

def calctheta(q, wavelength):
    """
    converts two theta value to q value for a given wavelength

    Parameters
    ----------
    float
        q value in m^-1 to be converted to angle in degrees
    wavelength : float
        value of wavelength for incident radiation in angstrom.

    Returns
    -------

    twotheta : float
        value of angle in degrees converted to  q.
    """
    return np.degrees(np.arcsin(q * (wavelength / (4 * np.pi)) * 1e10)) * 2

def calcq( twotheta, wavelength):
    """
    converts two theta value to q value for a given wavelength

    Parameters
    ----------
    twotheta : float
        value of angle in degrees to be converted to  q.
    wavelength : float
        value of wavelength for incident radiation in angstrom.

    Returns
    -------
    float
        q value in m^-1.

    """
    return (4 * np.pi / wavelength) * \
        np.sin(np.radians(twotheta / 2)) * 1e-10

def calcqstep(gammastep, gammastart, wavelength):
    """
    calculates the equivalent q-step for a given 2theta step
    """
    qstep = calcq(gammastart + gammastep, wavelength) - \
        calcq(gammastart, wavelength)
    return qstep


def sohqcalc(angle, kmod):
    """
    calculates q value based on a opposite and hypotenuse of a
    right-angled triangle

    Parameters
    ----------
    angle : float
        angle of trig triangle being analysed.
    kmod : float
        wavevector value.

    Returns
    -------
    float
        q value.

    """
    return np.sin(np.radians(angle)) * kmod * 1e-10

def toa_ang_calc(opp,adj):
    return  np.degrees(np.arctan(opp / adj))

def pix_to_addminus(pixhigh,pixlow,pixscale,detector_distance,vertangles,axis):

    add_section = toa_ang_calc(pixhigh * pixscale,detector_distance)
    minus_section = toa_ang_calc(pixlow * pixscale,detector_distance)

    
    if axis == 'hor':
        maxvertrad = np.radians(np.max(vertangles))
        add_section = np.degrees(
            np.arctan(np.tan(np.radians(add_section)) / abs(np.cos(maxvertrad))))
        minus_section = np.degrees(
            np.arctan(np.tan(np.radians(minus_section)) / abs(np.cos(maxvertrad))))
    return add_section,minus_section


def gamdel2rots(gamma, delta):
    """
    Parameters
    ----------
    gamma : float
        angle rotation of gamma diffractometer circle in degrees.
    delta : float
        angle rotation of delta diffractometer circle in degrees.

    Returns
    -------
    rots : list of rotations rot1,rot2,rot3 in radians to be using by pyFAI.

    """
    rotdelta = R.from_euler('y', -delta, degrees=True)
    rotgamma = R.from_euler('z', gamma, degrees=True)
    totalrot = rotgamma * rotdelta
    fullrot = np.identity(4)
    fullrot[0:3, 0:3] = totalrot.as_matrix()
    vals = tf.euler_from_matrix(fullrot, 'rxyz')
    rots = vals[2], -vals[1], vals[0]
    return rots


# def get_bin_axvals(data_in, ind):
#     """
#     create axes information for binoviewer output in the form
#     ind,start,stop,step,startind,stopind
#     """
#     # print(data_in,type(data_in[0]))
#     single_list = [np.int64, np.float64, int, float]
#     if type(data_in[0]) in single_list:
#         data = data_in
#     else:
#         data = data_in[0]
#     startval = data[0]
#     stopval = data[-1]
#     stepval = data[1] - data[0]
#     startind = int(np.floor(startval / stepval))
#     stopind = int(startind + len(data) - 1)
#     return [ind, startval, stopval, stepval,
#             float(startind), float(stopind)]


def get_geometry_indices(setup,rotation):
    if rotation:
        rot_option = 'rot'
    else:
        rot_option = 'norot'

    chosen_setup = f'{setup}_{rot_option}'

    geometry_indices = {'vertical_rot': ['hor0', 'thvert'],
                    'vertical_norot': ['hor0', 'thvert'],
                    'DCD_rot': ['vert0', 'delvert'],
                    'DCD_norot': ['vert0', 'delvert'],
                    'horizontal_rot': ['vert0', 'delvert'],
                    'horizontal_norot': ['vert0', 'delvert'],
                    }

    return geometry_indices[chosen_setup]

def get_vert_pixang_limits(imshape,beam_centre,index,angles):
    pixlow = imshape[index] - beam_centre[index]
    pixhigh = beam_centre[index]
    highsection = np.max(angles)
    lowsection = np.min(angles)
    return pixlow,pixhigh,lowsection,highsection

def get_hor_pixang_limits(imshape,beam_centre,index,angles,setup,rotation):
    pixhigh = beam_centre[index]
    pixlow = imshape[index] - beam_centre[index]
    if (setup == 'vertical') & (
            rotation):
        pixhigh, pixlow = pixlow, pixhigh
    highsection = np.max(angles)
    lowsection = np.min(angles)
    return pixlow,pixhigh,lowsection,highsection

def get_correction_scales(setup,rotated):
    if (setup=='vertical') & (rotated):
        correctionscales = {'vert': -1, 'hor': -1} #GOOD
    elif setup=='vertical':
        correctionscales = {'vert': -1, 'hor': 1} # GOOD
    elif (setup == 'DCD') & (rotated):
        correctionscales = {'vert': -1, 'hor': -1} #GOOD
    elif setup == 'DCD':
        correctionscales = {'vert': -1, 'hor': -1} #GOOD
    elif (setup=='horizontal') & (rotated):
        correctionscales = {'vert': -1, 'hor': -1}#GOOD
    else:
        correctionscales = {'vert': -1, 'hor': -1}#GOOD
    return correctionscales

def get_pix_scale(pixel_size,slitvertratio,slithorratio,axis):
    if (slitvertratio is not None) & (axis == 'vert'):
        return pixel_size * slitvertratio
    elif (slithorratio is not None) & (axis == 'hor'):
        return pixel_size * slithorratio
    return pixel_size


def calc_qupplow_vert(maxangle,minangle,kmod,horangles,inc_angles):
    """
    calculates vertical extent in q based on verticle angle limits
    """

    maxanglerad=np.radians(maxangle)
    minanglerad=np.radians(minangle)
    qupp = sohqcalc(maxangle, kmod)  # *2
    qlow = sohqcalc(minangle, kmod)  # *2
    maxtthrad = np.radians(np.max(horangles))

    maxincrad = np.radians(np.max(inc_angles))
    extraincq = kmod * 1e-10 * np.sin(maxincrad)

    minusexitq_x = kmod * 1e-10 * \
        np.cos(maxanglerad) * np.cos(maxtthrad) * np.sin(maxincrad)
    minusexitq_z = kmod * 1e-10 * \
        np.sin(maxanglerad) * (1 - np.cos(maxincrad))
    extravert = extraincq - minusexitq_x - minusexitq_z
    qupp += extravert

    minusexitq_x = kmod * 1e-10 * \
        np.cos(minanglerad) * np.cos(maxtthrad) * np.sin(maxincrad)
    minusexitq_z = kmod * 1e-10 * \
        np.sin(minanglerad) * (1 - np.cos(maxincrad))
    extravert = extraincq - minusexitq_x - minusexitq_z
    qlow += extravert
    return qlow,qupp

def calc_qupplow_hor(maxangle,minangle,kmod,vertangles,vertindex,beam_centre,pixel_size,det_dist):
    maxanglerad=np.radians(maxangle)
    minanglerad=np.radians(minangle)
    qupp = sohqcalc(maxangle, kmod)  # *2
    qlow = sohqcalc(minangle, kmod)  # *2
    vertsign = [1 if np.sign(np.max(vertangles)) >= 0 else -1]
    maxvert = np.max(vertangles) + vertsign[0] * np.degrees(np.arctan(
        (beam_centre[vertindex] * pixel_size) / det_dist))
    maxvertrad = np.radians(maxvert)
    s1 = kmod * np.cos(maxvertrad) * np.sin(maxanglerad)
    s2 = kmod * (1 - np.cos(maxvertrad) * np.cos(maxanglerad))
    qupp_withvert = np.sqrt(
        np.square(s1) + np.square(s2)) * 1e-10 * np.sign(maxangle)
    s3 = kmod * np.cos(maxvertrad) * np.sin(minanglerad)
    s4 = kmod * (1 - np.cos(maxvertrad) * np.cos(minanglerad))
    qlow_withvert = np.sqrt(
        np.square(s3) + np.square(s4)) * 1e-10 * np.sign(minangle)

    if abs(qupp_withvert) > abs(qupp):
        qupp = qupp_withvert

    if abs(qlow_withvert) > abs(qlow):
        qlow = qlow_withvert 

    return qlow,qupp