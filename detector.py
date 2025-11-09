from constants import *
import numpy as np
import math
import random
import radiometry_data
from radiometry_data import FILTER_DATA
from constants import *
from radiometry_calcs import mag, amag



##########################################################

def makeBlankDetector(n):
    detector = np.zeros((n,DETECTOR_ARRAY_SIZE), dtype=float)
    filt = [""] * n
    return(filt, detector)




##########################################################

def makeDetector(n, band, fov,ifov, aper, qe = 0.5, photfrac=0.7, solarex = 20 * DEGREE,   lunarex = 10 * DEGREE,  earthex= 15 * DEGREE):    
    '''
    makeDetector takes parameters of a sensor and stuffs a filter array and a detector array, which it returns.
    n is the number of sensors to produce
    band is the band the measurement takes place in (see radiometry_data)
    fov is the field of view- assumed square- in radians
    ifov is the pixel fov - assumed square - in radians
    aper is the aperture diameter - assumed round - in meters
    qe is th quantum efficiency of the system from entrance aperture
        to detectro
    photfrac is the fraction of the light captured in the photometry aperture
    solarex is the solar exclusion angle in radians
    lunarex is the lunar exclusion angle in radians
    earthex is the earth exclusion angle in radians

    This function is called when a new satellite is created.
    It uses the data from FILTER_DATA in radiometry_data.py, which is
    often in units of magnitudes,and 

    THIS VERSION IS FOR A GROUND OBSERVAOTRY
    '''
    filt, detect = makeBlankDetector(n)
    detect[:,APERTURE_IDX] = math.pi * (aper/2)**2  #aperture size square meters
    detect[:,PIXEL_SIZE_IDX] = math.pi * (ifov/2)**2  #pixel size sterradianss
    detect[:,QE_IDX] = qe   # Total QE
    detect[:,PHOT_EFF_IDX] = photfrac  # fraction in photometry bucket
    pixels = (ifov/fov)**2 # pixels
    detect[:,PIXELS_IDX] = pixels   # total pixels in the array
    detect[:,SOLAR_EXCL_IDX] = solarex  # solar exclusion angle
    detect[:,LUNAR_EXCL_IDX] = lunarex  # lunar exclusion angle
    detect[:,EARTH_EXCL_IDX] = earthex  # eearth exclusion angle
    detect[:,SKY_BACK_IDX] =  amag(FILTER_DATA[band]['sky']) * FILTER_DATA[band]['zero_point'] / (ARCSEC**2) # photon backgroud
    detect[:,FILTER_ZP_IDX] = FILTER_DATA[band]['zero_point'] # Filter Zero Point
    filt = [band] * n
    return(filt,detect)

                                                      
###########################################################
                                                  


def requiredIntegrationTime(limitingMag, SNR, filt,  d):
    '''
    requiredIntegrationTime(limitingMag, d)
    takes a two dimensional detector array ("detect")and calculates
          all the integration tiemes
    and returns those as a vector.
    For comparison with the equations paper, we first extract the variables
    to the conventional names used in that paper.
    '''
    gamma = SNR
    print(f"gamma {gamma:.2e}")
    beta = d[0 , SKY_BACK_IDX]
    print(f"beta is, {beta:.2e}")
    omega = d[: , PIXEL_SIZE_IDX]
    print("omega", omega)
    alpha = amag(limitingMag) * d[0, FILTER_ZP_IDX]
    print(f"alpha is {alpha:.2e}")
    A = d[:, APERTURE_IDX]
    print('A', A)
    eta = d[:, QE_IDX]
    print('eta', eta)
    f = d[:,PHOT_EFF_IDX]
    print('f', f)
    t = gamma**2 * beta * omega /( alpha**2 * A * eta * f**2)
#    t  = (SNR * SNR * d[:,SKY_BACK_IDX] * d[:,PIXEL_SIZE_IDX]) / \
#       ( d[:,QE_IDX] *\
#         d[:,PHOT_EFF_IDX]**2  *\
#         math.pi * \
#         (d[:,APERTURE_IDX]/2)**2 *\
#         amag(limitingMag)**2 *\
#         d[:,FILTER_ZP_IDX])
    return(t)
                 

###########################################################

def testdetector():
    '''
    testdetector creates an example that can be compared
    with some of the stuff in Curio.
    '''
    filt, detect = makeDetector(1, "V", 1 * DEGREE,
                              2 * ARCSEC, 1, 
                              qe = 0.2, photfrac = 1.0)
    print( requiredIntegrationTime(20.5,10, filt, detect) )
    
