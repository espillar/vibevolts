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

def makeDetector(n, band, fov,ifov, aper,  \
                 qe = 0.5, photfrac=0.7, solarex = 20 * DEGREE, \
                 lunarex = 10 * DEGREE,  earthex= 15 * DEGREE):
    '''
    makeDetector takes parameters of a sensor and stuffs a filter array and a detector array, which it returns.
    I expect this function to be calld when a new satellite is created.
    '''
    filt, detect = makeBlankDetector(n)
    detect[:,APERTURE_IDX] = aper  #aperture size me
    detect[:,PIXEL_SIZE_IDX] = ifov  #pixel size rads
    detect[:,QE_IDX] = qe   # Total QE
    detect[:,PHOT_EFF_IDX] = photfrac  # fraction in photometry bucket
    pixels = (ifov/fov)**2 # pixels
    detect[:,PIXELS_IDX] = pixels   # total pixels in the array
    detect[:,SOLAR_EXCL_IDX] = solarex  # solar exclusion angle
    detect[:,LUNAR_EXCL_IDX] = lunarex  # lunar exclusion angle
    detect[:,EARTH_EXCL_IDX] = earthex  # eearth exclusion angle
    detect[:,SKY_BACK_IDX] = FILTER_DATA[band]['space'] # photon backgroud
    detect[:,FILTER_ZP_IDX] = FILTER_DATA[band]['zero_point'] # Filter Zero Point
    filt = [band] * n
    return(filt,detect)

                                                      
###########################################################
                                                  


def requiredIntegrationTime(limitingMag, SNR, filt,  d):
    '''
    requiredIntegrationTime(limitingMag, d)
    takes a two dimensional detector array ("detect")and calculates all the integration tiemes
    and returns those as a vector.
    The SNR is set to 
    '''
    t  = (SNR * SNR * d[:,SKY_BACK_IDX] * d[:,PIXEL_SIZE_IDX]) / \
       ( d[:,QE_IDX] *\
         d[:,PHOT_EFF_IDX]**2  *\
         math.pi * \
         (d[:,APERTURE_IDX]/2)**2 *\
         amag(limitingMag)**2 *\
         d[:,FILTER_ZP_IDX])
    return(t)

###########################################################

def testdetector():
    '''
    testdetector creates an example that can be compared
    with some of the stuff in Curio.
    '''
    filt, detect = makeDetector(1, "V", 1 * DEGREE,
                              1 * ARCSEC, 1, 
                              qe = 0.2, photfrac = 1.0)
    print( requiredIntegrationTime(20.5,10, filt, detect) )
    
