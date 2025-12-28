from constants import *
import numpy as np
import math
from radiometry_data import FILTER_DATA
from types import SimpleNamespace
from radiometry_calcs import *


##########################################################

def makeBlankDetector(n):
    """
    makeBlankDetector makes and returns a detector SipleNamespace
    with parameters
    aperture
    pixelArea
    qe
    photoEff
    pixCount
    solarEx
    lunarEx
    earlEx
    skyBack
    zpCal
    itime
    fov
    ifov
    filt
    """
    detector = SimpleNamespace()
    detector.aperture = np.zeros(n, dtype=float)  # Aperture area in square meters
    detector.pixelArea = np.zeros(n, dtype=float) # pixel area in square arcsec
    detector.qe = np.zeros(n, dtype=float)        # Quantum efficiency from aperture to detectoras a fraction (0.0 to 1.0)
    detector.photoEff = np.zeros(n, dtype=float)   # Fraction of photons in photometry bucket
    detector.pixCount = np.zeros(n, dtype=float)   # Total number of pixels in the detector (count)
    detector.solarEx = np.zeros(n, dtype=float)     # Solar exclusion angle in radians
    detector.lunarex = np.zeros(n, dtype=float)    # Lunar exclusion angle in radians
    detector.earthEx = np.zeros(n, dtype=float)    # Earth exclusion angle (above the limb) in radians
    detector.skyBack = np.zeros(n, dtype=float)    # Sky Background in photons per square steradian
    detector.zpCal = np.zeros(n, dtype=float)  # Filter calibration zeropoint" photons per square meter per second second
    detector.itime = np.zeros(n, dtype=float)      # Integration Time required to reach a desired limiting magniude
    detector.fov = np.zeros(n, dtype=float)       
    detector.ifov = np.zeros(n, dtype=float)      
    detector.filt = [""] * n                       # filter
    return detector



##########################################################

def makeDetector(n, band, fov, ifov, aper, qe = 0.5, photfrac=0.7, \
     solarex= 20.0 * DEGREE,   lunarex= 10.0 * DEGREE,  earthex= 15.0 * DEGREE):
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
    detect = makeBlankDetector(n)
    detect.aperture[:] = math.pi * (aper/2)**2  #aperture size square meters
    detect.pixelArea[:] = math.pi * (ifov/2)**2  #pixel size sterradianss
    detect.qe[:] = qe   # Total QE
    detect.photoEff[:] = photfrac  # fraction in photometry bucket
    pixels = (ifov/fov)**2 # pixels
    detect.pixCount[:] = pixels   # total pixels in the array
    detect.solarEx[:] = solarex  # solar exclusion angle
    detect.lunarex[:] = lunarex  # lunar exclusion angle
    detect.earthEx[:] = earthex  # eearth exclusion angle
    detect.skyBack[:] =  amag(FILTER_DATA[band]['sky']) * FILTER_DATA[band]['zero_point'] / (ARCSEC**2) # photon backgroud
    detect.zpCal[:] = FILTER_DATA[band]['zero_point'] # Filter Zero Point
    detect.itime[:] = requiredIntegrationTime(20, 4, detect)
    detect.fov[:] = fov
    detect.ifov[:] = ifov
    detect.filt = [band] * n
    return detect


###########################################################



def requiredIntegrationTime(limitingMag, SNR, d, debug = 0):
    '''
    requiredIntegrationTime(limitingMag, d)
    takes a two dimensional detector array ("detect")and calculates
          all the integration tiemes
    and returns those as a vector.
    For comparison with the equations paper, we first extract the variables
    to the conventional names used in that paper.
    '''
    gamma = SNR
    beta = d.skyBack[0]
    omega = d.pixelArea
    alpha = amag(limitingMag) * d.zpCal[0]
    A = d.aperture
    eta = d.qe
    f = d.photoEff
    if debug == 1:
        print(f"gamma {gamma:.2e}")
        print(f"beta is, {beta:.2e}")
        print("omega", omega)
        print(f"alpha is {alpha:.2e}")
        print('A', A)
        print('eta', eta)
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
    detect = makeDetector(1, "V", 1 * DEGREE,
                              2 * ARCSEC, 1,
                              qe = 0.2, photfrac = 1.0)
    print( requiredIntegrationTime(20.5,10, detect, debug=1) )

if __name__ == '__main__':
    testdetector()
