from constants import *
import numpy as np
import math
from radiometry_data import FILTER_DATA
from types import SimpleNamespace
from radiometry_calcs import *
from pointing import update_detector_pointing, generate_pointing_sphere
import random


#########################################################

def setDetectorFOV(sim_data, fovSize):
    """
    setDetectorFOV goes through the detectors in sim_data
    and changes the FOVs of all of them to size (radians).
    This is meant to be an ad-hoc function for test,
    not a regular operational thing.
    """
    count = len(sim_data['detector'].fov)
    sim_data['detector'].fov = np.full(count, fovSize)

    #########################################################

def setDetectorIntegrationTime(sim_data, itime):
    """
    setDetectorFOV goes through the detectors in sim_data
    and changes the FOVs of all of them to size (radians).
    This is meant to be an ad-hoc function for test,
    not a regular operational thing.
    """
    count = len(sim_data['detector'].fov)
    sim_data['detector'].itime = np.full(count, itime)


##########################################################

def makeBlankDetector(n):
    """
    makeBlankDetector makes and returns a detector SipleNamespace
    with parameters
    apertureArea
    pixelArea
    qe
    photoEff
    pixCount
    solarEx
    lunarEx
    earlEx
    skyBack
    zpCal
    integrationTime
    fov
    ifov
    filt
    pointing = (n,3) vectors where you are pointing
    pointing_state (n,2) length of chain and current index
    """
    detector = SimpleNamespace()
    detector.apertureArea = np.zeros(n, dtype=float)
        # Aperture area in square meters
    detector.pixelArea = np.zeros(n, dtype=float)
        # pixel area in square arcsec
    detector.qe = np.zeros(n, dtype=float)
    # Quantum efficiency from apertureArea to detectoras a fraction (0.0 to 1.0)
    detector.photoEff = np.zeros(n, dtype=float)
    # Fraction of photons in photometry bucket
    detector.pixCount = np.zeros(n, dtype=float)
    # Total number of pixels in the detector (count)
    detector.solarEx = np.zeros(n, dtype=float)
    # Solar exclusion angle in radians
    detector.lunarex = np.zeros(n, dtype=float)
    # Lunar exclusion angle in radians
    detector.earthEx = np.zeros(n, dtype=float)
    # Earth exclusion angle (above the limb) in radians
    detector.skyBack = np.zeros(n, dtype=float)
    # Sky Background in photons per square steradian
    detector.zpCal = np.zeros(n, dtype=float)
    # Filter calibration zeropoint" photons per square meter per second second
    detector.integrationTime = np.zeros(n, dtype=float)
    # Integration Time required to reach a desired limiting magniude
    detector.fov = np.zeros(n, dtype=float)
    detector.ifov = np.zeros(n, dtype=float)
    detector.filt = [""] * n                       # filter
    detector.pointing = np.zeros((n,3), dtype = float)
    detector.pointing_state = np.zeros((2,n),dtype=int)
    return detector

##########################################################

def makeDetector(n, band, fov, ifov, aper, qe = 0.5, photfrac=0.7, \
     solarex= 20.0 * DEGREE,   lunarex= 10.0 * DEGREE,  earthex= 15.0 * DEGREE):
    '''
    makeDetector takes parameters of a sensor and stuffs a filter array and a detector array, which it returns.
    n is the number of sensors to produce
    band is the band the measurement takes place in (see radiometry_data)
    fov is the field of view- assumed square?- in radians
    ifov is the pixel fov - assumed square - in radians
    aper is the apertureArea diameter - assumed round - in meters
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
    detect.apertureArea[:] = math.pi * (aper/2)**2  #aperture size square meters
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
    detect.integrationTime[:] = requiredIntegrationTime(20, 4, detect)
    detect.fov[:] = fov
    detect.ifov[:] = ifov
    detect.filt = [band] * n
    detect.pointing = np.zeros((n,3), dtype = float)
    detect.pointing_state = np.zeros((2,n),dtype=int)
    return detect



###########################################################

def detectorPointingInitialize(sim_data, grid_points):
    """
    We assume that sim_data['detectors'] loaded, but the
    pointing part of detectors is currently empty.
    pointing and pointing_state inside detect are initialized, and
    also adding a pointing sphere to sim_data.
    """


    # BROKEN the number of points should be the number of sensors
    sensorCount = len(sim_data['detector'].filt)
    generate_pointing_sphere(sim_data, grid_points)
    detect = sim_data['detector']
    detect.pointing_state = np.zeros((2,sensorCount), dtype = int)
#    print( 'detect.pointing_state.shape ', detect.pointing_state.shape)
    detect.pointing_state[POINTING_COUNT_IDX,:] = grid_points
    detect.pointing_state[POINTING_PLACE_IDX,:] = np.random.randint(0, grid_points-1, size=sensorCount)
    update_detector_pointing(sim_data)

    
    
###########################################################



def requiredIntegrationTime(limitingMag, SNR, d, debug = 0):
    '''
    Calculates the required integration time to achieve a given limiting magnitude with a specified signal-to-noise ratio (SNR).

    This function is based on the radiometric equation, solving for the integration time 't'.
    The calculation is derived from the following relationship:
    SNR = (Signal) / sqrt(Signal + Background)
    t = SNR^2 * (beta * omega) / (alpha^2 * A * eta * f)

    For comparison with an external document ("equations paper"), the function
    extracts variables from the detector object 'd' and assigns them to conventional names.

    Args:
        limitingMag (float): The desired limiting magnitude.
        SNR (float): The target signal-to-noise ratio.
        d (SimpleNamespace): A detector object containing the required parameters.
        debug (int, optional): If set to 1, prints the intermediate variables
                               used in the calculation. Defaults to 0.

    Returns:
        float: The required integration time in seconds.
    '''
    gamma = SNR
    beta = d.skyBack[0]
    omega = d.pixelArea
    alpha = amag(limitingMag) * d.zpCal[0]
    A = d.apertureArea
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
    t = gamma**2 * beta * omega /( alpha**2 * A * eta * f)
#    t  = (SNR * SNR * d[:,SKY_BACK_IDX] * d[:,PIXEL_SIZE_IDX]) / \
#       ( d[:,QE_IDX] *\
#         d[:,PHOT_EFF_IDX]**2  *\
#         math.pi * \
#         (d[:,APERTURE_AREA_IDX]/2)**2 *\
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
