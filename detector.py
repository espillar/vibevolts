from dataclasses import dataclass, field
from typing import List
from constants import *
import numpy as np
import math
from minimalsimulation import DictDataclass
from radiometry_data import FILTER_DATA
from radiometry_calcs import *


@dataclass
class DetectorArray(DictDataclass):
    """
    DetectorArray models detector parameters for all satellites.

    It inherits from DictDataclass so it supports both attribute-style
    (dot notation) and dictionary subscript access, and participates in
    the SimulationState dataclass hierarchy.  All array-valued fields
    have a length equal to the number of detectors (satellites).

    Fields:
        apertureArea:    Entrance-aperture area in m^2 (1-D array).
        pixelOmega:      Pixel solid angle in sr, assuming square pixels
                         (1-D array).
        qe:              Total system quantum efficiency from entrance
                         aperture to detector (1-D array).
        photoEff:        Fraction of light captured in the photometry
                         aperture (1-D array).
        pixCount:        Number of pixels in the detector array (1-D).
        solarEx:         Solar exclusion angle in radians (1-D array).
        lunarEx:         Lunar exclusion angle in radians (1-D array).
        earthEx:         Earth limb exclusion angle in radians (1-D array).
        skyBack:         Sky background flux in photons/s/m^2/sr (1-D).
        zpCal:           Filter zero-point calibration (1-D array).
        integrationTime: Integration time in seconds (1-D array).
        fov:             Full field-of-view diameter in radians (1-D).
        ifov:            Pixel field-of-view in radians (1-D array).
        filt:            Filter band name for each detector (list of str).
        pointing:        Pointing unit vector for each detector,
                         shape (n, 3).
        pointing_state:  Pointing scheduler state, shape (2, n).
                         Row 0 = total grid points, row 1 = current index.
    """
    apertureArea: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    pixelOmega: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    qe: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    photoEff: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    pixCount: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    solarEx: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    lunarEx: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    earthEx: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    skyBack: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    zpCal: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    integrationTime: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    fov: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    ifov: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float))
    filt: List[str] = field(default_factory=list)
    pointing: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=float))
    pointing_state: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 0), dtype=int))


def setDetectorFOV(sim_data, fovSize):
    """
    setDetectorFOV goes through the detectors in sim_data
    and changes the FOVs of all of them to size (radians).
    This is meant to be an ad-hoc function for test,
    not a regular operational thing.
    """
    det = sim_data.detector if hasattr(sim_data, 'detector') else sim_data['detector']
    count = len(det.fov)
    det.fov = np.full(count, fovSize)


def setDetectorIntegrationTime(sim_data, itime):
    """
    setDetectorIntegrationTime goes through the detectors in sim_data
    and changes the integration times of all of them to itime (seconds).
    This is meant to be an ad-hoc function for test,
    not a regular operational thing.
    """
    det = sim_data.detector if hasattr(sim_data, 'detector') else sim_data['detector']
    count = len(det.fov)
    det.integrationTime = np.full(count, itime)


##########################################################

def makeBlankDetector(n: int) -> DetectorArray:
    """
    makeBlankDetector makes and returns a DetectorArray
    with parameters initialized to zero arrays of length n.
    """
    return DetectorArray(
        apertureArea=np.zeros(n, dtype=float),
        pixelOmega=np.zeros(n, dtype=float),
        qe=np.zeros(n, dtype=float),
        photoEff=np.zeros(n, dtype=float),
        pixCount=np.zeros(n, dtype=float),
        solarEx=np.zeros(n, dtype=float),
        lunarEx=np.zeros(n, dtype=float),
        earthEx=np.zeros(n, dtype=float),
        skyBack=np.zeros(n, dtype=float),
        zpCal=np.zeros(n, dtype=float),
        integrationTime=np.zeros(n, dtype=float),
        fov=np.zeros(n, dtype=float),
        ifov=np.zeros(n, dtype=float),
        filt=[""] * n,
        pointing=np.zeros((n, 3), dtype=float),
        pointing_state=np.zeros((2, n), dtype=int)
    )

##########################################################

def makeDetector(n, band, fov, ifov, aper, intTime: float = 1.0,
                 qe=0.5, photfrac=0.7,
                 solarex=20.0 * DEGREE, lunarex=10.0 * DEGREE,
                 earthex=15.0 * DEGREE):
    '''
    makeDetector takes parameters of a sensor and stuffs a filter array
    and a detector array, which it returns.

    Args:
        n:        Number of sensors to produce.
        band:     Band the measurement takes place in (see radiometry_data).
        fov:      Field of view - assumed square - in radians.
        ifov:     Pixel fov - assumed square - in radians.
        aper:     Aperture diameter - assumed to be a disk - in meters.
                  The actual value stored and used is the apertureArea.
        intTime:  Integration time in seconds.  Defaults to 1.0.
        qe:       Quantum efficiency of the system from entrance aperture
                  to detector.
        photfrac: Fraction of light captured in the photometry aperture.
        solarex:  Solar exclusion angle in radians.
        lunarex:  Lunar exclusion angle in radians.
        earthex:  Earth limb exclusion angle in radians.

    THIS VERSION IS FOR A GROUND OBSERVATORY.
    '''
    detect = makeBlankDetector(n)
    detect.apertureArea[:] = math.pi * (aper / 2) ** 2
    detect.pixelOmega[:] = ifov ** 2
    detect.qe[:] = qe
    detect.photoEff[:] = photfrac
    pixels = (ifov / fov) ** 2
    detect.pixCount[:] = pixels
    detect.solarEx[:] = solarex
    detect.lunarEx[:] = lunarex
    detect.earthEx[:] = earthex
    detect.skyBack[:] = (
        amag(FILTER_DATA[band]['sky'])
        * FILTER_DATA[band]['zero_point']
        / (ARCSEC ** 2)
    )
    detect.zpCal[:] = FILTER_DATA[band]['zero_point']
    detect.integrationTime[:] = intTime
    detect.fov[:] = fov
    detect.ifov[:] = ifov
    detect.filt = [band] * n
    detect.pointing = np.zeros((n, 3), dtype=float)
    detect.pointing_state = np.zeros((2, n), dtype=int)
    return detect


def requiredIntegrationTime(limitingMag, SNR, d, debug=0):
    '''
    Calculates the required integration time to achieve a given limiting
    magnitude with a specified signal-to-noise ratio (SNR).

    Args:
        limitingMag (float): The desired limiting magnitude.
        SNR (float):         The target signal-to-noise ratio.
        d (DetectorArray):   A detector object containing the required
                             parameters.
        debug (int, optional): If set to 1, prints the intermediate
                               variables used in the calculation.
                               Defaults to 0.

    This function is based on the radiometric equation, solving for
    the integration time t:
        SNR = (Signal) / sqrt(Signal + Background)
        t = SNR^2 * (beta * omega) / (alpha^2 * A * eta * f^2)

    Returns:
        float: The required integration time in seconds.
    '''
    gamma = SNR
    beta = d.skyBack[0]
    omega = d.pixelOmega
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
    t = gamma ** 2 * beta * omega / (alpha ** 2 * A * eta * (f ** 2))
    return t


def appendDetector(cd: DetectorArray, new_cd: DetectorArray) -> None:
    """
    Appends the attributes of new_cd to the existing detector object cd
    in-place.

    Iterates over all declared DetectorArray dataclass fields.  Numpy
    1-D arrays are concatenated with np.append; 2-D arrays are joined
    with np.vstack (or np.hstack for pointing_state which is (2, n));
    list fields are extended.
    """
    for attr_name in DetectorArray.__dataclass_fields__:
        new_val = getattr(new_cd, attr_name)
        cur_val = getattr(cd, attr_name)
        if isinstance(cur_val, np.ndarray):
            if cur_val.ndim == 1:
                setattr(cd, attr_name, np.append(cur_val, new_val))
            elif cur_val.ndim == 2:
                if attr_name == 'pointing_state':
                    setattr(cd, attr_name, np.hstack([cur_val, new_val]))
                else:
                    setattr(cd, attr_name, np.vstack([cur_val, new_val]))
        elif isinstance(cur_val, list):
            cur_val.extend(new_val)


###########################################################

def testdetector():
    '''
    testdetector creates an example that can be compared
    with some of the stuff in Curio.
    '''
    detect = makeDetector(1, "V", 1 * DEGREE,
                          2 * ARCSEC, 1,
                          qe=0.2, photfrac=1.0)
    print(requiredIntegrationTime(20.5, 10, detect, debug=1))

if __name__ == '__main__':
    testdetector()
