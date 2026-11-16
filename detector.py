import math
from typing import Any
import numpy as np

from constants import ARCSEC, DEGREE
from minimalsimulation import ComponentState
from radiometry_calcs import FILTER_DATA, amag


class DetectorArray(ComponentState):
    """
    DetectorArray models detector parameters for a collection of assets.

    Supports both attribute-style (dot notation) and dictionary subscript
    access. All array-valued fields have length `n` along axis 0.

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
        pointing:        Pointing unit vector for each detector, shape (n, 3).
        pointing_state:  Pointing scheduler state, shape (n, 2).
                         Col 0 = total grid points, col 1 = current index.
    """
    def __init__(self, n: int = 0, **kwargs):
        defaults = {
            'apertureArea': np.zeros(n, dtype=float),
            'pixelOmega': np.zeros(n, dtype=float),
            'qe': np.zeros(n, dtype=float),
            'photoEff': np.zeros(n, dtype=float),
            'pixCount': np.zeros(n, dtype=float),
            'solarEx': np.zeros(n, dtype=float),
            'lunarEx': np.zeros(n, dtype=float),
            'earthEx': np.zeros(n, dtype=float),
            'skyBack': np.zeros(n, dtype=float),
            'zpCal': np.zeros(n, dtype=float),
            'integrationTime': np.zeros(n, dtype=float),
            'fov': np.zeros(n, dtype=float),
            'ifov': np.zeros(n, dtype=float),
            'filt': [""] * n,
            'pointing': np.zeros((n, 3), dtype=float),
            'pointing_state': np.zeros((n, 2), dtype=int),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


def _set_detector_field(sim_data: Any, field: str, value: float) -> None:
    if hasattr(sim_data, 'iter_detector_components'):
        comps = list(sim_data.iter_detector_components())
        if comps:
            for _, _, det, _ in comps:
                arr = getattr(det, field)
                setattr(det, field, np.full(len(arr), value))
            return
    det = sim_data.detector if hasattr(sim_data, 'detector') else sim_data['detector']
    setattr(det, field, np.full(len(det.fov), value))


def setDetectorFOV(sim_data: Any, fovSize: float) -> None:
    """
    Sets the field-of-view of all detectors in the simulation to a single size.

    This is an ad-hoc helper function mainly used for testing, not regular operation.

    Args:
        sim_data: The main simulation data structure.
        fovSize: The new field-of-view diameter in radians.
    """
    _set_detector_field(sim_data, 'fov', fovSize)


def setDetectorIntegrationTime(sim_data: Any, itime: float) -> None:
    """
    Sets the integration time of all detectors in the simulation to a single value.

    This is an ad-hoc helper function mainly used for testing, not regular operation.

    Args:
        sim_data: The main simulation data structure.
        itime: The new integration time in seconds.
    """
    _set_detector_field(sim_data, 'integrationTime', itime)


##########################################################

def makeBlankDetector(n: int) -> DetectorArray:
    """
    makeBlankDetector makes and returns a DetectorArray
    with parameters initialized to zero arrays of length n.
    """
    return DetectorArray(n=n)

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
        fov:      Full field-of-view diameter in radians.
        ifov:     Pixel field-of-view diameter in radians.
        aper:     Aperture diameter - assumed to be a disk - in meters.
                  The actual value stored and used is the apertureArea.
        intTime:  Integration time in seconds.  Defaults to 1.0.
        qe:       Quantum efficiency of the system from entrance aperture
                  to detector.
        photfrac: Fraction of light captured in the photometry aperture.
        solarex:  Solar exclusion angle in radians.
        lunarex:  Lunar exclusion angle in radians.
        earthex:  Earth limb (or horizon elevation limit) exclusion angle in radians.

    Creates a detector for any platform (space or ground).
    '''
    detect = DetectorArray(n=n)
    detect.apertureArea[:] = math.pi * (aper / 2) ** 2
    detect.pixelOmega[:] = ifov ** 2
    detect.qe[:] = qe
    detect.photoEff[:] = photfrac
    detect.pixCount[:] = (fov / ifov) ** 2
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
    beta = d.skyBack
    omega = d.pixelOmega
    alpha = amag(limitingMag) * d.zpCal
    A = d.apertureArea
    eta = d.qe
    f = d.photoEff
    if debug == 1:
        print(f"gamma {gamma:.2e}")
        print("beta is,", beta)
        print("omega", omega)
        print("alpha is", alpha)
        print('A', A)
        print('eta', eta)
        print('f', f)
    t = gamma ** 2 * beta * omega / (alpha ** 2 * A * eta * (f ** 2))
    return t


def appendDetector(cd: DetectorArray, new_cd: DetectorArray) -> None:
    """
    Appends the attributes of new_cd to the existing detector object cd
    in-place.
    """
    cd.append(new_cd)
