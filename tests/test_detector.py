import numpy as np
import pytest
from types import SimpleNamespace
from detector import makeDetector
from constants import DEGREE, ARCSEC

def test_makeDetector_intTime_default():
    """
    Test that makeDetector sets integrationTime to 1.0 by default.
    """
    n_detectors = 1
    band = "V"
    fov = 1 * DEGREE
    ifov = 2 * ARCSEC
    aper = 1.0 # 1 meter aperture

    detector = makeDetector(n_detectors, band, fov, ifov, aper)

    assert isinstance(detector, SimpleNamespace)
    assert np.allclose(detector.integrationTime, 1.0)
    assert detector.integrationTime.shape == (n_detectors,)

def test_makeDetector_intTime_custom_value():
    """
    Test that makeDetector sets integrationTime to a custom value when provided.
    """
    n_detectors = 1
    band = "R"
    fov = 0.5 * DEGREE
    ifov = 1 * ARCSEC
    aper = 0.5 # 0.5 meter aperture
    custom_int_time = 5.0

    detector = makeDetector(n_detectors, band, fov, ifov, aper, intTime=custom_int_time)

    assert isinstance(detector, SimpleNamespace)
    assert np.allclose(detector.integrationTime, custom_int_time)
    assert detector.integrationTime.shape == (n_detectors,)

def test_makeDetector_multiple_detectors():
    """
    Test that makeDetector correctly initializes integrationTime for multiple detectors.
    """
    n_detectors = 3
    band = "I"
    fov = 2 * DEGREE
    ifov = 4 * ARCSEC
    aper = 2.0
    custom_int_time = 10.0

    detector_default = makeDetector(n_detectors, band, fov, ifov, aper)
    assert np.allclose(detector_default.integrationTime, 1.0)
    assert detector_default.integrationTime.shape == (n_detectors,)

    detector_custom = makeDetector(n_detectors, band, fov, ifov, aper, intTime=custom_int_time)
    assert np.allclose(detector_custom.integrationTime, custom_int_time)
    assert detector_custom.integrationTime.shape == (n_detectors,)
