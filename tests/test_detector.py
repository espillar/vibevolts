import numpy as np
import pytest
from detector import makeDetector, DetectorArray
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

    assert isinstance(detector, DetectorArray)
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

    assert isinstance(detector, DetectorArray)
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


def test_setDetectorIntegrationTime():
    """
    Test that setDetectorIntegrationTime correctly updates the integrationTime attribute.
    """
    from detector import setDetectorIntegrationTime
    sim_data = {
        'detector': makeDetector(2, "V", 1 * DEGREE, 2 * ARCSEC, 1.0)
    }

    assert np.allclose(sim_data['detector'].integrationTime, 1.0)

    setDetectorIntegrationTime(sim_data, 5.0)

    assert np.allclose(sim_data['detector'].integrationTime, 5.0)
    # Check that it didn't create a spurious 'itime' attribute
    assert not hasattr(sim_data['detector'], 'itime')


def test_appendDetector():
    """
    Test that appendDetector correctly appends attributes from one detector to another.
    """
    from detector import appendDetector

    d1 = makeDetector(1, "V", 1 * DEGREE, 2 * ARCSEC, 1.0)
    d2 = makeDetector(2, "R", 2 * DEGREE, 4 * ARCSEC, 2.0)

    assert len(d1.filt) == 1
    assert len(d2.filt) == 2

    appendDetector(d1, d2)

    assert len(d1.filt) == 3
    assert d1.filt == ["V", "R", "R"]
    assert np.allclose(d1.apertureArea, [np.pi * 0.5**2, np.pi, np.pi])


def test_requiredIntegrationTime():
    """
    Test that requiredIntegrationTime calculates integration time correctly
    and handles the squared photoEff (photfrac) term.
    """
    from detector import requiredIntegrationTime
    # Make a detector with photfrac = 0.5
    d = makeDetector(1, "V", 1 * DEGREE, 2 * ARCSEC, 1.0, qe=0.8, photfrac=0.5)
    
    t = requiredIntegrationTime(20.0, 10.0, d)
    
    gamma = 10.0
    beta = d.skyBack[0]
    omega = d.pixelOmega[0]
    from radiometry_calcs import amag
    alpha = amag(20.0) * d.zpCal[0]
    A = d.apertureArea[0]
    eta = d.qe[0]
    f = d.photoEff[0]
    
    expected_t = (gamma**2 * beta * omega) / (alpha**2 * A * eta * (f**2))
    assert np.allclose(t, expected_t)

