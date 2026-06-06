import numpy as np
import pytest
from datetime import datetime, timezone
from minimalsimulation import create_empty_simulation
from celestialbodies import add_celestial_bodies
from constellation import geos
from constants import ARCSEC

def test_geos_detector_initialization():
    """
    Test that geos correctly initializes satellite detectors with non-zero parameters.
    """
    sim_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_time)
    add_celestial_bodies(sim_data)

    n_sats = 5
    fov_val = 0.1
    geos(sim_data, n_sats, fov_val)

    assert sim_data['counts']['satellites'] == n_sats
    assert 'detector' in sim_data
    
    detector = sim_data['detector']
    
    # Assert detector properties are initialized and non-zero
    assert np.all(detector.apertureArea > 0.0)
    assert np.all(detector.pixelOmega > 0.0)
    assert np.all(detector.qe > 0.0)
    assert np.all(detector.photoEff > 0.0)
    assert np.allclose(detector.fov, fov_val)
    assert np.allclose(detector.ifov, 3.0 * ARCSEC)
    assert detector.filt == ["V"] * n_sats
