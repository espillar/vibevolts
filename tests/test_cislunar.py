import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from cislunar import (
    CISLUNAR_ORBITS,
    MU_EARTH_MOON,
    SAMPLE_L1_HALO_STATE,
    T_STAR,
    add_cislunar_satellites,
    emb_rotating_to_gcrs,
    gcrs_to_emb_rotating,
    get_moon_gcrs_state,
    jacobi_constant,
    propagate_cr3bp,
    propagate_cislunar,
)
from minimalsimulation import create_empty_simulation


def test_emb_gcrs_round_trip():
    t = datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    moon_pos, moon_vel = get_moon_gcrs_state(t)

    earth_rot = np.array([-MU_EARTH_MOON, 0.0, 0.0])
    moon_rot = np.array([1.0 - MU_EARTH_MOON, 0.0, 0.0])

    earth_gcrs, _ = emb_rotating_to_gcrs(earth_rot, moon_pos, moon_vel)
    moon_gcrs, _ = emb_rotating_to_gcrs(moon_rot, moon_pos, moon_vel)

    assert np.allclose(earth_gcrs, np.zeros(3), atol=1e-6)
    assert np.allclose(moon_gcrs, moon_pos, atol=1e-6)

    sample_rot = np.array([0.85, 0.12, -0.04])
    sample_gcrs, _ = emb_rotating_to_gcrs(sample_rot, moon_pos, moon_vel)
    round_trip = gcrs_to_emb_rotating(sample_gcrs, moon_pos, moon_vel)
    assert np.allclose(round_trip, sample_rot, atol=1e-12)


@pytest.mark.parametrize("orbit_name", list(CISLUNAR_ORBITS.keys()))
def test_cr3bp_jacobi_conservation(orbit_name):
    orbit = CISLUNAR_ORBITS[orbit_name]
    s0 = orbit['state']
    c0 = jacobi_constant(s0)
    final_state = propagate_cr3bp(s0, orbit['period_nd'] * T_STAR)
    c1 = jacobi_constant(final_state)
    assert abs(c1 - c0) < 1e-10
    assert np.allclose(final_state, s0, atol=1e-6)


def test_add_and_propagate_cislunar():
    t0 = datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(t0)
    add_cislunar_satellites(sim_data, SAMPLE_L1_HALO_STATE, epoch=t0)

    assert sim_data.counts.cislunar == 1
    pos0 = sim_data.cislunar.position[0].copy()
    assert np.linalg.norm(pos0) > 2.5e8

    t1 = t0 + timedelta(days=3)
    propagate_cislunar(sim_data, t1)
    pos1 = sim_data.cislunar.position[0]
    assert not np.allclose(pos0, pos1)


def test_mixed_tle_and_cislunar_cadence():
    from cadenceController import initCadence, nextIntegration
    from celestialbodies import add_celestial_bodies, celestial_update
    from constants import ARCSEC, DEGREE
    from detector import makeDetector
    from propagation import add_satellites_from_tle
    from targets import add_fixed_points

    t0 = datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(t0)

    tle = """GEO-1
1 90001U 25001A   25209.50000000  .00000000  00000-0  00000-0 0  9991
2 90001   0.0500   0.0000 0001000   0.0000   0.0000  1.00270000    12"""
    add_satellites_from_tle(sim_data, tle, 'satellites')
    sim_data.satellites.detector = makeDetector(
        1, band='V', fov=10 * DEGREE, ifov=3 * ARCSEC, aper=1.0, intTime=60.0
    )

    add_cislunar_satellites(sim_data, SAMPLE_L1_HALO_STATE, epoch=t0)
    sim_data.cislunar.detector = makeDetector(
        1, band='V', fov=10 * DEGREE, ifov=3 * ARCSEC, aper=1.0, intTime=60.0
    )

    add_celestial_bodies(sim_data)
    celestial_update(sim_data, t0)
    add_fixed_points(sim_data, 10)
    initCadence(sim_data)

    assert sim_data.get_detector_positions().shape == (2, 3)
    assert list(sim_data.get_detector_categories()) == ['satellites', 'cislunar']

    sat_pos0 = sim_data.satellites.position[0].copy()
    cis_pos0 = sim_data.cislunar.position[0].copy()

    nextIntegration(sim_data)
    nextIntegration(sim_data)

    assert not np.allclose(sim_data.satellites.position[0], sat_pos0)
    assert not np.allclose(sim_data.cislunar.position[0], cis_pos0)
