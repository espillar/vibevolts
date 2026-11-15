from typing import Any
import numpy as np

from constants import (
    ARCSEC,
    GEO_RADIUS,
    ORBITAL_A_IDX,
    ORBITAL_RAAN_IDX,
    POINTING_COUNT_IDX,
    POINTING_PLACE_IDX,
)
from minimalsimulation import SatellitesState
from pointing import generate_pointing_sphere
from propagation import propagate_satellites
from detector import makeDetector


def _add_geo_constellation_core(sim_data: Any, n: int, fov: float, detect: Any) -> None:
    """
    Shared core logic for creating GEO constellations.

    Initializes pointing spheres, creates Keplerian orbital elements for N equally
    spaced equatorial satellites, sets up their detector pointing arrays, and
    propagates the constellation to the current simulation time.

    Args:
        sim_data: The main simulation data structure (SimulationState).
        n: The number of satellites to create in the constellation.
        fov: The field of view diameter in radians.
        detect: The DetectorArray object configured for these satellites.
    """
    theta = fov / 2
    solid_angle = 2 * np.pi * (1 - np.cos(theta))
    grid_points = int(4 * np.pi / solid_angle * 1.25)

    generate_pointing_sphere(sim_data, grid_points)

    orbital_elements = np.zeros((n, 6), dtype=float)
    orbital_elements[:, ORBITAL_A_IDX] = GEO_RADIUS
    orbital_elements[:, ORBITAL_RAAN_IDX] = np.arange(n) * 2 * np.pi / n

    epochs_list = [sim_data.get('time', sim_data.get('start_time'))] * n

    pointing_state = np.zeros((n, 2), dtype=int)
    pointing_state[:, POINTING_COUNT_IDX] = grid_points
    pointing_state[:, POINTING_PLACE_IDX] = np.random.randint(0, grid_points, size=n)

    detect.pointing = np.zeros((n, 3), dtype=float)
    detect.pointing_state = pointing_state

    new_sats = SatellitesState(
        position=np.zeros((n, 3), dtype=float),
        velocity=np.zeros((n, 3), dtype=float),
        orbital_elements=orbital_elements,
        epochs=epochs_list,
        detector=detect,
    )

    if not sim_data.satellites:
        sim_data.satellites = new_sats
    else:
        sim_data.satellites.append(new_sats)

    propagate_satellites(
        sim_data,
        sim_data.time if sim_data.time is not None else sim_data.start_time,
    )


def geos(sim_data, n, fov) -> None:
    """
    Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

    Args:
        sim_data: The main simulation data dictionary.
        n: The number of satellites to create.
        fov: The diameter of the field of view of the satellite in radians.
    """
    new_detector = makeDetector(
        n=n,
        band="V",
        fov=fov,
        ifov=3.0 * ARCSEC,
        aper=1.0,
    )
    _add_geo_constellation_core(sim_data, n, fov, new_detector)


def geosmod(sim_data, n, band, fov, ifov, aper,
            limitingmag, snr=7.0) -> None:
    """
    Creates n equally spaced satellites in GEO and adds
    them to the 'satellites' group in the simulation.

    The integration time for each detector is computed from
    limitingmag and snr using requiredIntegrationTime().

    Args:
        sim_data: The main simulation data dictionary.
        n: The number of satellites to create.
        band: The band the measurement takes place in.
        fov: The field of view diameter in radians.
        ifov: The pixel fov in radians.
        aper: The aperture diameter in meters.
        limitingmag: The limiting magnitude. Used with snr
            to compute the required integration time.
        snr: The target signal-to-noise ratio at the
            limiting magnitude. Defaults to 7.0.
    """
    from detector import requiredIntegrationTime
    detect = makeDetector(n, band, fov, ifov, aper)
    itime = requiredIntegrationTime(limitingmag, snr, detect)
    detect.integrationTime[:] = itime
    _add_geo_constellation_core(sim_data, n, fov, detect)
