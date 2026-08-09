import numpy as np
from datetime import datetime
from typing import Dict, Any
import math
import random
import sys
import radiometry_data
from radiometry_data import FILTER_DATA
from constants import *
from minimalsimulation import *
from pointing import generate_pointing_sphere
from propagation import propagate_satellites # Added import
from detector import makeBlankDetector, makeDetector


#########################################################

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
    # Calculate solid angle 
    theta = fov / 2
    solid_angle = 2 * np.pi * (1 - np.cos(theta))
    
    # Calculate grid_points - blow things up by 0.25 for overlap
    grid_points = int(4 * np.pi / solid_angle * 1.25)

    # Generate and store the pointing sphere
    generate_pointing_sphere(sim_data, grid_points)

    orbital_elements_list = []
    epochs_list = []
    pointing_state_list = []

    # Geostationary orbit semi-major axis in meters
    a = GEO_RADIUS

    # Create a set of elements evenly spaced around the equator
    for i in range(n):
        elements = np.zeros(6)
        elements[ORBITAL_A_IDX] = a
        elements[ORBITAL_E_IDX] = 0.0
        elements[ORBITAL_I_IDX] = 0.0
        elements[ORBITAL_RAAN_IDX] = i * 2 * np.pi / n
        elements[ORBITAL_ARGP_IDX] = 0.0
        elements[ORBITAL_M_IDX] = 0.0
        orbital_elements_list.append(elements)
        epochs_list.append(sim_data.get('time', sim_data.get('start_time')))
   
        pointing_state = np.zeros(2, dtype=int)
        pointing_state[POINTING_COUNT_IDX] = grid_points
        pointing_state[POINTING_PLACE_IDX] = random.randint(0, grid_points - 1)
        pointing_state_list.append(pointing_state)

    orbital_elements = np.array(orbital_elements_list, dtype=float)
    pointing_state_array = np.array(pointing_state_list, dtype=int).T

    from detector import appendDetector
    
    detect.pointing = np.zeros((n, 3), dtype=float)
    detect.pointing_state = pointing_state_array
    
    current_count = sim_data.counts.get('satellites', 0)
    detect.category = ['satellites'] * n
    detect.asset_index = np.arange(current_count, current_count + n, dtype=int)

    if not sim_data.detector:
        sim_data.detector = detect
    else:
        appendDetector(sim_data.detector, detect)

    if not sim_data.satellites:
        sim_data.satellites = SatellitesState(
            position=np.zeros((n, 3), dtype=float),
            velocity=np.zeros((n, 3), dtype=float),
            acceleration=np.zeros((n, 3), dtype=float),
            orbital_elements=orbital_elements,
            epochs=epochs_list,
        )
    else:
        sat = sim_data.satellites
        sat.position = np.vstack([sat.position, np.zeros((n, 3), dtype=float)])
        sat.velocity = np.vstack([sat.velocity, np.zeros((n, 3), dtype=float)])
        sat.acceleration = np.vstack([sat.acceleration, np.zeros((n, 3), dtype=float)])
        sat.orbital_elements = np.vstack([sat.orbital_elements, orbital_elements])
        sat.epochs.extend(epochs_list)
    
    sim_data = propagate_satellites(
        sim_data,
        sim_data.time if sim_data.time is not None else sim_data.start_time
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
        aper=1.0
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

