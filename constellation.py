import numpy as np
from datetime import datetime
from typing import Dict, Any
import math

from constants import (
    ORBITAL_A_IDX, ORBITAL_E_IDX, ORBITAL_I_IDX,
    ORBITAL_RAAN_IDX, ORBITAL_ARGP_IDX, ORBITAL_M_IDX,
    POINTING_COUNT_IDX, POINTING_PLACE_IDX
)
from pointing import generate_pointing_sphere

def geos(sim_data: Dict[str, Any], n: int, constellation: str, fov: float) -> None:
    """
    Creates n equally spaced satellites in GEO and adds them to the simulation.

    Args:
        sim_data: The main simulation data dictionary.
        n: The number of satellites to create.
        constellation: The name of the constellation.
        fov: The field of view of the satellite in radians.
    """
    # Calculate solid angle
    theta = fov / 2
    solid_angle = 2 * np.pi * (1 - np.cos(theta))

    # Calculate grid_points
    grid_points = int(4 * np.pi / solid_angle * 1.25)

    # Generate and store the pointing sphere
    generate_pointing_sphere(sim_data, grid_points)

    orbital_elements_list = []
    epochs_list = []
    pointing_state_list = []

    # Geostationary orbit semi-major axis in meters
    a = 42164000.0

    for i in range(n):
        elements = np.zeros(6)
        elements[ORBITAL_A_IDX] = a
        elements[ORBITAL_E_IDX] = 0.0
        elements[ORBITAL_I_IDX] = 0.0
        elements[ORBITAL_RAAN_IDX] = i * 2 * np.pi / n
        elements[ORBITAL_ARGP_IDX] = 0.0
        elements[ORBITAL_M_IDX] = 0.0
        orbital_elements_list.append(elements)
        epochs_list.append(sim_data['start_time'])

        pointing_state = np.zeros(2, dtype=int)
        pointing_state[POINTING_COUNT_IDX] = grid_points
        pointing_state[POINTING_PLACE_IDX] = int(grid_points * i / n)
        pointing_state_list.append(pointing_state)


    orbital_elements = np.array(orbital_elements_list, dtype=float)
    pointing_state_array = np.array(pointing_state_list, dtype=int)

    if constellation not in sim_data:
        sim_data[constellation] = {}

    sim_data['counts'][constellation] = n
    sim_data[constellation] = {
        'position': np.zeros((n, 3), dtype=float),
        'velocity': np.zeros((n, 3), dtype=float),
        'acceleration': np.zeros((n, 3), dtype=float),
        'orbital_elements': orbital_elements,
        'epochs': epochs_list,
        'pointing': np.zeros((n, 3), dtype=float),
        'pointing_state': pointing_state_array,
        'detector': np.zeros((n, 7), dtype=float),
    }