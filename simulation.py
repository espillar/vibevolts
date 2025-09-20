import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any
from generate_log_spherical_points import generate_log_spherical_points
from constants import *

def create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]:
    """
    Initializes a minimal, empty data structure for a space simulation.

    Args:
        start_time: The starting time and date of the simulation. This must be a
                    timezone-aware datetime object set to UTC.
        delta_time: The time step for the simulation in seconds.

    Returns:
        A dictionary representing the basic simulation state.
    """
    if not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime object.")
    if start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware. Please set tzinfo.")

    simulation_data: Dict[str, Any] = {
        'start_time': start_time,
        'delta_time': delta_time,
        'counts': {},
        'pointing_spheres': {},
    }
    return simulation_data

def add_celestial_bodies(sim_data: Dict[str, Any]) -> None:
    """
    Adds celestial body structures (for Sun and Moon) to the simulation data.

    Args:
        sim_data: The simulation data dictionary.
    """
    sim_data['counts']['celestial'] = 2  # Sun and Moon
    sim_data['celestial'] = {
        'position': np.zeros((2, 3), dtype=float),
        'velocity': np.zeros((2, 3), dtype=float),
        'acceleration': np.zeros((2, 3), dtype=float),
    }

def add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100) -> None:
    """
    Adds a structure for fixed reference points in the GCRS frame.

    Args:
        sim_data: The simulation data dictionary.
        num_points: The number of fixed points to generate.
    """
    sim_data['counts']['fixedpoints'] = num_points
    sim_data['fixedpoints'] = {
        'position': generate_log_spherical_points(
            num_points=num_points,
            inner_radius=2000000,
            outer_radius=84328000
        )[0],
        'visibility': np.zeros((num_points, 0), dtype=int) # Visibility will be resized later
    }