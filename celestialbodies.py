import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import get_body, GCRS
import astropy.units as u

def add_celestial_bodies(sim_data: Any) -> None:
    """
    Adds celestial body structures (for Sun and Moon) to the simulation data.

    Args:
        sim_data: The simulation data structure.
    """
    sim_data.celestial = {
        'position': np.zeros((2, 3), dtype=float),
        'velocity': np.zeros((2, 3), dtype=float),
        'acceleration': np.zeros((2, 3), dtype=float),
    }

def celestial_update(data_struct: Any, time_date: Optional[datetime] = None) -> Any:
    """
    Calculates and updates the positions of the Sun and Moon.
    """
    if time_date is None:
        time_date = data_struct.time

    if time_date.tzinfo is None:
        raise ValueError("time_date must be timezone-aware.")

    astro_time = Time(time_date)
    sun_coords = get_body("sun", astro_time)
    sun_gcrs = sun_coords.transform_to(GCRS(obstime=astro_time))
    moon_coords = get_body("moon", astro_time)
    moon_gcrs = moon_coords.transform_to(GCRS(obstime=astro_time))

    celestial_pos = data_struct.celestial.position
    celestial_pos[0] = sun_gcrs.cartesian.xyz.to(u.m).value
    celestial_pos[1] = moon_gcrs.cartesian.xyz.to(u.m).value

    return data_struct
