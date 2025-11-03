import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from sgp4.api import Satrec
from astropy.time import Time
from astropy.coordinates import get_body, GCRS
import astropy.units as u

from constants import (
    ORBITAL_A_IDX, ORBITAL_E_IDX, ORBITAL_I_IDX,
    ORBITAL_RAAN_IDX, ORBITAL_ARGP_IDX, ORBITAL_M_IDX,
    DETECTOR_ARRAY_SIZE
)

def add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None:
    """
    Adds and initializes a category of satellites from a TLE file.
    Mote although the TLEs are loaded, positions etc. are not.

    Args:
        sim_data: The main simulation data dictionary.
        tle_file_path: Path to the TLE file.
        sat_category: The key for this satellite category (e.g., 'satellites').

    Data added to the set_category element of sim_data
    position
    velocity
    acceleration
    orbital_elements
    epochs
    pointing
    """
    orbital_elements, epochs = readtle(tle_file_path)
    num_sats = len(epochs)

    sim_data['counts'][sat_category] = num_sats
    sim_data[sat_category] = {
        'position': np.zeros((num_sats, 3), dtype=float),
        'velocity': np.zeros((num_sats, 3), dtype=float),
        'acceleration': np.zeros((num_sats, 3), dtype=float),
        'orbital_elements': orbital_elements,
        'epochs': epochs,
        'pointing': np.zeros((num_sats, 3), dtype=float),
        'pointing_state': np.zeros((num_sats, 2), dtype=int),
        'detector': np.zeros((num_sats, DETECTOR_ARRAY_SIZE), dtype=object),
    }

def celestial_update(data_struct: Dict[str, Any], time_date: datetime) -> Dict[str, Any]:
    """
    Calculates and updates the positions of the Sun and Moon.
    """
    if time_date.tzinfo is None:
        raise ValueError("time_date must be timezone-aware.")

    astro_time = Time(time_date)
    sun_coords = get_body("sun", astro_time)
    sun_gcrs = sun_coords.transform_to(GCRS(obstime=astro_time))
    moon_coords = get_body("moon", astro_time)
    moon_gcrs = moon_coords.transform_to(GCRS(obstime=astro_time))

    celestial_pos = data_struct['celestial']['position']
    celestial_pos[0] = sun_gcrs.cartesian.xyz.to(u.m).value
    celestial_pos[1] = moon_gcrs.cartesian.xyz.to(u.m).value

    return data_struct

def readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]:
    """
    Reads a TLE file and extracts orbital elements and epochs for each satellite.

    The array returned ahd the orbital elements in "canonical" order.
    """
    orbital_elements_list = []
    epochs_list = []
    with open(tle_file_path, 'r') as f:
        lines = f.readlines()

    for i in range(0, len(lines), 3):
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()

        satellite = Satrec.twoline2rv(line1, line2)

        jd, fr = satellite.jdsatepoch, satellite.jdsatepochF
        epoch_dt = Time(jd, fr, format='jd', scale='utc').to_datetime(timezone.utc)
        epochs_list.append(epoch_dt)

        a = satellite.a * satellite.radiusearthkm * 1000.0
        e = satellite.ecco
        inc = satellite.inclo
        raan = satellite.nodeo
        argp = satellite.argpo
        M = satellite.mo

        elements = np.zeros(6)
        elements[ORBITAL_A_IDX] = a
        elements[ORBITAL_E_IDX] = e
        elements[ORBITAL_I_IDX] = inc
        elements[ORBITAL_RAAN_IDX] = raan
        elements[ORBITAL_ARGP_IDX] = argp
        elements[ORBITAL_M_IDX] = M
        orbital_elements_list.append(elements)

    return np.array(orbital_elements_list, dtype=float), epochs_list

def propagate_satellites_new(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]:
    """
    Updates satellite positions and pointing vectors based on their orbital elements.
    """
    MU_EARTH = 3.986004418e14
    time_date_timestamp = time_date.timestamp()

    if sat_category:
        categories = [sat_category]
    else:
        categories = ['satellites', 'red_satellites']

    for category in categories:
        if category not in data_struct['counts'] or data_struct['counts'][category] == 0:
            continue

        elements = data_struct[category]['orbital_elements']
        epochs = data_struct[category]['epochs']

        epoch_timestamps = np.array([e.timestamp() for e in epochs])
        delta_t_array = time_date_timestamp - epoch_timestamps

        a = elements[:, ORBITAL_A_IDX]
        e = elements[:, ORBITAL_E_IDX]
        i = elements[:, ORBITAL_I_IDX]
        raan = elements[:, ORBITAL_RAAN_IDX]
        argp = elements[:, ORBITAL_ARGP_IDX]
        M0 = elements[:, ORBITAL_M_IDX]

        n = np.sqrt(MU_EARTH / a**3)
        M = (M0 + n * delta_t_array) % (2 * np.pi)

        E = M.copy()
        for _ in range(10):
            f_E = E - e * np.sin(E) - M
            f_prime_E = 1 - e * np.cos(E)
            f_prime_E[f_prime_E == 0] = 1e-10
            E = E - f_E / f_prime_E

        tan_nu_half = np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)
        nu = 2 * np.arctan(tan_nu_half)

        r = a * (1 - e * np.cos(E))

        x_pqw = r * np.cos(nu)
        y_pqw = r * np.sin(nu)

        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)
        cos_argp = np.cos(argp)
        sin_argp = np.sin(argp)
        cos_i = np.cos(i)
        sin_i = np.sin(i)

        P_x = cos_argp * cos_raan - sin_argp * sin_raan * cos_i
        P_y = cos_argp * sin_raan + sin_argp * cos_raan * cos_i
        P_z = sin_argp * sin_i

        Q_x = -sin_argp * cos_raan - cos_argp * sin_raan * cos_i
        Q_y = -sin_argp * sin_raan + cos_argp * cos_raan * cos_i
        Q_z = cos_argp * sin_i

        x_gcrs = x_pqw * P_x + y_pqw * Q_x
        y_gcrs = x_pqw * P_y + y_pqw * Q_y
        z_gcrs = x_pqw * P_z + y_pqw * Q_z

        positions = np.vstack((x_gcrs, y_gcrs, z_gcrs)).T
        data_struct[category]['position'] = positions

        norms = np.linalg.norm(positions, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1.0
        data_struct[category]['pointing'] = positions / norms

    return data_struct
