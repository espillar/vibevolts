import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from sgp4.api import Satrec
from astropy.time import Time
from astropy.coordinates import get_body, GCRS
import astropy.units as u

from simulation import (
    ORBITAL_A_IDX, ORBITAL_E_IDX, ORBITAL_I_IDX,
    ORBITAL_RAAN_IDX, ORBITAL_ARGP_IDX, ORBITAL_M_IDX
)

def celestial_update(data_struct: Dict[str, Any], time_date: datetime) -> Dict[str, Any]:
    """
    Calculates and updates the positions of the Sun and Moon.

    This function uses the astropy library to get the precise GCRS coordinates
    of the Sun and Moon for the given time, and updates the 'celestial' position
    component in the main data structure. Velocities and accelerations are not
    calculated and remain zero.

    Args:
        data_struct: The main simulation data dictionary from initializeStructures.
        time_date: The timezone-aware datetime object (in UTC) for the calculation.

    Returns:
        The modified data_struct with updated celestial positions.
    """
    if time_date.tzinfo is None:
        raise ValueError("time_date must be timezone-aware.")

    # Convert the python datetime object to an astropy Time object
    astro_time = Time(time_date)

    # Get Sun position in the GCRS frame using the modern get_body function
    sun_coords = get_body("sun", astro_time)
    sun_gcrs = sun_coords.transform_to(GCRS(obstime=astro_time))

    # Get Moon position in the GCRS frame using the modern get_body function
    moon_coords = get_body("moon", astro_time)
    moon_gcrs = moon_coords.transform_to(GCRS(obstime=astro_time))

    # Update the celestial data arrays (index 0 for Sun, 1 for Moon)
    # Position data is converted from astropy's representation to a simple
    # numpy array in meters.
    celestial_pos = data_struct['celestial']['position']
    celestial_pos[0] = sun_gcrs.cartesian.xyz.to(u.m).value
    celestial_pos[1] = moon_gcrs.cartesian.xyz.to(u.m).value

    return data_struct

def readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]:
    """
    Reads a TLE file and extracts orbital elements and epochs for each satellite.

    Args:
        tle_file_path: The path to the TLE file.

    Returns:
        A tuple containing:
        - A NumPy array of orbital elements.
        - A list of datetime objects representing the epoch for each satellite.
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

def propagate_satellites(data_struct: Dict[str, Any], time_date: datetime) -> Dict[str, Any]:
    """
    Updates satellite positions and pointing vectors based on their orbital elements.

    This function propagates the orbits of all satellites ('satellites' and
    'red_satellites') from their TLE epoch to the specified time_date
    using Kepler's laws. After calculating the new position, it sets the
    satellite's pointing vector to be radially outward from the Earth's center.

    Args:
        data_struct: The main simulation data dictionary.
        time_date: The timezone-aware datetime object (in UTC) to propagate to.

    Returns:
        The modified data_struct with updated satellite positions and pointing vectors.
    """
    MU_EARTH = 3.986004418e14

    time_date_timestamp = time_date.timestamp()

    for sat_category in ['satellites', 'red_satellites']:
        if data_struct['counts'][sat_category] == 0:
            continue

        elements = data_struct[sat_category]['orbital_elements']
        epochs = data_struct[sat_category]['epochs']

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
        data_struct[sat_category]['position'] = positions

        norms = np.linalg.norm(positions, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1.0
        data_struct[sat_category]['pointing'] = positions / norms

    return data_struct
