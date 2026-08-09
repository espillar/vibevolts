import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple
from sgp4.api import Satrec
from astropy.time import Time
from astropy.coordinates import get_body, GCRS
import astropy.units as u

from constants import *



def add_satellites_from_tle(sim_data: Any, tle_file_path: str, sat_category: str) -> None:
    """
    Adds and initializes a category of satellites from a TLE file.
    Note although the TLEs are loaded, positions etc. are not.

    Args:
        sim_data: The main simulation data structure.
        tle_file_path: Path to the TLE file.
        sat_category: The key for this satellite category (e.g., 'satellites').

    Data added to the sat_category element of sim_data:
        position, velocity, acceleration, orbital_elements, epochs
    """
    from detector import makeBlankDetector, appendDetector
    orbital_elements, epochs = readtle(tle_file_path)
    num_sats = len(epochs)

    new_detector = makeBlankDetector(num_sats)
    new_detector.category = [sat_category] * num_sats
    new_detector.asset_index = np.arange(num_sats, dtype=int)
    if 'detector' not in sim_data or not sim_data.detector:
        sim_data.detector = new_detector
    else:
        appendDetector(sim_data.detector, new_detector)
    sim_data[sat_category] = {
        'position': np.zeros((num_sats, 3), dtype=float),
        'velocity': np.zeros((num_sats, 3), dtype=float),
        'acceleration': np.zeros((num_sats, 3), dtype=float),
        'orbital_elements': orbital_elements,
        'epochs': epochs,
    }

def readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]:
    """
    Reads a TLE file and extracts orbital elements and epochs for each satellite.

    The array returned has the orbital elements in "canonical" order.
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

def propagate_satellites(data_struct: Any, time_date: datetime, sat_category: str = None) -> Any:
    """
    Propagates satellite positions and velocities to a given time
    using Keplerian two-body orbital mechanics.

    Args:
        data_struct: The main simulation data structure.
        time_date: The datetime to propagate the satellites to.
            NOTE: the datetime in data_struct is NOT updated by
            this function.
        sat_category: Optional satellite category key. If None,
            all satellite categories are auto-discovered from
            data_struct.counts.

    Returns:
        The updated data_struct with new positions and velocities.
    """
    MU_EARTH = 3.986004418e14
    time_date_timestamp = time_date.timestamp()

    if sat_category:
        categories = [sat_category]
    elif hasattr(data_struct, 'counts') and data_struct.counts:
        non_sat_keys = {'celestial', 'observatories', 'fixedpoints'}
        categories = [
            k for k, v in data_struct.counts.items()
            if k not in non_sat_keys and v > 0
        ]
        if not categories:
            categories = ['satellites']
    else:
        categories = ['satellites']

    for category in categories:
        if category not in data_struct.counts or data_struct.counts[category] == 0:
            continue

        elements = data_struct[category].orbital_elements
        epochs = data_struct[category].epochs

        epoch_timestamps = np.array([e.timestamp() for e in epochs])
        delta_t_array = time_date_timestamp - epoch_timestamps

        a = elements[:, ORBITAL_A_IDX]
        e = elements[:, ORBITAL_E_IDX]
        i = elements[:, ORBITAL_I_IDX]
        raan = elements[:, ORBITAL_RAAN_IDX]
        argp = elements[:, ORBITAL_ARGP_IDX]
        M0 = elements[:, ORBITAL_M_IDX]

        # Guard against uninitialized/invalid orbits (a <= 0 or e >= 1.0)
        safe_a = np.where(a <= 0, EARTH_RADIUS + 400000.0, a)
        safe_e = np.clip(e, 0.0, 0.999999)

        n = np.sqrt(MU_EARTH / safe_a**3)
        M = (M0 + n * delta_t_array) % (2 * np.pi)

        E = M.copy()
        for _ in range(10):
            f_E = E - safe_e * np.sin(E) - M
            f_prime_E = 1 - safe_e * np.cos(E)
            f_prime_E[f_prime_E == 0] = 1e-10
            E = E - f_E / f_prime_E

        tan_nu_half = np.sqrt((1 + safe_e) / (1 - safe_e)) * np.tan(E / 2)
        nu = 2 * np.arctan(tan_nu_half)

        r = safe_a * (1 - safe_e * np.cos(E))

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

        # In-place assignment into existing position array buffer (zero allocations)
        pos = data_struct[category].position
        if pos.shape != (len(x_gcrs), 3):
            pos = np.empty((len(x_gcrs), 3), dtype=float)
            data_struct[category].position = pos
        pos[:, 0] = x_gcrs
        pos[:, 1] = y_gcrs
        pos[:, 2] = z_gcrs

        # Velocity in PQW (perifocal) frame
        p = safe_a * (1 - safe_e**2)
        v_factor = np.sqrt(MU_EARTH / p)
        vx_pqw = -v_factor * np.sin(nu)
        vy_pqw = v_factor * (safe_e + np.cos(nu))

        # Transform velocity to GCRS using same P, Q basis
        vx_gcrs = vx_pqw * P_x + vy_pqw * Q_x
        vy_gcrs = vx_pqw * P_y + vy_pqw * Q_y
        vz_gcrs = vx_pqw * P_z + vy_pqw * Q_z

        # In-place assignment into existing velocity array buffer (zero allocations)
        vel = data_struct[category].velocity
        if vel.shape != (len(vx_gcrs), 3):
            vel = np.empty((len(vx_gcrs), 3), dtype=float)
            data_struct[category].velocity = vel
        vel[:, 0] = vx_gcrs
        vel[:, 1] = vy_gcrs
        vel[:, 2] = vz_gcrs

    return data_struct
