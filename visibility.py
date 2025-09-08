import numpy as np
from typing import Dict, Any, Tuple, Optional

from constants import (
    EARTH_RADIUS, MOON_RADIUS, DETECTOR_SOLAR_EXCL_IDX,
    DETECTOR_LUNAR_EXCL_IDX, DETECTOR_EARTH_EXCL_IDX
)

def solarexclusion(data_struct: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates solar exclusion for all satellites based on their pointing vectors.

    This function operates in a vectorized manner on all satellites in the
    'satellites' category. It computes the angle between each satellite's
    pointing vector and the vector from the satellite to the Sun.

    Args:
        data_struct: The main simulation data dictionary.

    Returns:
        A tuple containing:
        - exclusion_vector (np.ndarray): An array of the same length as the
          number of satellites. An element is 1 if the satellite is within
          the solar exclusion angle, 0 otherwise.
        - angle_vector (np.ndarray): An array containing the calculated angle
          in radians for each satellite.
    """
    num_sats = data_struct['counts']['satellites']
    if num_sats == 0:
        return np.array([]), np.array([])

    sun_pos = data_struct['celestial']['position'][0]
    sat_pos = data_struct['satellites']['position']
    sat_pointing = data_struct['satellites']['pointing']
    solar_exclusion_angles = data_struct['satellites']['detector'][:, DETECTOR_SOLAR_EXCL_IDX]

    vec_sat_to_sun = sun_pos - sat_pos

    norm_sat_to_sun = np.linalg.norm(vec_sat_to_sun, axis=1)
    norm_sat_pointing = np.linalg.norm(sat_pointing, axis=1)

    valid_norms = (norm_sat_to_sun > 1e-9) & (norm_sat_pointing > 1e-9)

    angle_vector = np.full(num_sats, np.pi)

    if np.any(valid_norms):
        dot_product = np.einsum('ij,ij->i', vec_sat_to_sun[valid_norms], sat_pointing[valid_norms])
        cos_angle = dot_product / (norm_sat_to_sun[valid_norms] * norm_sat_pointing[valid_norms])
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_vector[valid_norms] = np.arccos(cos_angle)

    exclusion_vector = (angle_vector < solar_exclusion_angles).astype(int)

    return exclusion_vector, angle_vector

def exclusion(
    data_struct: Dict[str, Any],
    satellite_index: int,
    print_debug: bool = False
) -> int:
    """
    Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_index: The index of the satellite to check.
        print_debug: If True, prints detailed debug information for the calculation.

    Returns:
        0 if the satellite's view is excluded, 1 otherwise.
    """
    sat_pos = data_struct['satellites']['position'][satellite_index]
    sat_pointing = data_struct['satellites']['pointing'][satellite_index]
    sun_pos = data_struct['celestial']['position'][0]
    moon_pos = data_struct['celestial']['position'][1]

    detector_props = data_struct['satellites']['detector'][satellite_index]
    solar_excl_angle = detector_props[DETECTOR_SOLAR_EXCL_IDX]
    lunar_excl_angle = detector_props[DETECTOR_LUNAR_EXCL_IDX]
    earth_excl_angle = detector_props[DETECTOR_EARTH_EXCL_IDX]

    vec_to_sun = sun_pos - sat_pos
    vec_to_moon = moon_pos - sat_pos
    vec_to_earth = -sat_pos

    dist_to_sun = np.linalg.norm(vec_to_sun)
    dist_to_moon = np.linalg.norm(vec_to_moon)
    dist_to_earth = np.linalg.norm(vec_to_earth)

    u_vec_to_sun = vec_to_sun / dist_to_sun if dist_to_sun > 0 else np.array([0.,0.,0.])
    u_vec_to_moon = vec_to_moon / dist_to_moon if dist_to_moon > 0 else np.array([0.,0.,0.])
    u_vec_to_earth = vec_to_earth / dist_to_earth if dist_to_earth > 0 else np.array([0.,0.,0.])

    norm_pointing = np.linalg.norm(sat_pointing)
    u_sat_pointing = sat_pointing / norm_pointing if norm_pointing > 0 else np.array([0.,0.,0.])

    sun_flag, moon_flag, earth_flag = False, False, False

    cos_angle_sun = np.clip(np.dot(u_sat_pointing, u_vec_to_sun), -1.0, 1.0)
    angle_sun = np.arccos(cos_angle_sun)
    if angle_sun < solar_excl_angle:
        sun_flag = True

    cos_angle_moon = np.clip(np.dot(u_sat_pointing, u_vec_to_moon), -1.0, 1.0)
    angle_moon = np.arccos(cos_angle_moon)
    apparent_radius_moon = np.arctan(MOON_RADIUS / dist_to_moon) if dist_to_moon > 0 else 0
    if (angle_moon - apparent_radius_moon) < lunar_excl_angle:
        moon_flag = True

    cos_angle_earth = np.clip(np.dot(u_sat_pointing, u_vec_to_earth), -1.0, 1.0)
    angle_earth = np.arccos(cos_angle_earth)
    apparent_radius_earth = np.arctan(EARTH_RADIUS / dist_to_earth) if dist_to_earth > 0 else 0
    if (angle_earth - apparent_radius_earth) < earth_excl_angle:
        earth_flag = True

    if print_debug:
        print(f"--- Exclusion Debug for Satellite {satellite_index} ---")
        print(f"  - Sun Flag:   {sun_flag}, Angle: {np.rad2deg(angle_sun):.2f} deg, Excl: {np.rad2deg(solar_excl_angle):.2f} deg")
        print(f"  - Moon Flag:  {moon_flag}, Angle: {np.rad2deg(angle_moon):.2f} deg, Excl: {np.rad2deg(lunar_excl_angle):.2f} deg")
        print(f"  - Earth Flag: {earth_flag}, Angle: {np.rad2deg(angle_earth):.2f} deg, Excl: {np.rad2deg(earth_excl_angle):.2f} deg")

    is_excluded = sun_flag or moon_flag or earth_flag
    return 0 if is_excluded else 1

def update_visibility_table(
    data_struct: Dict[str, Any],
    print_debug_for_sat: Optional[int] = None
) -> None:
    """
    Updates the visibility table for each satellite against each fixed point.

    Args:
        data_struct: The main simulation data dictionary.
        print_debug_for_sat: If an integer is provided, the `exclusion` function's
                             debug printout will be enabled for that satellite index.
    """
    num_satellites = data_struct['counts']['satellites']
    fixed_points = data_struct['fixedpoints']['position']
    num_fixed_points = len(fixed_points)
    visibility_table = data_struct['fixedpoints']['visibility']

    if num_satellites == 0 or num_fixed_points == 0:
        return

    satellite_positions = data_struct['satellites']['position']

    for i in range(num_satellites):
        sat_pos = satellite_positions[i]
        should_print_debug = (i == print_debug_for_sat)
        for j in range(num_fixed_points):
            fixed_point_pos = fixed_points[j]
            pointing_vector = fixed_point_pos - sat_pos
            data_struct['satellites']['pointing'][i] = pointing_vector
            visibility_table[j, i] = exclusion(data_struct, i, print_debug=should_print_debug)
