import numpy as np
from typing import Dict, Any, Tuple, Optional

from constants import (
    EARTH_RADIUS, MOON_RADIUS
)

#def solarexclusion(data_struct: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
#    """
#    Calculates solar exclusion for all satellites based on their pointing vectors.
#
#    This function operates in a vectorized manner on all satellites in the
#    'satellites' category. It computes the angle between each satellite's
#    pointing vector and the vector from the satellite to the Sun.#
#
#    Args:
#        data_struct: The main simulation data dictionary.
#
#    Returns:
#        A tuple containing:
#        - exclusion_vector (np.ndarray): An array of the same length as the
#          number of satellites. An element is 1 if the satellite is within
#          the solar exclusion angle, 0 otherwise.
#        - angle_vector (np.ndarray): An array containing the calculated angle
#          in radians for each satellite.
#    """
#    num_sats = data_struct['counts']['satellites']
#    if num_sats == 0:
#        return np.array([]), np.array([])
#
#    sun_pos = data_struct['celestial']['position'][0]
#    sat_pos = data_struct['satellites']['position']
#    sat_pointing = data_struct['satellites']['pointing']
#    solar_exclusion_angles = data_struct['satellites']['detector'].solarEx
#
#    vec_sat_to_sun = sun_pos - sat_pos
#
#    norm_sat_to_sun = np.linalg.norm(vec_sat_to_sun, axis=1)
#    norm_sat_pointing = np.linalg.norm(sat_pointing, axis=1)
#
#    valid_norms = (norm_sat_to_sun > 1e-9) & (norm_sat_pointing > 1e-9)
#
#    angle_vector = np.full(num_sats, np.pi)
#
#    if np.any(valid_norms):
#        dot_product = np.einsum('ij,ij->i', vec_sat_to_sun[valid_norms], sat_pointing[valid_norms])
#        cos_angle = dot_product / (norm_sat_to_sun[valid_norms] * norm_sat_pointing[valid_norms])
#        cos_angle = np.clip(cos_angle, -1.0, 1.0)
#        angle_vector[valid_norms] = np.arccos(cos_angle)
#
#    exclusion_vector = (angle_vector < solar_exclusion_angles).astype(int)
#
#    return exclusion_vector, angle_vector

def exclusion(
    data_struct: Dict[str, Any],
    satellite_index: int,
    print_debug: bool = False
) -> int:
    """
    Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.

    This function is vectorized to check for exclusions from all three bodies (Sun,
    Moon, Earth) simultaneously for a single satellite.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_index: The index of the satellite to check.
        print_debug: If True, prints detailed debug information for the calculation.

    Returns:
        0 if the satellite's view is excluded by any of the bodies, 1 otherwise.
    """
    sat_pos = data_struct['satellites']['position'][satellite_index]
    sat_pointing = data_struct['satellites']['pointing'][satellite_index]
    
    norm_pointing = np.linalg.norm(sat_pointing)
    if norm_pointing < 1e-9:
        return 1  # Not pointing anywhere, so not excluded

    u_sat_pointing = sat_pointing / norm_pointing

    # Celestial body positions and radii
    body_positions = np.array([
        data_struct['celestial']['position'][0],  # Sun
        data_struct['celestial']['position'][1],  # Moon
        [0.0, 0.0, 0.0]  # Earth's center
    ])
    body_radii = np.array([0.0, MOON_RADIUS, EARTH_RADIUS])

    # Satellite-specific exclusion angles
    detector_props = data_struct['satellites']['detector']
    exclusion_angles = np.array([
        detector_props.solarEx[satellite_index],
        detector_props.lunarex[satellite_index],
        detector_props.earthEx[satellite_index]
    ])

    # --- Vectorized Calculations ---
    vecs_to_bodies = body_positions - sat_pos
    dists_to_bodies = np.linalg.norm(vecs_to_bodies, axis=1)

    # Calculate angles to bodies, handling potential division by zero
    angles = np.full(3, np.pi)
    valid_dists = dists_to_bodies > 1e-9
    
    u_vecs_to_bodies = np.zeros_like(vecs_to_bodies)
    u_vecs_to_bodies[valid_dists] = vecs_to_bodies[valid_dists] / dists_to_bodies[valid_dists, np.newaxis]

    cos_angles = np.clip(np.dot(u_vecs_to_bodies, u_sat_pointing), -1.0, 1.0)
    angles = np.arccos(cos_angles)

    # Calculate apparent radii (for Moon and Earth)
    apparent_radii = np.zeros(3)
    non_zero_dists = (dists_to_bodies[1:] > 0)
    apparent_radii[1:][non_zero_dists] = np.arctan(body_radii[1:][non_zero_dists] / dists_to_bodies[1:][non_zero_dists])

    # Check for exclusion
    is_excluded = (angles - apparent_radii) < exclusion_angles

    if print_debug:
        body_names = ["Sun", "Moon", "Earth"]
        print(f"--- Exclusion Debug for Satellite {satellite_index} ---")
        for i in range(3):
            print(f"  - {body_names[i]:<5} Flag: {is_excluded[i]}, "
                  f"Angle: {np.rad2deg(angles[i]):.2f} deg, "
                  f"Excl: {np.rad2deg(exclusion_angles[i]):.2f} deg")

    return 0 if np.any(is_excluded) else 1

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

    # Ensure the visibility table has the correct shape
    if visibility_table.shape != (num_fixed_points, num_satellites):
        data_struct['fixedpoints']['visibility'] = np.zeros((num_fixed_points, num_satellites), dtype=int)
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
