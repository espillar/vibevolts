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
    sat_category: str = 'satellites',
    print_debug: bool = False
) -> int:
    """
    Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.

    This function is vectorized to check for exclusions from all three bodies (Sun,
    Moon, Earth) simultaneously for a single satellite.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_index: The index of the satellite to check.
        sat_category: The satellite category (e.g. 'satellites', 'red_satellites').
        print_debug: If True, prints detailed debug information for the calculation.

    Returns:
        1 if the satellite's view is excluded by any of the bodies, 0 otherwise.
    """
    sat_pos = data_struct[sat_category]['position'][satellite_index]
    sat_pointing = data_struct['detector'].pointing[satellite_index]
    
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
    detector_props = data_struct['detector']
    exclusion_angles = np.array([
        detector_props.solarEx[satellite_index],
        detector_props.lunarEx[satellite_index],
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

    return 1 if np.any(is_excluded) else 0

def update_exclusion_table(
    data_struct: Dict[str, Any],
    print_debug_for_sat: Optional[int] = None
) -> None:
    """
    Updates the exclusion table for all satellites against all fixed points.
    This uses a highly optimized NumPy vectorized approach to compute all
    target-satellite exclusions simultaneously.
    """
    num_sats = data_struct['counts'].get('satellites', 0)
    num_fixed_points = data_struct['counts'].get('fixedpoints', 0)
    
    if num_sats == 0 or num_fixed_points == 0:
        return

    targets = data_struct['fixedpoints']['position']
    sat_pos = data_struct['satellites']['position']

    # Compute pointing vectors from each satellite to each fixed point
    pointing_vectors = targets[:, np.newaxis, :] - sat_pos[np.newaxis, :, :]
    
    norm_pointing = np.linalg.norm(pointing_vectors, axis=2)
    safe_norm = np.where(norm_pointing == 0, 1.0, norm_pointing)
    u_sat_pointing = pointing_vectors / safe_norm[:, :, np.newaxis]

    # Celestial body positions and radii
    body_positions = np.array([
        data_struct['celestial']['position'][0],
        data_struct['celestial']['position'][1],
        [0.0, 0.0, 0.0]
    ])
    body_radii = np.array([0.0, MOON_RADIUS, EARTH_RADIUS])

    # vecs_to_bodies shape: (num_sats, 3, 3)
    vecs_to_bodies = body_positions[np.newaxis, :, :] - sat_pos[:, np.newaxis, :]
    dists_to_bodies = np.linalg.norm(vecs_to_bodies, axis=2)

    # Normalize vectors to bodies
    safe_dists = np.where(dists_to_bodies == 0, 1.0, dists_to_bodies)
    u_vecs_to_bodies = vecs_to_bodies / safe_dists[:, :, np.newaxis]

    # Compute angles using einsum: (num_fixed_points, num_sats, 3)
    cos_angles = np.einsum('tsd,sbd->tsb', u_sat_pointing, u_vecs_to_bodies)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    # Apparent radii of Moon and Earth (Sun is 0)
    apparent_radii = np.zeros((num_sats, 3))
    apparent_radii[:, 1:] = np.arctan(body_radii[1:][np.newaxis, :] / safe_dists[:, 1:])

    # Get satellite detector exclusion angles
    detector_props = data_struct['detector']
    exclusion_angles = np.vstack([
        detector_props.solarEx,
        detector_props.lunarEx,
        detector_props.earthEx
    ]).T

    # Check exclusion: (num_fixed_points, num_sats, 3)
    is_excluded = (angles - apparent_radii[np.newaxis, :, :]) < exclusion_angles[np.newaxis, :, :]
    
    # Check if any celestial body causes exclusion
    exclusion_matrix = np.any(is_excluded, axis=2).astype(int)

    # If pointing vector norm is zero, it's not pointing anywhere (excluded/1)
    exclusion_matrix[norm_pointing < 1e-9] = 1

    # Print debug info if requested
    if print_debug_for_sat is not None and 0 <= print_debug_for_sat < num_sats:
        original_pointing = data_struct['detector'].pointing.copy()
        for j in range(num_fixed_points):
            data_struct['detector'].pointing[print_debug_for_sat] = (
                pointing_vectors[j, print_debug_for_sat]
            )
            exclusion(data_struct, print_debug_for_sat, print_debug=True)
        data_struct['detector'].pointing = original_pointing

    data_struct['fixedpoints']['exclusion'] = exclusion_matrix
