import numpy as np
from typing import Dict, Any, Tuple, Optional

from constants import (
    EARTH_RADIUS, MOON_RADIUS
)
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
    # Determine asset category and index using the detector's internal tracking
    detector_props = data_struct.detector
    asset_category = detector_props.category[satellite_index]
    asset_idx = detector_props.asset_index[satellite_index]

    sat_pos = getattr(data_struct, asset_category).position[asset_idx]
    sat_pointing = data_struct.detector.pointing[satellite_index]
    
    norm_pointing = np.linalg.norm(sat_pointing)
    if norm_pointing < 1e-9:
        return 1  # Not pointing anywhere, treat as excluded

    u_sat_pointing = sat_pointing / norm_pointing

    detector_props = data_struct.detector

    # Local horizon exclusion check for ground-based observatories
    if asset_category == 'observatories':
        norm_pos = np.linalg.norm(sat_pos)
        if norm_pos > 1e-9:
            zenith_normal = sat_pos / norm_pos
            cos_zenith = np.clip(np.dot(zenith_normal, u_sat_pointing), -1.0, 1.0)
            zenith_angle = np.arccos(cos_zenith)
            # Minimum elevation angle is stored in earthEx
            min_elevation = detector_props.earthEx[satellite_index]
            max_zenith = np.pi / 2.0 - min_elevation
            if zenith_angle > max_zenith:
                if print_debug:
                    print(f"--- Horizon Exclusion for Observatory {asset_idx} ---")
                    print(f"  Zenith Angle: {np.rad2deg(zenith_angle):.2f} deg, "
                          f"Max Zenith: {np.rad2deg(max_zenith):.2f} deg (elev < {np.rad2deg(min_elevation):.2f} deg)")
                return 1

    # Celestial body positions and radii
    body_positions = np.array([
        data_struct.celestial.position[0],  # Sun
        data_struct.celestial.position[1],  # Moon
        [0.0, 0.0, 0.0]  # Earth's center
    ])
    body_radii = np.array([0.0, MOON_RADIUS, EARTH_RADIUS])

    # Satellite-specific exclusion angles
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

    if asset_category == 'observatories':
        # For ground stations, Earth limb checks are replaced by the horizon check above.
        is_excluded[2] = False

    if print_debug:
        body_names = ["Sun", "Moon", "Earth"]
        asset_name = "Observatory" if asset_category == "observatories" else "Satellite"
        print(f"--- Exclusion Debug for {asset_name} {asset_idx} ---")
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
    Updates the exclusion table for all detectors against all
    fixed points.  Uses the detector category / asset_index
    mapping so that satellites and observatories are handled
    uniformly.

    For space-based detectors the standard Sun / Moon / Earth-limb
    exclusion checks are applied.  For ground-based observatories
    the Earth-limb check is replaced by a local-horizon elevation
    check (the minimum elevation is stored in detector.earthEx).

    The resulting matrix is stored in
    data_struct.fixedpoints.exclusion with shape
    (num_fixed_points, num_detectors).
    """
    num_fixed_points = data_struct.counts.get('fixedpoints', 0)
    num_detectors = len(data_struct.detector.filt)

    if num_detectors == 0 or num_fixed_points == 0:
        return

    targets = data_struct.fixedpoints.position
    detector_props = data_struct.detector

    # --- Build observer positions using category / asset_index ---
    category_array = np.array(detector_props.category)
    asset_index_array = detector_props.asset_index

    num_sats = data_struct.counts.get('satellites', 0)
    num_obs = data_struct.counts.get('observatories', 0)

    _pos_map = {}
    if num_sats > 0:
        _pos_map['satellites'] = data_struct.satellites.position
    if num_obs > 0:
        _pos_map['observatories'] = (
            data_struct.observatories.position
        )

    observer_pos = np.zeros((num_detectors, 3), dtype=float)
    for det_i in range(num_detectors):
        cat = category_array[det_i]
        pos_array = _pos_map.get(cat)
        if pos_array is not None:
            observer_pos[det_i] = (
                pos_array[asset_index_array[det_i]]
            )

    # --- Pointing vectors: (num_targets, num_detectors, 3) ---
    pointing_vectors = (
        targets[:, np.newaxis, :]
        - observer_pos[np.newaxis, :, :]
    )

    norm_pointing = np.linalg.norm(pointing_vectors, axis=2)
    safe_norm = np.where(norm_pointing == 0, 1.0, norm_pointing)
    u_pointing = pointing_vectors / safe_norm[:, :, np.newaxis]

    # --- Celestial body geometry ---
    body_positions = np.array([
        data_struct.celestial.position[0],
        data_struct.celestial.position[1],
        [0.0, 0.0, 0.0]
    ])
    body_radii = np.array([0.0, MOON_RADIUS, EARTH_RADIUS])

    # vecs_to_bodies: (num_detectors, 3_bodies, 3_xyz)
    vecs_to_bodies = (
        body_positions[np.newaxis, :, :]
        - observer_pos[:, np.newaxis, :]
    )
    dists_to_bodies = np.linalg.norm(vecs_to_bodies, axis=2)
    safe_dists = np.where(
        dists_to_bodies == 0, 1.0, dists_to_bodies
    )
    u_vecs_to_bodies = (
        vecs_to_bodies / safe_dists[:, :, np.newaxis]
    )

    # Angles: (num_targets, num_detectors, 3_bodies)
    #   t = target, d = detector, b = body, i = xyz
    cos_angles = np.einsum(
        'tdi,dbi->tdb', u_pointing, u_vecs_to_bodies
    )
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    # Apparent angular radii: (num_detectors, 3_bodies)
    apparent_radii = np.zeros((num_detectors, 3))
    apparent_radii[:, 1:] = np.arctan(
        body_radii[1:][np.newaxis, :] / safe_dists[:, 1:]
    )

    # Exclusion angles: (num_detectors, 3_bodies)
    exclusion_angles = np.vstack([
        detector_props.solarEx,
        detector_props.lunarEx,
        detector_props.earthEx
    ]).T

    # Exclusion test: (num_targets, num_detectors, 3_bodies)
    is_excluded = (
        (angles - apparent_radii[np.newaxis, :, :])
        < exclusion_angles[np.newaxis, :, :]
    )

    # --- Observatory horizon check (replaces Earth-limb) ---
    obs_mask = (category_array == 'observatories')
    if np.any(obs_mask):
        obs_indices = np.where(obs_mask)[0]
        obs_positions = observer_pos[obs_indices]
        obs_norms = np.linalg.norm(obs_positions, axis=1)
        safe_obs_norms = np.where(
            obs_norms == 0, 1.0, obs_norms
        )
        zenith_vecs = (
            obs_positions / safe_obs_norms[:, np.newaxis]
        )

        # cos(zenith angle) for every (target, obs) pair
        cos_zenith = np.einsum(
            'toi,oi->to',
            u_pointing[:, obs_indices, :],
            zenith_vecs
        )
        cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
        zenith_angles = np.arccos(cos_zenith)

        min_elev = exclusion_angles[obs_indices, 2]
        max_zenith = np.pi / 2.0 - min_elev

        # Replace Earth-body exclusion with horizon check
        is_excluded[:, obs_indices, 2] = (
            zenith_angles > max_zenith[np.newaxis, :]
        )

    # Collapse body axis → scalar excluded flag
    exclusion_matrix = np.any(is_excluded, axis=2).astype(int)

    # Zero-norm pointing → treat as excluded
    exclusion_matrix[norm_pointing < 1e-9] = 1

    # --- Debug output ---
    if (print_debug_for_sat is not None
            and 0 <= print_debug_for_sat < num_detectors):
        original_pointing = data_struct.detector.pointing.copy()
        for j in range(num_fixed_points):
            data_struct.detector.pointing[print_debug_for_sat] = (
                pointing_vectors[j, print_debug_for_sat]
            )
            exclusion(
                data_struct,
                print_debug_for_sat,
                print_debug=True,
            )
        data_struct.detector.pointing = original_pointing

    data_struct.fixedpoints.exclusion = exclusion_matrix

