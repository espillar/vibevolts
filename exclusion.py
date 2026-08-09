import numpy as np
from typing import Dict, Any, Tuple, Optional

from constants import (
    EARTH_RADIUS, MOON_RADIUS
)
def exclusion(
    data_struct: Dict[str, Any],
    satellite_index: int,
    print_debug: bool = False
) -> int:
    """
    Determines if a satellite's pointing vector is excluded
    by the Sun, Moon, or Earth.

    This function is vectorized to check for exclusions from
    all three bodies (Sun, Moon, Earth) simultaneously for a
    single satellite.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_index: The index of the detector to check.
        print_debug: If True, prints detailed debug
            information for the calculation.

    Returns:
        1 if the detector's view is excluded by any of the
        bodies, 0 otherwise.
    """
    detector_props = data_struct.get_all_detectors()
    if detector_props is None or len(detector_props) == 0:
        return 1

    sat_positions = data_struct.get_detector_positions()
    sat_pos = sat_positions[satellite_index]
    sat_pointing = detector_props.pointing[satellite_index]

    norm_pointing = np.linalg.norm(sat_pointing)
    if norm_pointing < 1e-9:
        return 1  # Not pointing anywhere, treat as excluded

    u_sat_pointing = sat_pointing / norm_pointing

    # Local horizon exclusion check for ground-based observatories
    num_sats = len(data_struct.satellites.detector) if (data_struct.satellites and getattr(data_struct.satellites, 'detector', None)) else 0
    if satellite_index >= num_sats and data_struct.observatories:
        norm_pos = np.linalg.norm(sat_pos)
        if norm_pos > 1e-9:
            zenith_normal = sat_pos / norm_pos
            cos_zenith = np.clip(np.dot(zenith_normal, u_sat_pointing), -1.0, 1.0)
            zenith_angle = np.arccos(cos_zenith)
            # Minimum elevation angle is stored in earthEx
            min_elevation = detector_props.earthEx[satellite_index]
            max_zenith = np.pi / 2.0 - min_elevation
            if zenith_angle > max_zenith:
                return 1
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

    is_observatory = (satellite_index >= num_sats and data_struct.observatories is not None)
    if is_observatory:
        # For ground stations, Earth limb checks are replaced by the horizon check above.
        is_excluded[2] = False

    if print_debug:
        body_names = ["Sun", "Moon", "Earth"]
        asset_name = "Observatory" if is_observatory else "Satellite"
        asset_idx = (satellite_index - num_sats) if is_observatory else satellite_index
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
    detector_props = data_struct.get_all_detectors()
    if detector_props is None or len(detector_props.filt) == 0 or num_fixed_points == 0:
        return

    num_detectors = len(detector_props.filt)
    targets = data_struct.fixedpoints.position

    # --- Build observer positions using get_detector_positions ---
    observer_pos = data_struct.get_detector_positions()

    # --- Pointing vectors: (num_targets, num_detectors, 3) ---
    pointing_vectors = (
        targets[:, np.newaxis, :]
        - observer_pos[np.newaxis, :, :]
    )

    dist_to_target = np.linalg.norm(pointing_vectors, axis=2)
    safe_norm = np.where(dist_to_target == 0, 1.0, dist_to_target)
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
    exclusion_angles = np.column_stack([
        detector_props.solarEx,
        detector_props.lunarEx,
        detector_props.earthEx
    ])

    # Exclusion test: (num_targets, num_detectors, 3_bodies)
    is_excluded = (
        (angles - apparent_radii[np.newaxis, :, :])
        < exclusion_angles[np.newaxis, :, :]
    )

    # --- Observatory horizon check (replaces Earth-limb) ---
    num_sats = len(data_struct.satellites.detector) if (data_struct.satellites and getattr(data_struct.satellites, 'detector', None)) else 0
    num_obs = len(data_struct.observatories.detector) if (data_struct.observatories and getattr(data_struct.observatories, 'detector', None)) else 0
    if num_obs > 0:
        obs_indices = np.arange(num_sats, num_sats + num_obs)
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

    # Zero distance to target → treat as excluded
    exclusion_matrix[dist_to_target < 1e-9] = 1

    # --- Debug output ---
    if (print_debug_for_sat is not None
            and 0 <= print_debug_for_sat < num_detectors):
        original_pointing = detector_props.pointing.copy()
        for j in range(num_fixed_points):
            detector_props.pointing[print_debug_for_sat] = (
                pointing_vectors[j, print_debug_for_sat]
            )
            exclusion(
                data_struct,
                print_debug_for_sat,
                print_debug=True,
            )
        detector_props.pointing = original_pointing

    data_struct.fixedpoints.exclusion = exclusion_matrix

