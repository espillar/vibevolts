import numpy as np
from typing import Any, Optional, Tuple, Union

from constants import EARTH_RADIUS, MOON_RADIUS


def compute_exclusions(
    obs_pos: np.ndarray,
    u_pointing: np.ndarray,
    is_ground: np.ndarray,
    exclusion_angles: np.ndarray,
    body_positions: np.ndarray,
    body_radii: np.ndarray,
    use_arcsin: bool = False,
    return_angles: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Vectorized exclusion and limb/horizon occlusion check.

    Computes whether each pointing direction is excluded by any celestial body
    (or below the local horizon for ground-based observatories, which replaces
    the final body in `body_positions` assumed to be Earth).

    Args:
        obs_pos: Observer positions in GCRS, shape (D, 3).
        u_pointing: Unit pointing vectors, shape (..., D, 3).
        is_ground: Boolean mask of ground observatories, shape (D,).
        exclusion_angles: Exclusion angles in radians, shape (D, B).
        body_positions: Celestial body positions in GCRS, shape (B, 3).
        body_radii: Physical radii of the bodies in meters, shape (B,).
        use_arcsin: If True, uses exact arcsin(R / d) for apparent angular
            radius (clamped to pi/2 inside the body). Otherwise uses arctan(R / d).
        return_angles: If True, returns (is_excluded, angles).

    Returns:
        Boolean array of shape (..., D, B), or (is_excluded, angles) if
        return_angles is True.
    """
    vecs_to_bodies = body_positions[np.newaxis, :, :] - obs_pos[:, np.newaxis, :]
    dists_to_bodies = np.linalg.norm(vecs_to_bodies, axis=2)  # (D, B)
    valid_dists = dists_to_bodies > 1e-9

    u_vecs_to_bodies = np.zeros_like(vecs_to_bodies)
    u_vecs_to_bodies[valid_dists] = (
        vecs_to_bodies[valid_dists] / dists_to_bodies[valid_dists, np.newaxis]
    )

    cos_angles = np.einsum('...di,dbi->...db', u_pointing, u_vecs_to_bodies)
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    apparent_radii = np.zeros_like(dists_to_bodies)
    non_zero = (body_radii[np.newaxis, :] > 0) & valid_dists
    if np.any(non_zero):
        radii_bcast = np.broadcast_to(body_radii[np.newaxis, :], dists_to_bodies.shape)
        if use_arcsin:
            ratio = np.clip(
                radii_bcast[non_zero] / np.maximum(dists_to_bodies[non_zero], radii_bcast[non_zero]),
                0.0,
                1.0,
            )
            apparent_radii[non_zero] = np.arcsin(ratio)
        else:
            apparent_radii[non_zero] = np.arctan(
                radii_bcast[non_zero] / dists_to_bodies[non_zero]
            )

    is_excluded = (angles - apparent_radii) < exclusion_angles

    # Replace Earth-limb check (last body) with local-horizon check for ground observatories
    if np.any(is_ground):
        obs_norms = np.linalg.norm(obs_pos[is_ground], axis=1)
        valid_obs = obs_norms > 1e-9
        zenith_vecs = np.zeros_like(obs_pos[is_ground])
        zenith_vecs[valid_obs] = (
            obs_pos[is_ground][valid_obs] / obs_norms[valid_obs, np.newaxis]
        )
        cos_zenith = np.einsum(
            '...di,di->...d', u_pointing[..., is_ground, :], zenith_vecs
        )
        zenith_angles = np.arccos(np.clip(cos_zenith, -1.0, 1.0))
        max_zenith = np.pi / 2.0 - exclusion_angles[is_ground, -1]
        horizon_excl = valid_obs & (zenith_angles > max_zenith)
        is_excluded[..., is_ground, -1] = horizon_excl

    # Observers at the origin have no well-defined Earth limb
    obs_all_norms = np.linalg.norm(obs_pos, axis=1)
    zero_obs = obs_all_norms <= 1e-9
    if np.any(zero_obs):
        is_excluded[..., zero_obs, -1] = False

    if return_angles:
        return is_excluded, angles
    return is_excluded


def check_sim_exclusions(
    sim_data: Any,
    u_pointing: np.ndarray,
    mask: Any = None,
    earth_only: bool = False,
    use_arcsin: bool = False,
    return_angles: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Packs observer positions, ground flags, detector exclusion angles, and
    celestial body positions/radii from sim_data and runs compute_exclusions.
    """
    obs_pos = sim_data.get_detector_positions(mask)
    is_ground = sim_data.get_detector_categories(mask) == 'observatories'
    det = sim_data.get_all_detectors()
    if mask is not None:
        det = det.subset(mask)
    if earth_only:
        exclusion_angles = det.earthEx[:, np.newaxis]
        body_positions = np.zeros((1, 3), dtype=float)
        body_radii = np.array([EARTH_RADIUS])
    else:
        exclusion_angles = np.column_stack([det.solarEx, det.lunarEx, det.earthEx])
        body_positions = np.array([
            sim_data.celestial.position[0],
            sim_data.celestial.position[1],
            [0.0, 0.0, 0.0],
        ])
        body_radii = np.array([0.0, MOON_RADIUS, EARTH_RADIUS])
    return compute_exclusions(
        obs_pos=obs_pos,
        u_pointing=u_pointing,
        is_ground=is_ground,
        exclusion_angles=exclusion_angles,
        body_positions=body_positions,
        body_radii=body_radii,
        use_arcsin=use_arcsin,
        return_angles=return_angles,
    )


def exclusion(
    data_struct: Any,
    satellite_index: int,
    print_debug: bool = False
) -> int:
    """
    Determines if a satellite's or observatory's pointing vector is excluded
    by the Sun, Moon, or Earth (or below the local horizon for observatories).
    """
    detector_props = data_struct.get_all_detectors()
    if detector_props is None or len(detector_props) == 0:
        return 1

    sat_pointing = detector_props.pointing[satellite_index]
    norm_pointing = np.linalg.norm(sat_pointing)
    if norm_pointing < 1e-9:
        return 1  # Not pointing anywhere, treat as excluded

    u_pointing = (sat_pointing / norm_pointing)[np.newaxis, np.newaxis, :]
    is_excluded, angles = check_sim_exclusions(
        data_struct, u_pointing, mask=[satellite_index], return_angles=True
    )

    excl_flags = is_excluded[0, 0]
    if print_debug:
        body_names = ["Sun", "Moon", "Earth"]
        is_observatory = (
            data_struct.get_detector_categories([satellite_index])[0] == 'observatories'
        )
        num_sats = (
            len(data_struct.satellites.detector)
            if (data_struct.satellites and getattr(data_struct.satellites, 'detector', None))
            else 0
        )
        asset_name = "Observatory" if is_observatory else "Satellite"
        asset_idx = (satellite_index - num_sats) if is_observatory else satellite_index
        excl_deg = np.rad2deg([
            detector_props.solarEx[satellite_index],
            detector_props.lunarEx[satellite_index],
            detector_props.earthEx[satellite_index],
        ])
        print(f"--- Exclusion Debug for {asset_name} {asset_idx} ---")
        for i in range(3):
            print(
                f"  - {body_names[i]:<5} Flag: {excl_flags[i]}, "
                f"Angle: {np.rad2deg(angles[0, 0, i]):.2f} deg, "
                f"Excl: {excl_deg[i]:.2f} deg"
            )

    return 1 if np.any(excl_flags) else 0


def update_exclusion_table(
    data_struct: Any,
    print_debug_for_sat: Optional[int] = None
) -> None:
    """
    Updates the exclusion table for all detectors against all fixed points.

    The resulting matrix is stored in data_struct.fixedpoints.exclusion with
    shape (num_fixed_points, num_detectors).
    """
    num_fixed_points = data_struct.counts.get('fixedpoints', 0)
    detector_props = data_struct.get_all_detectors()
    if detector_props is None or len(detector_props.filt) == 0 or num_fixed_points == 0:
        return

    num_detectors = len(detector_props.filt)
    targets = data_struct.fixedpoints.position
    observer_pos = data_struct.get_detector_positions()

    pointing_vectors = targets[:, np.newaxis, :] - observer_pos[np.newaxis, :, :]
    dist_to_target = np.linalg.norm(pointing_vectors, axis=2)
    safe_norm = np.where(dist_to_target == 0, 1.0, dist_to_target)
    u_pointing = pointing_vectors / safe_norm[:, :, np.newaxis]

    is_excluded = check_sim_exclusions(data_struct, u_pointing)
    exclusion_matrix = np.any(is_excluded, axis=2).astype(int)
    exclusion_matrix[dist_to_target < 1e-9] = 1

    if print_debug_for_sat is not None and 0 <= print_debug_for_sat < num_detectors:
        original_pointing = detector_props.pointing.copy()
        for j in range(num_fixed_points):
            detector_props.pointing[print_debug_for_sat] = (
                pointing_vectors[j, print_debug_for_sat]
            )
            exclusion(data_struct, print_debug_for_sat, print_debug=True)
        detector_props.pointing = original_pointing

    data_struct.fixedpoints.exclusion = exclusion_matrix
