
import numpy as np
from lambertian import lambertiansphere, includedAngle
import radiometry_calcs
from constants import EARTH_RADIUS


def get_spherical_coords(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts 3D Cartesian coordinates to spherical angles (theta, phi).

    Args:
        arr: An (N, 3) NumPy array of Cartesian coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - theta: The inclination angle from the z-axis in radians [0, pi].
            - phi: The azimuthal angle in the x-y plane in radians [-pi, pi].
    """
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    r = np.linalg.norm(arr, axis=1)
    # theta: angle from the z-axis [0, pi]
    # We use np.divide to handle potential division by zero if r is 0
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    # phi: angle in the x-y plane [-pi, pi]
    phi = np.arctan2(y, x)
    return theta, phi

#######################################################


def scandetectors(sim_data: dict, print_output: int = 0, mask: np.ndarray = None):
    """
    Scans for and processes detector data within the simulation.

    This function orchestrates the process of determining which fixed targets
    are visible to each satellite's detector and calculates the resulting
    signal-to-noise ratio (SNR).

    The general flow involves:
    1.  **Data Extraction**: Relevant simulation parameters such as satellite positions,
        detector characteristics (pointing, FOV, integration time, aperture area),
        target properties (positions, sizes, albedo), and celestial body positions
        (e.g., Sun) are extracted from the `sim_data` dictionary. If a `mask` is 
        provided, only the specified subset of detectors is extracted.
    2.  **Background Flux Calculation**: Utilizes `radiometry_calcs.fluxes` to determine
        the background contributions from the Sun, space, and sky based on the
        detector's filter.
    3.  **Visibility Determination**: For the active satellites, the function calculates
        the angular separation between each detector's pointing vector and the
        vectors to all fixed targets using vectorized broadcasting. A 2D mask is then
        applied to identify only those target-detector pairs that fall within the
        respective fields of view (FOV).
    4.  **Detector Flux Calculation**: For the visible targets, the incident flux
        on the detector is calculated using the `lambertian.lambertiansphere` model
        in a vectorized manner. This calculation considers the Sun's position, the 
        target's position, albedo, radius, and the solar flux.
    5.  **SNR Calculation**: Based on the calculated detector flux, the integration time,
        and the detector's aperture area, the signal and noise levels are determined
        simultaneously for all detections. The signal-to-noise ratio (SNR) is then 
        computed from these values.
    6.  **Output/Storage**: The calculated signal, noise, and SNR for each detected
        target are returned in a dictionary. If `print_output` is enabled, they are
        also printed to the console.

    Args:
        sim_data (dict): The main simulation data dictionary.
        print_output (int): If > 0, prints detailed results for each detection.
        mask (np.ndarray, optional): A boolean mask of length equal to the total number 
                                     of satellites, indicating which detectors to process.
                                     If None, all detectors are processed.

    Returns:
        dict: A dictionary containing simulation 'time' and arrays for 'sat_indices', 
              'target_indices', 'signal', 'noise', and 'snr'.

    TODO: we are using the same filter for all the detectors!
    """
    # 1. Data Extraction and Masking
    num_sats = sim_data.counts.get('satellites', 0)
    num_obs = sim_data.counts.get('observatories', 0)
    num_total_detectors = len(sim_data.detector.filt)
    sim_time = sim_data.time

    if mask is None:
        mask = np.ones(num_total_detectors, dtype=bool)
    
    active_indices = np.where(mask)[0]

    # Initialize empty result structure
    results = {
        'time': sim_time,
        'sat_indices': np.array([], dtype=int),
        'target_indices': np.array([], dtype=int),
        'signal': np.array([], dtype=float),
        'noise': np.array([], dtype=float),
        'snr': np.array([], dtype=float)
    }

    if len(active_indices) == 0:
        return results

    # Build a position array aligned to the detector array order.
    # Each detector entry carries its category and asset_index so we look up
    # the correct row from the right position sub-array, regardless of the
    # order in which satellites / observatories were added.
    category_array = np.array(sim_data.detector.category)   # shape (num_total_detectors,)
    asset_index_array = sim_data.detector.asset_index        # shape (num_total_detectors,)

    # Pre-fetch position arrays once (avoids repeated attribute access)
    _pos_map = {}
    if num_sats > 0:
        _pos_map['satellites'] = sim_data.satellites.position
    if num_obs > 0:
        _pos_map['observatories'] = sim_data.observatories.position

    all_positions = np.zeros((num_total_detectors, 3), dtype=float)
    for det_i in range(num_total_detectors):
        cat = category_array[det_i]
        pos_array = _pos_map.get(cat)
        if pos_array is not None:
            all_positions[det_i] = pos_array[asset_index_array[det_i]]

    # Subset detector and observer positions
    satpositions = all_positions[mask]
    detectorVect = sim_data.detector.pointing[mask]
    fovs = sim_data.detector.fov[mask]
    integrationTime = sim_data.detector.integrationTime[mask]
    apertureArea = sim_data.detector.apertureArea[mask]
    pixelOmega = sim_data.detector.pixelOmega[mask]
    qe = sim_data.detector.qe[mask]
    
    # Target and celestial data
    targets = sim_data.fixedpoints.position
    sunVect = sim_data.celestial.position[0]
    albedo = sim_data.fixedpoints.albedo
    radius = sim_data.fixedpoints.size / 2

    # 2. Background Flux Calculation
    # Assuming same filter for active group (as per existing TODO)
    filter_name = sim_data.detector.filt[active_indices[0]]
    sun, space, sky = radiometry_calcs.fluxes(filter_name)

    if print_output:
        print(f'Using filter: {filter_name}')
        print(f'sun, space, sky \n    {sun:.3e}, {space:.3e}, {sky:.3e}')

    # 3. Visibility Determination (Vectorized Geometry)
    # toTargets shape: (num_active_sats, num_targets, 3)
    toTargets = targets[np.newaxis, :, :] - satpositions[:, np.newaxis, :]
    
    # Calculate norms for normalization and distance
    norms_toTargets = np.linalg.norm(toTargets, axis=2)
    norms_toTargets_safe = np.where(norms_toTargets == 0, 1.0, norms_toTargets)
    
    # Normalized vectors from sats to targets
    normalized_toTargets = toTargets / norms_toTargets_safe[:, :, np.newaxis]
    
    # Vectorized dot product for angles: (num_active_sats, num_targets)
    dot_products = np.einsum('si,sti->st', detectorVect, normalized_toTargets)
        # This line uses NumPy's Einstein Summation (einsum) to perform a
        # high-performance, vectorized dot   product between the satellite
        # pointing vectors and the vectors to every target.
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

    # Apply FOV mask: (num_active_sats, num_targets)
    # fov is the full field-of-view diameter; a target is visible
    # when its angular offset from boresight is less than fov / 2.
    visible_mask = angles < (fovs[:, np.newaxis] / 2)
    sat_hit_idx, target_hit_idx = np.where(visible_mask)
    
    if len(sat_hit_idx) == 0:
        if print_output:
            print("No targets visible to active detectors.")
        return results

    # 3.5 Occlusion Filter (Earth Limb & Horizon)
    hit_obs_pos = satpositions[sat_hit_idx]
    hit_toTargets_pre = toTargets[sat_hit_idx, target_hit_idx]
    hit_norms_pre = norms_toTargets[sat_hit_idx, target_hit_idx]
    
    u_hit_toTargets = hit_toTargets_pre / hit_norms_pre[:, np.newaxis]
    obs_norms = np.linalg.norm(hit_obs_pos, axis=1)
    valid_obs = obs_norms > 1e-9
    
    u_nadir = np.zeros_like(hit_obs_pos)
    u_nadir[valid_obs] = -hit_obs_pos[valid_obs] / obs_norms[valid_obs, np.newaxis]
    
    cos_angle_to_nadir = np.sum(u_hit_toTargets * u_nadir, axis=1)
    angle_to_nadir = np.arccos(np.clip(cos_angle_to_nadir, -1.0, 1.0))
    
    hit_earthEx = sim_data.detector.earthEx[mask][sat_hit_idx]
    hit_categories = np.array(sim_data.detector.category)[mask][sat_hit_idx]
    
    obs_is_ground = (hit_categories == 'observatories')

    # Ground observatory horizon check: angle from zenith must be <= (90deg - min_elevation)
    angle_to_zenith = np.pi - angle_to_nadir
    max_zenith = np.pi / 2.0 - hit_earthEx
    horizon_occluded = obs_is_ground & valid_obs & (angle_to_zenith > max_zenith)

    # Spacecraft Earth limb check: angle to Earth center must be >= (apparent_radius + earthEx)
    safe_obs_norms = np.where(obs_norms > EARTH_RADIUS, obs_norms, EARTH_RADIUS)
    apparent_radius = np.where(
        obs_norms > EARTH_RADIUS,
        np.arcsin(EARTH_RADIUS / safe_obs_norms),
        np.pi / 2.0
    )
    limb_occluded = (~obs_is_ground) & valid_obs & (angle_to_nadir < (apparent_radius + hit_earthEx))

    is_occluded = horizon_occluded | limb_occluded

    # Filter out occluded hits
    clear_hits = ~is_occluded
    sat_hit_idx = sat_hit_idx[clear_hits]
    target_hit_idx = target_hit_idx[clear_hits]
    
    if len(sat_hit_idx) == 0:
        if print_output:
            print("No targets visible (all occluded) to active detectors.")
        return results

    # 4. Detector Flux Calculation (Vectorized Radiometry)
    # Subset data for specific detector-target pairs (hits)
    hit_toTargets = toTargets[sat_hit_idx, target_hit_idx]
    hit_norms = norms_toTargets[sat_hit_idx, target_hit_idx]
    hit_albedo = albedo[target_hit_idx]
    hit_radius = radius[target_hit_idx]
    
    # Lambertian Phase Angle Calculation
    vec_from_sphere_to_observer = -hit_toTargets
    vec_from_sphere_to_light = np.tile(sunVect, (len(sat_hit_idx), 1))
    
    angle_light_observer = includedAngle(vec_from_sphere_to_light, vec_from_sphere_to_observer)

    target_brightness = lambertiansphere(
        angle_light_observer,
        hit_albedo,
        hit_radius,
        sun,
        debug=0 
    )

    detectorFlux = target_brightness / (np.pi * hit_norms ** 2)

    # 5. SNR Calculation
    hit_itime = integrationTime[sat_hit_idx]
    hit_aper = apertureArea[sat_hit_idx]
    hit_qe = qe[sat_hit_idx]
    hit_omega = pixelOmega[sat_hit_idx]
    hit_photoEff = sim_data.detector.photoEff[mask][sat_hit_idx]

    signal = detectorFlux * hit_itime * hit_aper * hit_qe * hit_photoEff
    noise = np.sqrt(space * hit_itime * hit_aper * hit_omega * hit_qe)
    snr = signal / noise

    # 6. Output/Storage
    if print_output:
        for k in range(len(sat_hit_idx)):
            i = active_indices[sat_hit_idx[k]]
            j = target_hit_idx[k]
            print(f"Sat {i} (itime={hit_itime[k]:.1f}s) detects Target {j}:")
            print(f"  Signal: {signal[k]:12.3e} | Noise: {noise[k]:12.3e} | SNR: {snr[k]:12.3e}")

    results.update({
        'sat_indices': active_indices[sat_hit_idx],
        'target_indices': target_hit_idx,
        'signal': signal,
        'noise': noise,
        'snr': snr
    })

    return results


