
import numpy as np
from lambertian import lambertiansphere, includedAngle
import radiometry_calcs
from exclusion import check_sim_exclusions


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
        sim_data (SimulationState): The main simulation data structure.
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
    all_detectors = sim_data.get_all_detectors()
    num_total_detectors = len(all_detectors.filt) if (all_detectors and hasattr(all_detectors, 'filt')) else 0
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

    if len(active_indices) == 0 or all_detectors is None:
        return results

    # Subset observer positions and detector properties cleanly
    satpositions = sim_data.get_detector_positions(mask)
    det = all_detectors.subset(mask)

    # Target and celestial data
    targets = sim_data.fixedpoints.position
    sunVect = sim_data.celestial.position[0]

    # 2. Background Flux Calculation
    filter_name = sim_data.detector.filt[active_indices[0]]
    sun, space, sky = radiometry_calcs.fluxes(filter_name)

    if print_output:
        print(f'Using filter: {filter_name}')
        print(f'sun, space, sky \n    {sun:.3e}, {space:.3e}, {sky:.3e}')

    # 3. Visibility Determination (Vectorized Geometry)
    toTargets = targets[np.newaxis, :, :] - satpositions[:, np.newaxis, :]
    norms_toTargets = np.linalg.norm(toTargets, axis=2)
    norms_toTargets_safe = np.where(norms_toTargets == 0, 1.0, norms_toTargets)
    normalized_toTargets = toTargets / norms_toTargets_safe[:, :, np.newaxis]

    dot_products = np.einsum('si,sti->st', det.pointing, normalized_toTargets)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

    visible_mask = angles < (det.fov[:, np.newaxis] / 2)
    sat_hit_idx, target_hit_idx = np.where(visible_mask)

    if len(sat_hit_idx) == 0:
        if print_output:
            print("No targets visible to active detectors.")
        return results

    # 3.5 Occlusion Filter (Earth Limb & Horizon)
    hit_norms_pre = norms_toTargets[sat_hit_idx, target_hit_idx]
    u_hit_toTargets = toTargets[sat_hit_idx, target_hit_idx] / np.where(
        hit_norms_pre == 0, 1.0, hit_norms_pre
    )[:, np.newaxis]
    is_occluded = check_sim_exclusions(
        sim_data,
        u_pointing=u_hit_toTargets[np.newaxis, :, :],
        mask=active_indices[sat_hit_idx],
        earth_only=True,
        use_arcsin=True,
    )[0, :, 0]

    clear_hits = ~is_occluded
    sat_hit_idx = sat_hit_idx[clear_hits]
    target_hit_idx = target_hit_idx[clear_hits]

    if len(sat_hit_idx) == 0:
        if print_output:
            print("No targets visible (all occluded) to active detectors.")
        return results

    # 4. Detector Flux Calculation (Vectorized Radiometry)
    hit_toTargets = toTargets[sat_hit_idx, target_hit_idx]
    hit_norms = norms_toTargets[sat_hit_idx, target_hit_idx]
    hits = det.subset(sat_hit_idx)

    angle_light_observer = includedAngle(
        np.tile(sunVect, (len(sat_hit_idx), 1)), -hit_toTargets
    )
    target_brightness = lambertiansphere(
        angle_light_observer,
        sim_data.fixedpoints.albedo[target_hit_idx],
        sim_data.fixedpoints.size[target_hit_idx] / 2,
        sun,
        debug=0,
    )
    detectorFlux = target_brightness / (np.pi * hit_norms ** 2)

    # 5. SNR Calculation
    signal = detectorFlux * hits.integrationTime * hits.apertureArea * hits.qe * hits.photoEff
    noise = np.sqrt(space * hits.integrationTime * hits.apertureArea * hits.pixelOmega * hits.qe)
    snr = signal / noise

    # 6. Output/Storage
    if print_output:
        for k in range(len(sat_hit_idx)):
            i = active_indices[sat_hit_idx[k]]
            j = target_hit_idx[k]
            print(f"Sat {i} (itime={hits.integrationTime[k]:.1f}s) detects Target {j}:")
            print(f"  Signal: {signal[k]:12.3e} | Noise: {noise[k]:12.3e} | SNR: {snr[k]:12.3e}")

    results.update({
        'sat_indices': active_indices[sat_hit_idx],
        'target_indices': target_hit_idx,
        'signal': signal,
        'noise': noise,
        'snr': snr
    })

    return results


