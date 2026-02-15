import numpy as np
from lambertian import lambertiansphere, includedAngle
import radiometry_calcs


def get_spherical_coords(arr):
    """
    given a n by 3 d array, return two 1D arrays
    with the theta and phi angles in radians
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


def scandetectors(sim_data: dict):
    """
    Scans for and processes detector data within the simulation.

    This function orchestrates the process of determining which fixed targets
    are visible to each satellite's detector and calculates the resulting
    signal-to-noise ratio (SNR).

    The general flow involves:
    1.  **Data Extraction**: Relevant simulation parameters such as satellite positions,
        detector characteristics (pointing, FOV, integration time, aperture area),
        target properties (positions, sizes, albedo), and celestial body positions
        (e.g., Sun) are extracted from the `sim_data` dictionary.
    2.  **Background Flux Calculation**: Utilizes `radiometry_calcs.fluxes` to determine
        the background contributions from the Sun, space, and sky based on the
        detector's filter.
    3.  **Visibility Determination**: For each satellite, the function calculates
        the angular separation between the detector's pointing vector and the
        vector from the satellite to each fixed target. A mask is then applied
        to identify only those targets that fall within the detector's field of view (FOV).
    4.  **Detector Flux Calculation**: For the visible targets, the incident flux
        on the detector is calculated using the `lambertian.lambertiansphere` model.
        This calculation considers the Sun's position, the target's position, albedo,
        radius, and the solar flux.
    5.  **SNR Calculation**: Based on the calculated detector flux, the integration time,
        and the detector's aperture area, the signal and noise levels are determined.
        The signal-to-noise ratio (SNR) is then computed from these values.
    6.  **Output/Storage**: The calculated signal, noise, and SNR for each detected
        target are made available, currently printed to console. Future enhancements
        might include storing these results in a structured format (e.g., pandas DataFrame).

    Args:
        sim_data (dict): The main simulation data dictionary.
                         This dictionary is expected to contain all
                         relevant simulation state and parameters.

    TODO: we are using the same filter for all the detectors!
    """
    # print("scansensors function called with simulation data.")

    
    satpositions = sim_data['satellites']['position']  # all sat positions
    detectorVect = sim_data['detector'].pointing  # detector pointings

    #    print('detectorVect ', detectorVect)

    detectorFov = sim_data['detector'].fov  # detector fields of view
    integrationTime = sim_data['detector'].integrationTime
    fovs = sim_data['detector'].fov  # field of view of the detector
    detectorArea = sim_data['detector'].apertureArea

    
    targets = sim_data['fixedpoints']['position']  # all target positions
    targetSize = sim_data['fixedpoints']['size']  # all target sizes
    sunVect = sim_data['celestial']['position'][0]  # sun position

    sun, space, sky = radiometry_calcs.fluxes(sim_data['detector'].filt[0])
    print(f'sun, space, sky  {sun:.3e}, {space:.3e}, {sky:.3e}')
    albedo = sim_data['fixedpoints']['albedo']
    radius = sim_data['fixedpoints']['size']/2
    

    # for i in range(1):
    for i in range(sim_data['counts']['satellites']):    
        satposition = satpositions[i, :]
        ray = detectorVect[i, :]
        toTargets = targets - satposition
        dot_products = np.einsum('ij,j->i', toTargets, ray)
        norms_V = np.linalg.norm(toTargets, axis=1)
        norms_W = np.linalg.norm(ray, axis=0)
        angles = np.arccos(np.clip(dot_products /
                                   (norms_V * norms_W), -1.0, 1.0))
        fov = fovs[i]
        mask = angles < fov
        visibleIndices = np.flatnonzero(mask)
        vec_from_sphere_to_light_expanded = np.tile(-sunVect, (len(visibleIndices), 1))
        vec_from_sphere_to_observer_actual = -toTargets[mask]
        
        angle_light_observer = includedAngle(vec_from_sphere_to_light_expanded, vec_from_sphere_to_observer_actual)
        norm_observer = np.linalg.norm(vec_from_sphere_to_observer_actual, axis=1)

        emitted_brightness = lambertiansphere(
            angle_light_observer,
            albedo[mask],
            radius[mask],
            sun
        )
        detectorFlux = emitted_brightness / (4 * np.pi * norm_observer ** 2)
        signal = detectorFlux * integrationTime[i] * detectorArea[i]
        noise = np.sqrt(detectorFlux * integrationTime[i] * detectorArea[i] +
                        space * integrationTime[i] * detectorArea[i])
        snr = signal/noise
        print('SignAl, noise, snr, integration time \n')
        for j in range(len(signal)):
            print(f"{signal[j]:.3e}, {noise[j]:.3e}, {snr[j]:.3e}, {integrationTime[i]:.3e}")

    return(0)

