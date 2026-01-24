# VibeVolts `sim_data` Dictionary Structure

This document details the structure of the main `sim_data` dictionary used throughout the VibeVolts simulation toolkit.

## Top-Level `sim_data` Keys

| Key | Type | Description | Functions Modifying This Key |
| --- | --- | --- | --- |
| `start_time` | `datetime` | The starting time and date of the simulation (UTC). | `simulation.create_empty_simulation` |
| `time` | `datetime` | The current time of the simulation. | `simulation.create_empty_simulation`, `celestialbodies.celestial_update`, `propagation.propagate_satellites`, `demo_constellation.demo_constellation`, `demo1.demo1`, `demo3.demo3` |
| `delta_time`| `float` | The time step for the simulation in seconds. | `simulation.create_empty_simulation` |
| `counts` | `dict` | A dictionary containing the counts of various objects. | `simulation.create_empty_simulation` (initializes) |
| | | `counts['celestial']` (int): Number of celestial bodies. | `celestialbodies.add_celestial_bodies` |
| | | `counts['satellites']` (int): Number of satellites. | `constellation.geos`, `constellation.geosmod`, `propagation.add_satellites_from_tle`, `testObjects.fixedSat` |
| | | `counts['observatories']` (int): Number of observatories. | `observatories.add_observatories` |
| | | `counts['fixedpoints']` (int): Number of fixed points. | `targets.add_fixed_points`, `testObjects.fixedTarget` |
| `pointing_spheres` | `dict` | A dictionary of pre-calculated pointing spheres, where keys are the number of points. | `simulation.create_empty_simulation` (initializes), `pointing.generate_pointing_sphere` |
| `celestial` | `dict` | Dictionary containing data for celestial bodies (Sun, Moon). | `celestialbodies.add_celestial_bodies`, `testObjects.fixSun` |
| | | `celestial['position']` (np.ndarray): `(2, 3)` array of positions. | `celestialbodies.add_celestial_bodies` (initializes), `celestialbodies.celestial_update` (modifies) |
| | | `celestial['velocity']` (np.ndarray): `(2, 3)` array of velocities. | `celestialbodies.add_celestial_bodies` |
| | | `celestial['acceleration']` (np.ndarray): `(2, 3)` array of accelerations. | `celestialbodies.add_celestial_bodies` |
| `satellites`| `dict` | Dictionary containing data for satellites. | `constellation.geos`, `constellation.geosmod`, `propagation.add_satellites_from_tle`, `testObjects.fixedSat` |
| | | `satellites['position']` (np.ndarray): `(n, 3)` array of GCRS positions. | `propagation.propagate_satellites` |
| | | `satellites['velocity']` (np.ndarray): `(n, 3)` array of GCRS velocities. | |
| | | `satellites['acceleration']` (np.ndarray): `(n, 3)` array of GCRS accelerations. | |
| | | `satellites['orbital_elements']` (np.ndarray): `(n, 6)` array of orbital elements. | |
| | | `satellites['epochs']` (list): List of `datetime` objects for TLE epochs. | |
| `fixedpoints`| `dict` | Dictionary for fixed reference points in the GCRS frame. | `targets.add_fixed_points`, `testObjects.fixedTarget` |
| | | `fixedpoints['position']` (np.ndarray): `(n, 3)` array of GCRS positions. | |
| | | `fixedpoints['exclusion']` (np.ndarray): `(n,)` array of exclusion flags. | `exclusion.update_exclusion_table` (inferred from `demo_exclusion_table.py`) |
| | | `fixedpoints['size']` (np.ndarray): `(n,)` array of object sizes. | |
| | | `fixedpoints['albedo']` (np.ndarray): `(n,)` array of object albedos. | |
| `observatories`| `dict` | Dictionary for ground-based observatories. | `observatories.add_observatories` |
| | | `observatories['position']` (np.ndarray): `(n, 3)` array of GCRS positions. | |
| | | `observatories['velocity']` (np.ndarray): `(n, 3)` array of GCRS velocities. | |
| | | `observatories['acceleration']` (np.ndarray): `(n, 3)` array of GCRS accelerations. | |
| | | `observatories['pointing']` (np.ndarray): `(n, 3)` array of pointing vectors. | |
| `detector` | `SimpleNamespace` | A namespace object containing detector parameters for all satellites/observatories. | `constellation.geos`, `constellation.geosmod`, `observatories.add_observatories`, `propagation.add_satellites_from_tle`, `detector.makeBlankDetector`, `detector.makeDetector`, `testObjects.fixedSat` |

---

## `detector` SimpleNamespace Structure

The `detector` object is a `SimpleNamespace` that contains NumPy arrays for various detector properties. Each array has a length `n`, corresponding to the number of detectors.

| Key | Type | Description | Functions Modifying This Key |
| --- | --- | --- | --- |
| `apertureArea` | `np.ndarray(float)` | Aperture area in square meters. | `detector.makeDetector` |
| `pixelArea` | `np.ndarray(float)` | Pixel area in square steradians. | `detector.makeDetector` |
| `qe` | `np.ndarray(float)` | Quantum efficiency from aperture to detector (0.0 to 1.0). | `detector.makeDetector` |
| `photoEff` | `np.ndarray(float)` | Fraction of photons in the photometry bucket. | `detector.makeDetector` |
| `pixCount` | `np.ndarray(float)` | Total number of pixels in the detector. | `detector.makeDetector` |
| `solarEx` | `np.ndarray(float)` | Solar exclusion angle in radians. | `detector.makeDetector`, `demo_sky_scan.demo_sky_scan`, `demo_exclusion_debug_print.demo_exclusion_debug_print`, `demo_exclusion_table.demo_exclusion_table`, `pointing.demo_exclusion_pointing` |
| `lunarex` | `np.ndarray(float)` | Lunar exclusion angle in radians. | `detector.makeDetector`, `demo_sky_scan.demo_sky_scan`, `demo_exclusion_debug_print.demo_exclusion_debug_print`, `demo_exclusion_table.demo_exclusion_table` |
| `earthEx` | `np.ndarray(float)` | Earth exclusion angle (above the limb) in radians. | `detector.makeDetector`, `demo_sky_scan.demo_sky_scan`, `demo_exclusion_debug_print.demo_exclusion_debug_print`, `demo_exclusion_table.demo_exclusion_table` |
| `skyBack` | `np.ndarray(float)` | Sky background in photons per square steradian. | `detector.makeDetector` |
| `zpCal` | `np.ndarray(float)` | Filter calibration zero-point (photons/s/m^2). | `detector.makeDetector` |
| `integrationTime` | `np.ndarray(float)` | Integration time required to reach a desired limiting magnitude. | `detector.makeDetector` |
| `fov` | `np.ndarray(float)` | Field of view in radians. | `detector.makeDetector`, `detector.setDetectorFOV` |
| `ifov` | `np.ndarray(float)` | Instantaneous field of view (pixel FOV) in radians. | `detector.makeDetector` |
| `filt` | `list[str]` | List of filter bands (e.g., 'V'). | `detector.makeDetector` |
| `pointing` | `np.ndarray(float)` | `(n, 3)` array of current pointing vectors for each detector. | `constellation.geos`, `pointing.update_detector_pointing`, `pointing.jerk`, `detector.detectorPointingInitialize` |
| `pointing_state` | `np.ndarray(int)` | `(2, n)` array. Row 0 is the number of points in the pointing grid; Row 1 is the current index in that grid. | `constellation.geos`, `pointing.update_detector_pointing`, `detector.detectorPointingInitialize` |

---

## `constants.py` Contents

This section summarizes the constants defined in the `constants.py` file.

| Constant | Value | Description |
| --- | --- | --- |
| `EARTH_RADIUS` | `6378137.0` | Earth's equatorial radius in meters. |
| `MOON_RADIUS` | `1737400.0` | Moon's mean radius in meters. |
| `ARCSEC` | `2.0*pi/(360*3600)` | Conversion factor from arcseconds to radians. |
| `DEGREE` | `3600 * ARCSEC` | Conversion factor from degrees to radians. |
| `ORBITAL_A_IDX` | `0` | Index for semi-major axis in orbital elements array. |
| `ORBITAL_E_IDX` | `1` | Index for eccentricity in orbital elements array. |
| `ORBITAL_I_IDX` | `2` | Index for inclination in orbital elements array. |
| `ORBITAL_RAAN_IDX` | `3` | Index for RAAN in orbital elements array. |
| `ORBITAL_ARGP_IDX` | `4` | Index for argument of perigee in orbital elements array. |
| `ORBITAL_M_IDX` | `5` | Index for mean anomaly in orbital elements array. |
| `POINTING_COUNT_IDX`| `0` | Index for the number of points in the pointing grid in the pointing state array. |
| `POINTING_PLACE_IDX`| `1` | Index for the current index in the pointing sequence in the pointing state array. |
