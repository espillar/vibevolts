# VibeVolts `sim_data` Dictionary Structure

This document details the structure of the main `sim_data` dictionary used throughout the
VibeVolts simulation toolkit.

## Top-Level `sim_data` Keys

*   **`start_time`** (`datetime`)
    *   *Description*: The starting time and date of the simulation in UTC.
    *   *Modified by*:
        `minimalsimulation.create_empty_simulation`
*   **`time`** (`datetime`)
    *   *Description*: The current timezone-aware execution time of the simulation.
    *   *Modified by*:
        `minimalsimulation.create_empty_simulation`, `celestialbodies.celestial_update`,
        `propagation.propagate_satellites`, `cadenceController.nextIntegration`
*   **`delta_time`** (`float`)
    *   *Description*: The default time step size for the simulation in seconds.
    *   *Modified by*:
        `minimalsimulation.create_empty_simulation`
*   **`counts`** (`dict`)
    *   *Description*: A tracker dictionary containing counts of active simulation assets.
    *   *Keys*:
        *   `celestial` (`int`): Number of celestial bodies. Modified by
            `celestialbodies.add_celestial_bodies`.
        *   `satellites` (`int`): Number of satellites. Modified by:
            `constellation.geos`, `constellation.geosmod`, `propagation.add_satellites_from_tle`,
            `radiometry_test.fixedSat`.
        *   `observatories` (`int`): Number of observatories. Modified by
            `observatories.add_observatories`.
        *   `fixedpoints` (`int`): Number of fixed points. Modified by:
            `targets.add_fixed_points`, `radiometry_test.fixedTarget`.
    *   *Modified by*:
        `minimalsimulation.create_empty_simulation` (initializes)
*   **`pointing_spheres`** (`dict`)
    *   *Description*: Dictionary storing cache arrays for spherical pointing grids.
    *   *Modified by*:
        `minimalsimulation.create_empty_simulation` (initializes),
        `pointing.generate_pointing_sphere`
*   **`celestial`** (`dict`)
    *   *Description*: Holds GCRS vector properties for the Sun and Moon.
    *   *Sub-keys*:
        *   `position` (`np.ndarray` of shape `(2, 3)`): Positions in meters.
            Modified by `celestialbodies.celestial_update`, `radiometry_test.fixSun`.
        *   `velocity` (`np.ndarray` of shape `(2, 3)`): Velocity vectors in m/s.
        *   `acceleration` (`np.ndarray` of shape `(2, 3)`): Acceleration vectors in m/s^2.
    *   *Modified by*:
        `celestialbodies.add_celestial_bodies` (initializes)
*   **`satellites`** (`dict`)
    *   *Description*: Holds orbital parameters, GCRS vectors, and epochs for satellite assets.
    *   *Sub-keys*:
        *   `position` (`np.ndarray` of shape `(n, 3)`): Current GCRS coordinates in meters.
            Modified by `propagation.propagate_satellites`.
        *   `velocity` (`np.ndarray` of shape `(n, 3)`): Current GCRS velocities.
        *   `acceleration` (`np.ndarray` of shape `(n, 3)`): Current GCRS accelerations.
        *   `orbital_elements` (`np.ndarray` of shape `(n, 6)`): Orbital elements in
            canonical order.
        *   `epochs` (`list` of `datetime`): Epoch times for orbit calculation.
    *   *Modified by*:
        `constellation.geos`, `constellation.geosmod`,
        `propagation.add_satellites_from_tle`, `radiometry_test.fixedSat`
*   **`fixedpoints`** (`dict`)
    *   *Description*: Holds stationary reference targets located in the GCRS frame.
    *   *Sub-keys*:
        *   `position` (`np.ndarray` of shape `(n, 3)`): Position vectors in meters.
        *   `exclusion` (`np.ndarray` of shape `(n,)`): Obstruction and visibility flags.
            Modified by `exclusion.update_exclusion_table`.
        *   `size` (`np.ndarray` of shape `(n,)`): Target diameters in meters.
        *   `albedo` (`np.ndarray` of shape `(n,)`): Reflective albedo values (0.0 to 1.0).
    *   *Modified by*:
        `targets.add_fixed_points`, `radiometry_test.fixedTarget`
*   **`observatories`** (`dict`)
    *   *Description*: Holds GCRS vector arrays for ground-based observation stations.
    *   *Sub-keys*:
        *   `position` (`np.ndarray` of shape `(n, 3)`): Ground GCRS coordinates.
        *   `velocity` (`np.ndarray` of shape `(n, 3)`): Station velocities in m/s.
        *   `acceleration` (`np.ndarray` of shape `(n, 3)`): Station accelerations in m/s^2.
        *   `pointing` (`np.ndarray` of shape `(n, 3)`): Main pointing vectors.
    *   *Modified by*:
        `observatories.add_observatories`
*   **`detector`** (`SimpleNamespace`)
    *   *Description*: Properties and pointing states for active optoelectronic sensors.
    *   *Modified by*:
        `constellation.geos`, `constellation.geosmod`, `observatories.add_observatories`,
        `propagation.add_satellites_from_tle`, `detector.makeBlankDetector`,
        `detector.makeDetector`, `radiometry_test.fixedSat`

---

## `detector` SimpleNamespace Structure

The `detector` object is a `SimpleNamespace` containing vectorized parameters of length `n`,
where each index corresponds to a specific active sensor:

*   **`apertureArea`** (`np.ndarray(float)`): Aperture area in square meters.
    *   *Modified by*: `detector.makeDetector`
*   **`pixelOmega`** (`np.ndarray(float)`): Angular pixel size in square steradians.
    *   *Modified by*: `detector.makeDetector`
*   **`qe`** (`np.ndarray(float)`): Quantum efficiency from entrance to focal plane.
    *   *Modified by*: `detector.makeDetector`
*   **`photoEff`** (`np.ndarray(float)`): Fraction of target light in photometry bucket.
    *   *Modified by*: `detector.makeDetector`
*   **`pixCount`** (`np.ndarray(float)`): Total pixels per sensor.
    *   *Modified by*: `detector.makeDetector`
*   **`solarEx`** (`np.ndarray(float)`): Solar exclusion angle in radians.
    *   *Modified by*: `detector.makeDetector`
*   **`lunarex`** (`np.ndarray(float)`): Lunar exclusion angle in radians.
    *   *Modified by*: `detector.makeDetector`
*   **`earthEx`** (`np.ndarray(float)`): Earth limb exclusion angle in radians.
    *   *Modified by*: `detector.makeDetector`
*   **`skyBack`** (`np.ndarray(float)`): Background sky brightness in photons/s/m^2/steradian.
    *   *Modified by*: `detector.makeDetector`
*   **`zpCal`** (`np.ndarray(float)`): Calibration zero-point (photons/s/m^2).
    *   *Modified by*: `detector.makeDetector`
*   **`integrationTime`** (`np.ndarray(float)`): Vectorized integration time per sensor.
    *   *Modified by*: `detector.makeDetector`
*   **`fov`** (`np.ndarray(float)`): Field of view diameter in radians.
    *   *Modified by*: `detector.makeDetector`, `detector.setDetectorFOV`
*   **`ifov`** (`np.ndarray(float)`): Instantaneous pixel field of view in radians.
    *   *Modified by*: `detector.makeDetector`
*   **`filt`** (`list[str]`): List of active filters (e.g., `'V'`).
    *   *Modified by*: `detector.makeDetector`
*   **`pointing`** (`np.ndarray(float)` of shape `(n, 3)`): Unit vector pointing directions.
    *   *Modified by*:
        `constellation.geos`, `pointing.update_detector_pointing`, `pointing.jerk`,
        `detector.detectorPointingInitialize`
*   **`pointing_state`** (`np.ndarray(int)` of shape `(2, n)`): Re-slew grids state. Row 0
    denotes total steps; Row 1 denotes current index.
    *   *Modified by*:
        `constellation.geos`, `pointing.update_detector_pointing`,
        `detector.detectorPointingInitialize`

---

## `constants.py` Constants

The physical and indexing constants defined within `constants.py`:

*   **`EARTH_RADIUS`** (`6378137.0`): Earth's equatorial radius in meters.
*   **`MOON_RADIUS`** (`1737400.0`): Moon's mean radius in meters.
*   **`ARCSEC`** (`2.0 * pi / (360 * 3600)`): Conversion factor from arcseconds to radians.
*   **`DEGREE`** (`3600 * ARCSEC`): Conversion factor from degrees to radians.
*   **`ORBITAL_A_IDX`** (`0`): Semi-major axis index.
*   **`ORBITAL_E_IDX`** (`1`): Eccentricity index.
*   **`ORBITAL_I_IDX`** (`2`): Inclination index.
*   **`ORBITAL_RAAN_IDX`** (`3`): Right Ascension of Ascending Node index.
*   **`ORBITAL_ARGP_IDX`** (`4`): Argument of perigee index.
*   **`ORBITAL_M_IDX`** (`5`): Mean anomaly index.
*   **`POINTING_COUNT_IDX`** (`0`): Pointing sequence steps capacity index.
*   **`POINTING_PLACE_IDX`** (`1`): Current pointing step index.
