# `sim_data` Dictionary Structure

This document outlines the structure of the `sim_data` dictionary used in the VibeVolts simulation toolkit.

## Initial Structure

The `sim_data` dictionary is created by `create_empty_simulation` in `simulation.py`.

- `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]`
  > Initializes a minimal, empty data structure for a space simulation.

  - **`start_time`**: `datetime` - The starting time of the simulation.
  - **`time`**: `datetime` - The current time of the simulation, initialized to `start_time`.
  - **`delta_time`**: `float` - The time step for the simulation in seconds.
  - **`counts`**: `dict` - A dictionary to hold counts of various simulation objects.
  - **`pointing_spheres`**: `dict` - A dictionary to hold pre-computed pointing spheres.

---

## `celestial`

Added by `add_celestial_bodies` in `celestialbodies.py`.

- `add_celestial_bodies(sim_data: Dict[str, Any]) -> None`
  > Adds celestial body structures (for Sun and Moon) to the simulation data.

- `celestial_update(data_struct: Dict[str, Any], time_date: Optional[datetime] = None) -> Dict[str, Any]`
  > Calculates and updates the positions of the Sun and Moon.

- **`sim_data['counts']['celestial']`**: `int` - The number of celestial bodies (2 for Sun and Moon).
- **`sim_data['celestial']`**: `dict`
  - **`position`**: `np.ndarray` (2, 3) - Position vectors (x, y, z) in meters in GCRS.
  - **`velocity`**: `np.ndarray` (2, 3) - Velocity vectors in m/s.
  - **`acceleration`**: `np.ndarray` (2, 3) - Acceleration vectors in m/s^2.

---

## `fixedpoints`

Added by `add_fixed_points` in `targets.py`.

- `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None`
  > Adds a structure for fixed reference points in the GCRS frame.

- **`sim_data['counts']['fixedpoints']`**: `int` - The number of fixed points.
- **`sim_data['fixedpoints']`**: `dict`
  - **`position`**: `np.ndarray` (num_points, 3) - Position vectors of fixed points.
  - **`exclusion`**: `np.ndarray` (num_points,) - Exclusion flag for each fixed point.
  - **`size`**: `np.ndarray` (num_points,) - Size of each fixed point.

---

## `satellites` (and other categories)

Added by `add_satellites_from_tle` in `propagation.py`, and `geos` and `geosmod` in `constellation.py`.

- `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None`
  > Adds and initializes a category of satellites from a TLE file.
- `geos(sim_data, n, fov) -> None`
  > Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.
- `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None`
  > Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.
- `propagate_satellites(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]`
  > Updates satellite positions and pointing vectors based on their orbital elements.

- **`sim_data['counts'][sat_category]`**: `int` - The number of satellites in the category.
- **`sim_data[sat_category]`**: `dict`
  - **`position`**: `np.ndarray` (n, 3) - Position vectors in meters.
  - **`velocity`**: `np.ndarray` (n, 3) - Velocity vectors in m/s.
  - **`acceleration`**: `np.ndarray` (n, 3) - Acceleration vectors in m/s^2.
  - **`orbital_elements`**: `np.ndarray` (n, 6) - Keplerian orbital elements.
  - **`epochs`**: `list[datetime]` - Epoch for each satellite's orbital elements.
  - **`pointing`**: `np.ndarray` (n, 3) - Pointing direction vector.
  - **`pointing_state`**: `np.ndarray` (n, 2) - State of the pointing sequence for each satellite.

---

## `observatories`

Added by `add_observatories` in `observatories.py`.

- `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None`
  > Adds observatory data structures to the simulation data.

- **`sim_data['counts']['observatories']`**: `int` - The number of observatories.
- **`sim_data['observatories']`**: `dict`
  - **`position`**: `np.ndarray` (n, 3) - Position vectors in meters.
  - **`velocity`**: `np.ndarray` (n, 3) - Velocity vectors in m/s.
  - **`acceleration`**: `np.ndarray` (n, 3) - Acceleration vectors in m/s^2.
  - **`pointing`**: `np.ndarray` (n, 3) - Pointing direction vector.

---

## `detector`

The `detector` object is a `SimpleNamespace` that holds detector properties. It is created by `makeBlankDetector` or `makeDetector` and assigned to `sim_data['detector']` by various functions (`add_satellites_from_tle`, `add_observatories`, `geos`, `geosmod`).

- **`aperture`**: `np.ndarray` (n,) - Aperture area in square meters.
- **`pixelArea`**: `np.ndarray` (n,) - Pixel area in square arcsec.
- **`qe`**: `np.ndarray` (n,) - Quantum efficiency.
- **`photoEff`**: `np.ndarray` (n,) - Fraction of photons in photometry bucket.
- **`pixCount`**: `np.ndarray` (n,) - Total number of pixels.
- **`solarEx`**: `np.ndarray` (n,) - Solar exclusion angle in radians.
- **`lunarex`**: `np.ndarray` (n,) - Lunar exclusion angle in radians.
- **`earthEx`**: `np.ndarray` (n,) - Earth exclusion angle in radians.
- **`skyBack`**: `np.ndarray` (n,) - Sky background in photons per square steradian.
- **`zpCal`**: `np.ndarray` (n,) - Filter calibration zeropoint.
- **`itime`**: `np.ndarray` (n,) - Integration time.
- **`fov`**: `np.ndarray` (n,) - Field of view.
- **`ifov`**: `np.ndarray` (n,) - Instantaneous field of view.
- **`filt`**: `list[str]` (n) - Filter name.
- **`pointing`**: `np.ndarray` (n, 3) - Pointing direction vector.
- **`pointing_state`**: `np.ndarray` (2, n) - State of the pointing sequence.

---

## `pointing_spheres`

Modified by `generate_pointing_sphere` in `pointing.py`.

- `generate_pointing_sphere(sim_data: Dict[str, Any], n_points: int, debug: bool = False) -> None`
  > Generates a pointing sphere with n_points and stores it in the sim_data['pointing_sphers'][n_points]

- **`sim_data['pointing_spheres'][n_points]`**: `np.ndarray` (n_points, 3) - An array of unit vectors.

---

## Radiometry Data (`radiometry_data.py`)

This file contains physical constants and data for standard astronomical filters.

### Physical Constants
- **`AU_M`**: Astronomical Unit in meters.
- **`RSUN_M`**: Radius of the Sun in meters.

### Radiometric Data (`FILTER_DATA`)
A dictionary containing data for standard astronomical filters. Each filter has the following keys:
- **`sun`**: Apparent magnitude of the sun in the filter band.
- **`sky`**: Sky brightness in magnitudes per square arcsecond.
- **`space`**: Sky brightness for space-based telescopes.
- **`central_wavelength`**: Central wavelength in nanometers.
- **`bandwidth`**: Bandwidth in nanometers.
- **`zero_point`**: Photon flux for a 0-magnitude object in photons per second per square meter.

The filters are categorized as:
- **Johnson-Cousins UBVRI Filters**: 'U', 'B', 'V', 'R', 'I'
- **Near-Infrared JHK Filters**: 'J', 'H', 'K'
- **SDSS Filters**: 'g', 'r', 'i', 'z'
- **Ground-Based Mid-IR Filters**: 'L', 'M', 'N'
- **JWST MIRI Filters**: 'F560W', 'F770W', 'F1000W', 'F1500W', 'F2550W'