# `sim_data` Data Structure Trace (Updated)

This document traces the construction of the `sim_data` dictionary, a central data structure in the VibeVolts simulation toolkit. It outlines which modules contribute to this structure and what data they add. The dictionary is typically passed to functions under the variable name `sim_data`.

For each module, the functions are organized into two subsections:
- **Defined Functions**: Functions defined within the module.
- **Called Functions**: Functions that are called by the functions in this module.

---

## `simulation.py`

**Description:** This module lays the foundation of the simulation data structure. It initializes the dictionary and adds fundamental simulation parameters.

### Defined Functions:
- `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]`

### Called Functions:
- None

---

## `celestialbodies.py`

**Description:** This module adds celestial bodies to the simulation.

### Defined Functions:
- `add_celestial_bodies(sim_data: Dict[str, Any]) -> None`

### Called Functions:
- None

---

## `propagation.py`

**Description:** This module is responsible for adding satellites to the simulation from Two-Line Element (TLE) sets and propagating their orbits.

### Defined Functions:
- `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None`
- `celestial_update(data_struct: Dict[str, Any], time_date: datetime) -> Dict[str, Any]`
- `readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]`
- `propagate_satellites_new(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]`

### Called Functions:
- `add_satellites_from_tle` calls:
    - `readtle` (from `propagation.py`)
    - `makeBlankDetector` (from `detector.py`)
- `celestial_update` calls:
    - `astropy.time.Time`
    - `astropy.coordinates.get_body`
    - `astropy.coordinates.GCRS`
- `readtle` calls:
    - `sgp4.api.Satrec.twoline2rv`
    - `astropy.time.Time`
- `propagate_satellites_new` calls:
    - `numpy` functions

---

## `observatories.py`

**Description:** This module adds ground-based observatories to the simulation.

### Defined Functions:
- `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None`

### Called Functions:
- `add_observatories` calls:
    - `makeBlankDetector` (from `detector.py`)

---

## `constellation.py`

**Description:** This module provides functions to create common satellite constellations. It uses functions from `propagation.py` to add satellites to the `sim_data` structure.

### Defined Functions:
- `geos(sim_data, n, fov) -> None`
- `geosmod(sim_data, n, band,fov,ifov, aper, limitingmag) -> None`

### Called Functions:
- `geos` calls:
    - `generate_pointing_sphere` (from `pointing.py`)
    - `makeBlankDetector` (from `detector.py`)
    - `propagate_satellites_new` (from `propagation.py`)
- `geosmod` calls:
    - `makeDetector` (from `detector.py`)
    - `generate_pointing_sphere` (from `pointing.py`)
    - `propagate_satellites_new` (from `propagation.py`)

---

## `pointing.py`

**Description:** This module manages the pointing control for satellites, updating their orientation in space.

### Defined Functions:
- `generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int, debug: bool = False) -> None`
- `update_satellite_pointing(data_struct: Dict[str, Any], debug: bool = False) -> None`
- `demo_exclusion_pointing()`
- `jerk(data_struct: Dict[str, Any], satellite_number: int) -> Dict[str, Any]`

### Called Functions:
- `generate_pointing_sphere` calls:
    - `resort_vectors_by_proximity` (from `fibonacciSearch.py`)
    - `pointing_vectors` (from `fibonacciSearch.py`)
- `update_satellite_pointing` calls:
    - `exclusion` (from `exclusion.py`)
- `demo_exclusion_pointing` calls:
    - `create_empty_simulation` (from `simulation.py`)
    - `add_celestial_bodies` (from `simulation.py`)
    - `add_satellites_from_tle` (from `propagation.py`)
    - `generate_pointing_sphere` (from `pointing.py`)
    - `celestial_update` (from `propagation.py`)
    - `propagate_satellites_new` (from `propagation.py`)
    - `update_satellite_pointing` (from `pointing.py`)
- `jerk` calls:
    - `numpy` functions

---

## `targets.py`

**Description:** This module is responsible for adding a set of fixed target points within the simulation space.

### Defined Functions:
- `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100) -> None`

### Called Functions:
- `add_fixed_points` calls:
    - `generate_log_spherical_points` (from `generate_log_spherical_points.py`)

---

## Modules for Calculation (No `sim_data` Modification)

The following modules contain functions that primarily perform calculations using data from `sim_data` but do not modify the structure itself. Their outputs are typically returned directly to the caller.

### `visibility.py`

#### Defined Functions:
- `is_visible(r1, r2, R_body)`

#### Called Functions:
- None

### `exclusion.py`

#### Defined Functions:
- `exclusion(data_struct: Dict[str, Any], satellite_index: int, print_debug: bool = False) -> int`
- `update_exclusion_table(data_struct: Dict[str, Any], print_debug_for_sat: Optional[int] = None) -> None`

#### Called Functions:
- `update_exclusion_table` calls:
    - `exclusion` (from `exclusion.py`)

### `lambertian.py`

#### Defined Functions:
- `simple_lambertian(diameter: float, distance: float, albedo: float, angle: float, base_brightness: float) -> float`
- `lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: float, radius: float) -> float`

#### Called Functions:
- None (uses `numpy`)

### `radiometry_calcs.py`

#### Defined Functions:
- `mag(x: float) -> float`
- `amag(x: float) -> float`
- `_planck_law(wav_m: float, temp_k: float) -> float`
- `blackbody_flux(temperature: float, lambda_short: float, lambda_long: float) -> float`
- `stefan_boltzmann_law(temperature: float) -> float`
- `plot_blackbody_spectrum(temperature: float)`
- `plot_blackbody_spectrum_visible_nir(temperature: float)`
- `sat_magnitude(size: float, range: float, angle: float, band: str) -> float`

#### Called Functions:
- `blackbody_flux` calls:
    - `scipy.integrate.quad`
- `plot_blackbody_spectrum` and `plot_blackbody_spectrum_visible_nir` call:
    - `plotly`
- `sat_magnitude` calls:
    - `FILTER_DATA` (from `radiometry_data.py`)