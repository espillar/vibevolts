# VibeVolts Gemini Documentation

## Project Overview

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets. These are intended be evolved in a discrete event simulation. 

## Data Structure

The main simulation data dictionary (`sim_data`) is initially created by `create_empty_simulation` in `simulation.py`. Other functions then add to or modify this dictionary.

### Initial `sim_data` Structure

- **`start_time`**: `datetime` - The starting time of the simulation.
- **`time`**: `datetime` - The current time of the simulation.
- **`delta_time`**: `float` - The simulation time step in seconds.
- **`counts`**: `dict` - A dictionary containing the counts of various objects in the simulation.
- **`pointing_spheres`**: `dict` - A dictionary to hold pre-computed pointing spheres.

### `sim_data` Dictionary Items and Modifying Functions

| Key | Modifying Function | File | Function Signature | Description |
|---|---|---|---|---|
| `celestial` | `add_celestial_bodies` | `celestialbodies.py` | `add_celestial_bodies(sim_data: Dict[str, Any]) -> None` | Adds celestial body structures (for Sun and Moon). |
| `celestial` | `celestial_update` | `celestialbodies.py` | `celestial_update(data_struct: Dict[str, Any], time_date: Optional[datetime] = None) -> Dict[str, Any]` | Calculates and updates the positions of the Sun and Moon. |
| `counts` | `add_fixed_points` | `targets.py` | `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None` | Adds a structure for fixed reference points in the GCRS frame. |
| `counts` | `add_celestial_bodies` | `celestialbodies.py` | `add_celestial_bodies(sim_data: Dict[str, Any]) -> None` | Adds celestial body structures (for Sun and Moon). |
| `counts` | `geos` | `constellation.py` | `geos(sim_data, n, fov) -> None` | Creates n equally spaced satellites in GEO. |
| `counts` | `geosmod` | `constellation.py` | `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None` | Creates n equally spaced satellites in GEO. |
| `counts` | `add_observatories` | `observatories.py` | `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None` | Adds observatory data structures. |
| `counts` | `add_satellites_from_tle` | `propagation.py` | `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None` | Adds and initializes a category of satellites from a TLE file. |
| `detector` | `add_satellites_from_tle` | `propagation.py` | `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None` | Adds and initializes a category of satellites from a TLE file. |
| `detector` | `add_observatories` | `observatories.py` | `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None` | Adds observatory data structures. |
| `detector` | `geos` | `constellation.py` | `geos(sim_data, n, fov) -> None` | Creates n equally spaced satellites in GEO. |
| `detector` | `geosmod` | `constellation.py` | `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None` | Creates n equally spaced satellites in GEO. |
| `detector` | `detectorPointingInitialize` | `detector.py` | `detectorPointingInitialize(sim_data, grid_points)` | Initializes the pointing state of the detectors. |
| `detector` | `update_detector_pointing` | `pointing.py` | `update_detector_pointing(sim_data: Dict[str, Any], debug: bool = False) -> None` | Updates the pointing vector for each detector. |
| `detector` | `jerk` | `pointing.py` | `jerk(sim_data: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]` | Moves the pointing vector of specific satellites. |
| `pointing_spheres` | `generate_pointing_sphere` | `pointing.py` | `generate_pointing_sphere(sim_data: Dict[str, Any], n_points: int, debug: bool = False) -> None` | Generates a pointing sphere. |
| `pointing_spheres` | `geos` | `constellation.py` | `geos(sim_data, n, fov) -> None` | Creates n equally spaced satellites in GEO. |
| `pointing_spheres` | `geosmod` | `constellation.py` | `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None` | Creates n equally spaced satellites in GEO. |
| `fixedpoints` | `add_fixed_points` | `targets.py` | `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None` | Adds a structure for fixed reference points in the GCRS frame. |
| `satellites` | `add_satellites_from_tle` | `propagation.py` | `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None` | Adds and initializes a category of satellites from a TLE file. |
| `satellites` | `geos` | `constellation.py` | `geos(sim_data, n, fov) -> None` | Creates n equally spaced satellites in GEO. |
| `satellites` | `geosmod` | `constellation.py` | `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None` | Creates n equally spaced satellites in GEO. |
| `satellites` | `propagate_satellites` | `propagation.py` | `propagate_satellites(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]` | Updates satellite positions. |
| `observatories` | `add_observatories` | `observatories.py` | `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None` | Adds observatory data structures. |

## Demos

The following demos are available in the `VibeVolts` toolkit:

- `demo_lambertian()`: Demonstrates the `lambertiansphere` brightness calculation.
- `demo_exclusion_pointing()`: Demonstrates satellite pointing with exclusion angles.

## Python Files

### `celestialbodies.py`

- `add_celestial_bodies(sim_data: Dict[str, Any]) -> None`
  > Adds celestial body structures (for Sun and Moon) to the simulation data.
- `celestial_update(data_struct: Dict[str, Any], time_date: Optional[datetime] = None) -> Dict[str, Any]`
  > Calculates and updates the positions of the Sun and Moon.

### `constellation.py`

- `geos(sim_data, n, fov) -> None`
  > Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.
- `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None`
  > Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

### `detector.py`

- `makeBlankDetector(n)`
  > Creates and returns a blank detector `SimpleNamespace`.
- `makeDetector(n, band, fov, ifov, aper, qe, photfrac, solarex, lunarex, earthex)`
  > Creates and returns a detector `SimpleNamespace` with specified parameters.
- `detectorPointingInitialize(sim_data, grid_points)`
  > Initializes the pointing state of the detectors.
- `requiredIntegrationTime(limitingMag, SNR, d, debug)`
  > Calculates the required integration time.

### `lambertian.py`

- `simple_lambertian(diameter: float, distance: float, albedo: float, angle: float, base_brightness: float) -> float`
  > Calculates the apparent brightness of a Lambertian sphere.
- `lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: np.ndarray, radius: np.ndarray, base_brightness: np.ndarray) -> np.ndarray`
  > Calculates the apparent brightness of multiple Lambertian spheres in a vectorized manner.

### `observatories.py`

- `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None`
  > Adds observatory data structures to the simulation data.

### `pointing.py`

- `generate_pointing_sphere(sim_data: Dict[str, Any], n_points: int, debug: bool = False) -> None`
  > Generates a pointing sphere with n_points and stores it in `sim_data`.
- `update_detector_pointing(sim_data: Dict[str, Any], debug: bool = False) -> None`
  > Updates the pointing vector for each detector.
- `demo_exclusion_pointing()`
  > Demonstrates satellite pointing with exclusion angles.
- `jerk(sim_data: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]`
  > Moves the pointing vector of specific satellites.

### `propagation.py`

- `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None`
  > Adds and initializes a category of satellites from a TLE file.
- `readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]`
  > Reads a TLE file and extracts orbital elements and epochs.
- `propagate_satellites(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]`
  > Updates satellite positions based on their orbital elements.

### `simulation.py`

- `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]`
  > Initializes a minimal, empty data structure for a space simulation.

### `targets.py`

- `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None`
  > Adds a structure for fixed reference points in the GCRS frame.

## Dependencies

- `numpy`
- `astropy`
- `jplephem`
- `sgp4`
- `plotly`
- `scipy`
- `ipython`