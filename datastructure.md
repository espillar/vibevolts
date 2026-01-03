# VibeVolts Data Structures

This document outlines the main data structures, functions, and dependencies of the VibeVolts simulation toolkit.

## Dependencies

VibeVolts requires the following Python libraries:

*   `numpy`
*   `astropy`
*   `jplephem`
*   `sgp4`
*   `plotly`
*   `scipy`
*   `ipython`

You can install them using pip:

```bash
pip install numpy astropy jplephem sgp4 plotly scipy ipython
```

## Data Structure (`sim_data`)

The core of the simulation is a dictionary named `sim_data` that holds the entire state of the simulation.

| Key | Type | Description | Modified By |
| --- | --- | --- | --- |
| `start_time` | `datetime` | The starting time of the simulation. | `simulation.create_empty_simulation` |
| `time` | `datetime` | The current time of the simulation. | `simulation.create_empty_simulation`, `celestialbodies.celestial_update`, `propagation.propagate_satellites`, `demo_constellation.demo_constellation`, `demo1.demo1`, `demo3.demo3` |
| `delta_time`| `float` | The time step for the simulation in seconds. | `simulation.create_empty_simulation` |
| `counts` | `dict` | A dictionary containing the counts of various objects. | `simulation.create_empty_simulation`, `celestialbodies.add_celestial_bodies`, `constellation.geos`, `constellation.geosmod`, `observatories.add_observatories`, `propagation.add_satellites_from_tle`, `targets.add_fixed_points` |
| `pointing_spheres` | `dict` | A dictionary of pre-calculated pointing spheres. | `simulation.create_empty_simulation`, `pointing.generate_pointing_sphere` |
| `celestial` | `dict` | Data for celestial bodies (Sun, Moon). | `celestialbodies.add_celestial_bodies`, `celestialbodies.celestial_update` |
| `satellites`| `dict` | Data for satellites. | `constellation.geos`, `constellation.geosmod`, `propagation.add_satellites_from_tle`, `propagation.propagate_satellites` |
| `detector` | `SimpleNamespace` | Detector parameters. | `constellation.geos`, `constellation.geosmod`, `observatories.add_observatories`, `propagation.add_satellites_from_tle`, `pointing.update_detector_pointing`, `pointing.jerk`, `detector.makeBlankDetector`, `detector.makeDetector`, `detector.detectorPointingInitialize` |
| `fixedpoints`| `dict` | Fixed reference points. | `targets.add_fixed_points` |
| `observatories`| `dict` | Observatory data. | `observatories.add_observatories` |

## Demos

The `all_demos.py` script runs the following demonstrations:

*   `demo2`
*   `demo3`
*   `demo4`
*   `demo_fixedpoints`
*   `demo_pointing_plot`
*   `demo_lambertian`
*   `demo_sky_scan`
*   `demo_pointing_vectors`
*   `demo_pointing_sequence`
*   `demo_constellation`
*   `demo_show_geo_search`
*   `demo_requiredIntegrationTime`
*   `demo_vector_resorting_plot`
*   `demo_exclusion_pointing`

## Python Files

### `all_demos.py`

#### `demo_vector_resorting_plot() -> go.Figure`
Runs the test_vector_resorting function and returns its figure.

#### `run_all_demos(save_html=False)`
Runs all demo functions, and either shows them inline or saves them to a single HTML file.

### `celestialbodies.py`

#### `add_celestial_bodies(sim_data: Dict[str, Any]) -> None`
Adds celestial body structures (for Sun and Moon) to the simulation data.

#### `celestial_update(data_struct: Dict[str, Any], time_date: Optional[datetime] = None) -> Dict[str, Any]`
Calculates and updates the positions of the Sun and Moon.

### `constants.py`
This file contains global constants for array indices and physical constants.

### `constellation.py`

#### `geos(sim_data, n,  fov) -> None`
Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

#### `geosmod(sim_data, n, band,fov,ifov, aper, limitingmag) -> None`
Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

### `demo_common.py`

#### `initialize_standard_simulation(start_time=None) -> Dict[str, Any]`
Initializes a standard simulation with a predefined set of satellites.

### `demo_constellation.py`

#### `demo_constellation() -> go.Figure`
Runs a demonstration of the constellation creation tools.

### `demo_exclusion_debug_print.py`

#### `demo_exclusion_debug_print()`
Demonstrates the debug printing feature of the exclusion function.

### `demo_exclusion_table.py`

#### `demo_exclusion_table() -> go.Figure`
Demonstrates the creation and visualization of the exclusion table.

### `demo_fixedpoints.py`

#### `demo_fixedpoints() -> go.Figure`
Demonstrates the fixedpoints data structure by plotting it in 3D.

### `demo_lambertian.py`

#### `demo_lambertian()`
Runs a demonstration of the lambertiansphere function, including example calculations and a plot.

### `demo_pointing_plot.py`

#### `demo_pointing_plot() -> go.Figure`
Demonstrates the plot_pointing_vectors function.

### `demo_pointing_sequence.py`

#### `demo_pointing_sequence() -> go.Figure`
Demonstrates the satellite pointing sequence functionality.

### `demo_pointing_vectors.py`

#### `demo_pointing_vectors() -> go.Figure`
Demonstrates the generation and plotting of pointing vectors.

### `demo_requiredIntegrationTime.py`

#### `demo_requiredIntegrationTime()`
Demonstrates the requiredIntegrationTime function.

### `demo_show_geo_search.py`

#### `demo_show_geo_search()`
This demo initializes a simulation, adds a GEO constellation, and then generates several plots to visualize the satellite pointing updates and the RA/Dec history of one satellite.

### `demo_sky_scan.py`

#### `demo_sky_scan() -> go.Figure`
Performs a sky scan from a GEO satellite to map celestial exclusion zones.

### `demo1.py`

#### `demo1() -> go.Figure`
Runs a full demonstration of the simulation tools.

### `demo2.py`

#### `demo2() -> go.Figure`
Runs a demonstration plotting satellite and celestial positions.

### `demo3.py`

#### `demo3() -> go.Figure`
Runs a demonstration plotting a single LEO satellite trajectory.

### `demo4.py`

#### `demo4() -> go.Figure`
Runs a demonstration plotting a single GEO satellite trajectory.

### `detector.py`

#### `makeBlankDetector(n)`
makeBlankDetector makes and returns a detector SipleNamespace with parameters

#### `makeDetector(n, band, fov, ifov, aper, qe = 0.5, photfrac=0.7,      solarex= 20.0 * DEGREE,   lunarex= 10.0 * DEGREE,  earthex= 15.0 * DEGREE)`
makeDetector takes parameters of a sensor and stuffs a filter array and a detector array, which it returns.

#### `detectorPointingInitialize(sim_data, grid_points)`
We assume that sim_data['detectors'] loaded, but the pointing part of detectors is currently empty.

#### `requiredIntegrationTime(limitingMag, SNR, d, debug = 0)`
requiredIntegrationTime(limitingMag, d) takes a two dimensional detector array ("detect")and calculates all the integration tiemes and returns those as a vector.

#### `testdetector()`
testdetector creates an example that can be compared with some of the stuff in Curio.

### `exclusion.py`

#### `exclusion(data_struct: Dict[str, Any], satellite_index: int, print_debug: bool = False) -> int`
Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.

### `fibonacciSearch.py`

#### `pointing_vectors(n: int) -> np.ndarray`
Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.

#### `resort_vectors_by_proximity(unit_vectors: np.ndarray) -> np.ndarray`
Resorts a list of vectors by making each subsequent vector the closest one in the remaining set to the previous one.

#### `plot_vectors_on_sphere(vectors: np.ndarray, title: str) -> go.Figure`
Creates a 3D plot of vectors on a sphere.

#### `test_vector_resorting()`
Tests the vector resorting and plots the Euclidean distance between subsequent vectors.

### `fluxes.py`

#### `fluxes(band)`
uses the FILTER_DATA table from radiometry_data.py for data Looks up in formation based on the argument band, which is usually something like an astronomical band... U, B, V, etc.

### `generate_log_spherical_points.py`

#### `generate_log_spherical_points(num_points: int, inner_radius: float, outer_radius: float, seed: int = None) -> tuple[np.ndarray, np.ndarray]`
Generates 3D points with logarithmic radial and uniform angular distribution.

### `generate_report.py`

#### `generate_demo_html_report()`
Runs all plotting demos and saves the output to a single HTML file.

### `lambertian.py`

#### `simple_lambertian(diameter: float, distance: float, albedo: float, angle: float, base_brightness: float) -> float`
Calculates the apparent brightness of a Lambertian sphere.

#### `lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: np.ndarray, radius: np.ndarray, base_brightness: np.ndarray) -> np.ndarray`
Calculates the apparent brightness of multiple Lambertian spheres in a vectorized manner.

### `observatories.py`

#### `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None`
Adds observatory data structures to the simulation data.

### `plot_satellite_brighness.py`

#### `plot_satellite_brightness()`
Plots the apparent V-band photon flux and magnitude of satellites with various diameters over a range of distances.

### `plotting_3d.py`

#### `plot_3d_scatter(positions: np.ndarray, title: str, plot_time: datetime, labels: Optional[List[str]] = None, marker_size: int = 1, trace_name: str = 'Points') -> go.Figure`
Creates a 3D plot of object positions.

### `plotting_vectors.py`

#### `plot_pointing_vectors(data_struct: Dict[str, Any], title: str, plot_time: datetime) -> go.Figure`
Creates a 3D plot of satellites with pointing vectors.

### `pointing.py`

#### `generate_pointing_sphere(sim_data: Dict[str, Any], n_points: int, debug: bool = False) -> None`
Generates a pointing sphere with n_points and stores it in the sim_data['pointing_sphers'][n_points] A pointing sphere is a 3 by n_points numpy array with the 3 representing unit vectors to be pointed to These positions will be used by the update_satellte_pointing to point sensors incrementaly.

#### `update_detector_pointing(sim_data: Dict[str, Any], debug: bool = False) -> None`
Updates the pointing vector for each detector, skipping excluded pointing directions.

#### `demo_exclusion_pointing()`
Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere.

#### `jerk(sim_data: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]`
Moves the pointing vector of specific satellites by 0.3 radians in a random direction.

### `propagation.py`

#### `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None`
Adds and initializes a category of satellites from a TLE file.

#### `readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]`
Reads a TLE file and extracts orbital elements and epochs for each satellite.

#### `propagate_satellites(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]`
data_struct is the "standard" structure for the simulation time_date - a datetime - is the time that the satellites are propagated to.

### `radiometry_calcs.py`

#### `fluxes(band)`
uses the FILTER_DATA table from radiometry_data.py for data Looks up in formation based on the argument band, which is usually something like an astronomical band... U, B, V, etc.

#### `mag(x: float) -> float`
Calculates a magnitude value from a linear ratio.

#### `amag(x: float) -> float`
Calculates the linear ratio from a magnitude value.

#### `blackbody_flux(temperature: float, lambda_short: float, lambda_long: float) -> float`
Numerically computes the integrated spectral radiance of a blackbody over a given wavelength band.

#### `stefan_boltzmann_law(temperature: float) -> float`
Calculates the total power radiated per unit area by a blackbody using the Stefan-Boltzmann law.

#### `plot_blackbody_spectrum(temperature: float)`
Plots the spectral radiance of a blackbody from 0.5 to 30 microns.

#### `plot_blackbody_spectrum_visible_nir(temperature: float)`
Plots the spectral radiance of a blackbody from 0.1 to 1 micron.

#### `sat_magnitude(size: float, range: float, angle: float, band: str) -> float`
given a satellite size and a waveband and range pull the brightness of the sun and the calibration from radiometry_data

### `radiometry_data.py`
This file contains radiometric data for standard astronomical filters.

### `scandetectors.py`

#### `get_spherical_coords(arr)`
given a n by 3 d array, return two 1D arrays with the theta and phi angles in radians

#### `scandetectors(sim_data: dict)`
Scans for and processes detector data within the simulation.

#### `findVectorMask(values: np.ndarray, floorValue: float) -> np.ndarray`
Compares values in a 1D numpy array to a floorValue and returns a boolean mask.

### `sim_check.py`

#### `sim_check(sim_data)`
Prints a brief summary of what's present in a sim_data structure.

### `simulation.py`

#### `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]`
Initializes a minimal, empty data structure for a space simulation.

### `targets.py`

#### `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None`
Adds a structure for fixed reference points in the GCRS frame.
