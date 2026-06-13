# VibeVolts Documentation

## Project Overview

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides tools to
initialize, propagate, and analyze the state of space-based and ground-based assets in a discrete
event simulation.

## Main Data Structure: `sim_data`

The `sim_data` dictionary is the central repository for the simulation state. It is passed between
modules to maintain and update the overall system state.

### Dictionary Items and Functions

| Item | Function(s) | Description |
| :--- | :--- | :--- |
| `start_time` | `create_empty_simulation` | Timezone-aware UTC start time (datetime). |
| `time` | `create_empty_simulation`, `nextIntegration` | Current simulation time (datetime). |
| `delta_time` | `create_empty_simulation` | Time step in seconds (float). |
| `counts` | `create_empty_simulation` | Dictionary of object counts. |
| `pointing_spheres` | `create_empty_simulation`, `generate_pointing_sphere` | Cache of pre-generated pointing vectors. |
| `observatories` | `add_observatories` | Dictionary of ground-based observatory data. |
| `detector` | `add_observatories`, `propagation.add_satellites_from_tle`, `constellation.geosmod` | `SimpleNamespace` containing detector parameters and state. |
| `satellites` | `propagation.add_satellites_from_tle`, `constellation.geosmod`, `radiometry_test.fixedSat` | Dictionary of satellite state (position, velocity, etc.). |
| `fixedpoints` | `targets.add_fixed_points`, `radiometry_test.fixedTarget` | Dictionary of fixed target/reference points. |
| `celestial` | `celestialbodies.add_celestial_bodies` | Positions of Sun and Moon. |
| `cadenceStructure` | `cadenceController.initCadence` | Schedule and grouping for detector scans. |

### Data Structure Modification Functions

#### `create_empty_simulation(start_time: datetime, delta_time: float = 60.0)`
Initializes the basic structure with `start_time`, `time`, `delta_time`, `counts`, and
`pointing_spheres`.

#### `add_observatories(sim_data: Dict[str, Any], num_observatories: int)`
Adds `observatories` and `detector` structures to `sim_data`.

#### `add_satellites_from_tle(sim_data, tle_file_path, sat_category)`
Adds satellite state arrays and a blank detector to `sim_data`.

#### `add_celestial_bodies(sim_data)`
Adds the `celestial` dictionary for Sun and Moon positions.

#### `initCadence(sim_data)`
Adds `cadenceStructure` to `sim_data` for scheduling.

#### `add_fixed_points(sim_data, num_points, size, innerRadius, outerRadius)`
Adds the `fixedpoints` structure with generated points.

## Demos

The following demos are available in the repository:

- `demo1.py`: Full demonstration of simulation tools.
- `demo2.py`: Plotting satellite and celestial positions.
- `demo3.py`: Single LEO satellite trajectory.
- `demogeo.py`: Single GEO satellite trajectory.
- `demo_fixedpoints.py`: Visualization of the fixedpoints data structure.
- `demo_pointing_plot.py`: Demonstration of pointing vector plotting.
- `demo_pointing_sequence.py`: Progression of pointing vectors over time.
- `demo_pointing_vectors.py`: Fibonacci sphere pointing vector generation.
- `demo_requiredIntegrationTime.py`: Radiometric calculation for integration time.
- `demo_show_geo_search.py`: GEO constellation sky search visualization.
- `demo_sky_scan.py`: Mapping celestial exclusion zones.
- `demo_lambertian.py`: Lambertian sphere brightness calculations.
- `demo_constellation.py`: Creation and visualization of constellations.
- `radiometry_test.py`: Setup with fixed satellites and targets.
- `pointing.py`: Solar exclusion and FOV demonstration.
- `demo_gap_time_histogram.py`: Target interobservation gap time calculations and histogram.
- `all_demos.py`: Main entry point to run all demos.

## Module Reference

### all_demos.py

#### `demo_vector_resorting_plot()`
Runs the `test_vector_resorting` function and returns its figure.

#### `run_all_demos(save_html: bool = False)`
Runs all demo functions, and either shows them inline or saves them to an HTML file.

### cadenceController.py

#### `initCadence(sim_data: dict)`
Initializes the `cadenceStructure` in `sim_data` based on detector integration times.

#### `nextIntegration(sim_data: dict, print_output: int = 0)`
Finds and performs the next scheduled integration scan, advancing simulation time.

#### `_update_next_schedule(sim_data: dict)`
Helper to find the earliest `scanNext` among all groups.

### celestialbodies.py

#### `add_celestial_bodies(sim_data: dict)`
Adds celestial body structures (Sun and Moon) to the simulation data.

#### `celestial_update(data_struct: dict, time_date: datetime)`
Calculates and updates the positions of the Sun and Moon at the specified time.

### constants.py
Contains physical constants used across the simulation (e.g., `GEO_RADIUS`, `AU`, `DEGREE`).

### constellation.py

#### `geos(sim_data: dict, n: int, fov: float)`
Adds `n` equally spaced satellites in GEO.

#### `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag)`
Adds `n` GEO satellites with detailed detector parameters.

### dataHandling.py

#### `class DataHandler`
Manages collection, Pandas conversion, export, and gap time analysis of scan results.

##### `add_results(results: dict)`
Appends a results dictionary to the collected list if detections were found.

##### `get_dataframe() -> pd.DataFrame`
Concatenates all results into a single Pandas DataFrame.

##### `save_to_csv(filename: str)`
Saves the collected results to a CSV file.

##### `save_to_parquet(filename: str)`
Saves the collected results to a Parquet file.

##### `clear()`
Resets the handler, removing all collected results.

##### `calculate_gap_times(target_id=None, pooled=False) -> dict | np.ndarray`
Calculates chronological time gaps (in seconds) between observations of targets.

##### `plot_gap_times_histogram(target_id=None, bins="auto", show_plot=True) -> go.Figure`
Generates and optionally displays a Plotly histogram of the interobservation gap times.

### detector.py

#### `setDetectorFOV(sim_data, fovSize)`
Ad-hoc function to change FOVs of all detectors to `fovSize` (radians).

#### `setDetectorIntegrationTime(sim_data, itime)`
Ad-hoc function to change integration times of all detectors to `itime` (seconds).

#### `makeBlankDetector(n: int)`
Returns a `SimpleNamespace` with empty arrays for `n` detectors.

#### `makeDetector(n, band, fov, ifov, aper, intTime=1.0, qe=0.5, photfrac=0.7, solarex=20*DEG, ...)`
Creates a detector with specified radiometric and geometric parameters.

#### `appendDetector(cd, new_cd)`
Appends the attributes of `new_cd` to the existing detector object `cd` in-place.

#### `detectorPointingInitialize(sim_data, grid_points)`
Initializes detector pointing state and adds a pointing sphere to `sim_data`.

#### `requiredIntegrationTime(limitingMag, SNR, d, debug=0)`
Calculates integration time required for a target SNR and limiting magnitude.

#### `testdetector()`
Example usage comparing results with external benchmarks.

### exclusion.py

#### `exclusion(data_struct, satellite_index, sat_category='satellites', print_debug=False)`
Checks if a satellite's view is obstructed by the Sun, Moon, or Earth.

#### `update_exclusion_table(data_struct, print_debug_for_sat=None)`
Updates the exclusion table for all satellites against all fixed points.

### fibonacciSearch.py

#### `pointing_vectors(n: int)`
Generates `n` equally spaced points on a unit sphere.

#### `resort_vectors_by_proximity(unit_vectors: np.ndarray)`
Reorders vectors to minimize angular distance between consecutive points.

#### `plot_vectors_on_sphere(vectors, title)`
Creates a 3D plot of vectors on a unit sphere.

#### `test_vector_resorting()`
Tests the vector resorting and plots distances between subsequent vectors.


### generate_log_spherical_points.py

#### `generate_log_spherical_points(num_points, inner_radius, outer_radius, seed=None)`
Generates 3D points with logarithmic radial and uniform angular distribution.

### generate_report.py

#### `generate_demo_html_report()`
Runs all plotting demos and saves the output to a single HTML file.

### lambertian.py

#### `lambertiansphere(angle_light_observer, albedo, radius, base_brightness, debug=0)`
Calculates emitted brightness from multiple Lambertian spheres.

#### `simple_lambertian(diameter, distance, albedo, angle, base_brightness)`
Scalar version of Lambertian brightness calculation.

#### `includedAngle(vectors1, vectors2)`
Calculates the angle between corresponding vectors in two arrays.

### minimalsimulation.py

#### `create_empty_simulation(start_time: datetime, delta_time: float = 60.0)`
Initializes the core `sim_data` dictionary.

### observatories.py

#### `add_observatories(sim_data: Dict[str, Any], num_observatories: int)`
Adds ground station data and detectors to the simulation.

### plot_satellite_brighness.py

#### `plot_satellite_brightness()`
Plots V-band photon flux and magnitude of satellites over a range of distances.

### plotting_3d.py

#### `plot_3d_scatter(positions, title, plot_time, labels, marker_size, trace_name, ...)`
Creates a 3D plot of object positions.

### plotting_vectors.py

#### `plot_pointing_vectors(data_struct, title, plot_time)`
Creates a 3D plot of satellites with their current pointing vectors.

### pointing.py

#### `generate_pointing_sphere(sim_data, n_points, debug=0)`
Generates and caches a set of pointing vectors on a sphere.

#### `update_detector_pointing(sim_data, sat_category='satellites', debug=False)`
Updates detector pointing vectors, skipping directions in exclusion zones.

#### `demo_exclusion_pointing()`
Demonstrates pointing with solar exclusion and FOV constraints.

#### `jerk(sim_data, satellite_indices)`
Randomly moves the pointing vector of specific satellites by 0.3 radians.

### propagation.py

#### `add_satellites_from_tle(sim_data, tle_file_path, sat_category)`
Initializes satellite structures from a TLE file.

#### `readtle(tle_file_path)`
Extracts orbital elements and epochs from a TLE file.

#### `propagate_satellites(data_struct, time_date, sat_category='satellites')`
Updates satellite positions to the specified `time_date`.

### radiometry_calcs.py

#### `fluxes(band: str)`
Returns solar, sky, and space fluxes for a given astronomical band.

#### `mag(x)`
Calculates magnitude from a linear ratio.

#### `amag(x)`
Calculates linear ratio from a magnitude.

#### `blackbody_flux(temperature, lambda_short, lambda_long)`
Computes integrated spectral radiance of a blackbody.

#### `stefan_boltzmann_law(temperature)`
Calculates total power radiated per unit area by a blackbody.

#### `plot_blackbody_spectrum(temperature)`
Plots the spectral radiance of a blackbody from 0.5 to 30 microns.

### radiometry_data.py
Contains spectral data and filter characteristics (e.g., `FILTER_DATA`).

### radiometry_test.py

#### `fixedSat(sim_data, x, y, z, fov=10*DEGREE)`
Creates a single satellite fixed at the given coordinates.

#### `fixedTarget(sim_data, size, x, y, z)`
Places a fixed target at the given coordinates.

#### `fixSun(sim_data)`
Fixes the Sun's position at 1 AU on the negative x-axis.

#### `demoFixed()`
Demonstrates setup with fixed satellites, targets, and Sun.

### scandetectors.py

#### `get_spherical_coords(arr)`
Converts Cartesian coordinates to theta and phi.

#### `scandetectors(sim_data, print_output=0, mask=None)`
Orchestrates target detection, calculating signal, noise, and SNR for visible targets.

### sim_check.py

#### `sim_check(sim_data)`
Prints a summary of the contents of a `sim_data` structure.

### simulationTemplate.py

#### `run_simulation_template()`
Basic simulation lifecycle template: Initialize -> Add Components -> Update Loop.

### targets.py

#### `add_fixed_points(sim_data, num_points, size, innerRadius, outerRadius)`
Generates and adds fixed reference targets to the simulation.

### verify_cadence.py

#### `run_verification()`
Verification script for the cadence controller.

### demo_gap_time_histogram.py

#### `demo_gap_time_histogram() -> go.Figure`
Runs a cadence simulation, calculates interobservation gaps, and returns a histogram plot.

### demo*.py (Demo Files)
Individual demonstration scripts (e.g., `demo1.py`, `demo_sky_scan.py`). Most contain a main
function named after the file (e.g., `demo1()` in `demo1.py`).

## Dependencies

- `numpy`: Numerical operations and array handling.
- `astropy`: Coordinate transformations and time handling.
- `jplephem`: Ephemeris data for celestial bodies.
- `sgp4`: Satellite propagation from TLEs.
- `plotly`: 3D visualization and interactive plots.
- `scipy`: Numerical integration and constants.
- `ipython`: Inline plot display in notebooks.
