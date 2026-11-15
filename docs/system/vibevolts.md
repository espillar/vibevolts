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
| `start_time` | `create_empty_simulation` | Start time (datetime) |
| `time` | `create_empty_simulation`, `nextIntegration` | Current time (datetime) |
| `delta_time` | `create_empty_simulation` | Time step in seconds |
| `counts` | `create_empty_simulation` | `CountsState` counts |
| `pointing_spheres` | `create_empty_simulation`, `generate_pointing_sphere` | Pointing cache |
| `observatories` | `add_observatories` | `ObservatoriesState` (with `detector`) |
| `satellites` | `add_satellites_from_tle`, `geosmod`, `fixedSat` | `SatellitesState` (w/ `detector`) |
| `detector` | `@property` on `SimulationState` | Aggregated `DetectorArray` accessor |
| `fixedpoints` | `add_fixed_points`, `fixedTarget` | `FixedPointsState` target points |
| `celestial` | `celestialbodies.add_celestial_bodies` | `CelestialState` Sun/Moon pos |
| `cadenceStructure` | `cadenceController.initCadence` | `CadenceState` scan schedule |

### Data Structure Modification Functions

#### `create_empty_simulation(start_time: datetime, delta_time: float = 60.0)`
Initializes the basic structure with `start_time`, `time`, `delta_time`, `counts`, and
`pointing_spheres`.

#### `add_observatories`
```python
def add_observatories(
    sim_data: Any,
    num_observatories: int,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
    altitudes: np.ndarray = None,
)
```
Adds `observatories` structure to `sim_data`, with embedded `detector` and optional geodetic
coordinates.

#### `add_satellites_from_tle(sim_data, tle_source, sat_category='satellites')`
Adds satellite state arrays and embedded `detector` to `sim_data[sat_category]` from a TLE
file path or raw multi-line TLE string.

#### `add_celestial_bodies(sim_data)`
Adds the `celestial` dictionary for Sun and Moon positions.

#### `initCadence(sim_data)`
Adds `cadenceStructure` to `sim_data` for scheduling.

#### `add_fixed_points(sim_data, num_points, size, innerRadius, outerRadius)`
Adds the `fixedpoints` structure with generated points.

## Demos

The following demos are available in the repository:

- `demo1.py`: Full demonstration of simulation tools (imported in `all_demos.py` but
  not executed by `run_all_demos`).
- `demo2.py`: Plotting satellite and celestial positions.
- `demo3.py`: Single LEO satellite trajectory.
- `demogeo.py`: Single GEO satellite trajectory.
- `demo_fixedpoints.py`: Visualization of the fixedpoints data structure.
- `demo_exclusion_table.py`: Demonstration of the exclusion table computation.
- `demo_pointing_plot.py`: Demonstration of pointing vector plotting.
- `demo_pointing_sequence.py`: Progression of pointing vectors over time.
- `demo_pointing_vectors.py`: Fibonacci sphere pointing vector generation.
- `demo_requiredIntegrationTime.py`: Radiometric calculation for integration time.
- `demo_show_geo_search.py`: GEO constellation sky search visualization.
- `demo_sky_scan.py`: Mapping celestial exclusion zones.
- `demo_lambertian.py`: Lambertian sphere brightness calculations.
- `demo_constellation.py`: Creation and visualization of constellations.
- `demo_observatories_only.py`: Demonstration using only ground-based observatories.
- `radiometry_test.py`: Setup with fixed satellites and targets (`demoFixed`).
- `pointing.py`: Solar exclusion and FOV demonstration (`demo_exclusion_pointing`).
- `demo_gap_time_histogram.py`: Target interobservation gap time calculations and histogram.
- `all_demos.py`: Main entry point to run all demos.

## Module Reference

### all_demos.py

#### `list_demo_groups()`
Lists available demo suite groups (`'orbits'`, `'pointing'`, `'radiometry'`) and their included
demo functions.

#### `demo_vector_resorting_plot()`
Runs the `test_vector_resorting` function and returns its figure.

#### `run_all_demos(group: str = 'all', demos: list = None, save_html: bool = False)`
Runs demo functions by thematic group (`'orbits'`, `'pointing'`, `'radiometry'`), specific list
of demos, or all demos. To prevent browser WebGL context limits (~16 max per page) from being
exceeded during notebook execution, `group='all'` automatically defaults to `save_html=True` unless
a specific smaller group is specified for inline display.

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
Contains physical constants and array-index constants used across the simulation:
- Radii: `EARTH_RADIUS`, `MOON_RADIUS`, `GEO_RADIUS`, `MOON_ORBIT_RADIUS` (meters)
- Angular: `ARCSEC`, `DEGREE` (radians)
- Orbital element indices: `ORBITAL_A_IDX`, `ORBITAL_E_IDX`, `ORBITAL_I_IDX`,
  `ORBITAL_RAAN_IDX`, `ORBITAL_ARGP_IDX`, `ORBITAL_M_IDX`
- Pointing state indices: `POINTING_COUNT_IDX`, `POINTING_PLACE_IDX`

### constellation.py

#### `geos(sim_data: dict, n: int, fov: float)`
Adds `n` equally spaced satellites in GEO.

#### `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag, snr=7.0)`
Adds `n` GEO satellites with detailed detector parameters. Integration time is computed
from `limitingmag` and `snr` using `requiredIntegrationTime`.

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

### demo_common.py

#### `get_sample_geo_tles(n: int = 4) -> str`
Returns the first `n` satellites (3 lines each) from `FOUR_GEO_TLE_DATA`.

#### `set_exclusion_deg(sim_data, solar=30.0, lunar=15.0, earth=10.0)`
Sets solar, lunar, and Earth exclusion angles (in degrees) on `sim_data.detector`.

#### `initialize_standard_simulation(start_time=None) -> Dict[str, Any]`
Initializes a standard simulation with `STANDARD_TLE_DATA`, celestial bodies, fixed points,
and satellite detectors, and propagates satellites to `start_time`.

#### `run_trajectory_demo(tle_data, time_steps, step_unit, unit_label, orbit_label, ...)`
Propagates a single satellite across `time_steps` and plots its 3D trajectory with Earth.

### detector.py

#### `setDetectorFOV(sim_data, fovSize)`
Ad-hoc function to change FOVs of all detectors to `fovSize` (radians).

#### `setDetectorIntegrationTime(sim_data, itime)`
Ad-hoc function to change integration times of all detectors to `itime` (seconds).

#### `makeBlankDetector(n: int)`
Returns a `DetectorArray` with empty arrays for `n` detectors.

#### `makeDetector`
```python
def makeDetector(
    n, band, fov, ifov, aper, intTime=1.0, qe=0.5, photfrac=0.7,
    solarex=20*DEGREE, lunarex=10*DEGREE, earthex=15*DEGREE
)
```
Creates a `DetectorArray` with specified radiometric and geometric parameters. Its
`pointing_state` array has shape `(n, 2)` (column 0 = grid size, column 1 = current index).

#### `appendDetector(cd, new_cd)`
Appends the attributes of `new_cd` to the existing detector object `cd` in-place.

#### `testdetector()`
Example usage comparing results with external benchmarks.

#### `requiredIntegrationTime(limitingMag, SNR, d, debug=0)`
Calculates integration time required for a target SNR and limiting magnitude.



### exclusion.py

#### `compute_exclusions`
```python
def compute_exclusions(
    obs_pos, u_pointing, is_ground, exclusion_angles,
    body_positions, body_radii, use_arcsin=False, return_angles=False
)
```
Vectorized exclusion and Earth-limb/local-horizon check shared by `exclusion`,
`update_exclusion_table`, and `scandetectors`.

#### `exclusion(data_struct, satellite_index, print_debug=False)`
Checks if a satellite or observatory detector's view is obstructed by the Sun, Moon, or Earth
(or below local horizon for observatories).

#### `update_exclusion_table(data_struct, print_debug_for_sat=None)`
Updates the exclusion table for all detectors against all fixed points.

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
Runs a **subset** of plotting demos (`demo1`, `demo2`, `demo3`, `demo_fixedpoints`,
`demo_exclusion_table`, `demo_pointing_plot`) and saves the output to `demo_plots.html`.
Not a full replacement for `all_demos.py`.

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

#### `add_observatories`
```python
def add_observatories(
    sim_data: Any,
    num_observatories: int,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
    altitudes: np.ndarray = None,
)
```
Adds ground station data and detectors to the simulation.

#### `propagate_observatories(sim_data: Any, time_date: Any)`
Updates observatory positions in the GCRS inertial frame based on Earth's sidereal rotation.

### plot_satellite_brightness.py

#### `plot_satellite_brightness()`
Plots V-band photon flux and magnitude of satellites over a range of distances.

### plotting_3d.py

#### `add_sphere_trace(fig, radius=EARTH_RADIUS, name='Earth', ...)`
Adds a 3D spherical surface trace (such as Earth or a unit sphere) to a Plotly figure, with
configurable `colorscale`, `opacity`, and `resolution`.

#### `tag_demo_figure(fig: go.Figure, demo_name: str) -> go.Figure`
Adds a standard `"Generated by: <demo_name>"` subtitle annotation to a Plotly figure.

#### `apply_3d_scene_layout(fig, title, axis_unit=' (m)', legend_title=None, **extra_layout)`
Applies standard 3D scene axis titles, `aspectmode='data'`, and compact margins to a figure.

#### `add_vector_traces(fig, origins, vectors, scale=1.0, color='red', width=3, ...)`
Adds 3D line segments from `origins` along `vectors` (scaled by `scale`) to a Plotly figure.

#### `plot_3d_scatter(positions, title, plot_time, labels, marker_size, trace_name, ...)`
Creates a 3D plot of object positions.

### plotting_vectors.py

#### `plot_pointing_vectors(data_struct, title, plot_time)`
Creates a 3D plot of satellites with their current pointing vectors.

### pointing.py

#### `generate_pointing_sphere(sim_data, n_points, debug=False)`
Generates and caches a set of pointing vectors on a sphere in `sim_data.pointing_spheres`.

#### `update_detector_pointing(sim_data, sat_category=None, debug=False)`
Updates detector pointing vectors, skipping directions in exclusion zones. If `sat_category` is
provided, updates only that component (`'satellites'` or `'observatories'`).

#### `detectorPointingInitialize(sim_data, grid_points)`
Initializes `(n, 2)` `pointing_state` and `(n, 3)` `pointing` arrays for all detectors and adds a
pointing sphere to `sim_data`. Calls `update_detector_pointing` to set an initial valid direction.

#### `demo_exclusion_pointing()`
Demonstrates pointing with solar exclusion and FOV constraints.

### propagation.py

#### `add_satellites_from_tle(sim_data, tle_source, sat_category='satellites')`
Initializes satellite structures from a TLE file path or raw multi-line TLE string.

#### `readtle(tle_source)`
Extracts orbital elements and epochs from a TLE file path or raw multi-line TLE string.

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
Creates a single satellite fixed at `(x, y, z)` in meters. `fov` sets the detector
field of view in radians. Appends to existing satellite data if any is present.

#### `fixedTarget(sim_data, size, x, y, z)`
Places a fixed Lambertian sphere target at `(x, y, z)` with the given `size` (radius,
meters) and a default albedo of 0.2.

#### `fixSun(sim_data)`
Fixes the Sun's position at 1 AU on the negative x-axis.

#### `demoFixed()`
Demonstrates setup with fixed satellites, targets, and Sun.

### scandetectors.py

#### `get_spherical_coords(arr)`
Converts an `(N, 3)` Cartesian array to spherical angles `(theta, phi)`.

#### `scandetectors(sim_data, print_output=0, mask=None)`
Orchestrates target detection, calculating signal, noise, and SNR for visible targets.
Observer positions are resolved uniformly via `sim_data.get_detector_positions(mask)`,
ensuring correct positions regardless of the order assets were added to `sim_data`.

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
