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
| `cislunar` | `cislunar.add_cislunar_satellites` | `CislunarState` (CR3BP + `detector`) |
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

#### `add_cislunar_satellites(sim_data, states_nd, epoch=None, sat_category='cislunar')`
Adds CR3BP cislunar satellites (`CislunarState` with embedded `detector`) to
`sim_data[sat_category]` and propagates initial GCRS positions/velocities.

#### `add_celestial_bodies(sim_data)`
Adds the `celestial` dictionary for Sun and Moon positions.

#### `initCadence(sim_data)`
Adds `cadenceStructure` to `sim_data` for scheduling.

#### `add_fixed_points(sim_data, num_points, size, innerRadius, outerRadius)`
Adds the `fixedpoints` structure with generated points.

## Demos

All consolidated demos live in `all_demos.py` (organized under `DEMO_GROUPS`), with `demoFixed`
also exposed from `radiometry_test.py`:

- **Orbits (`'orbits'`)**: `demo2`, `demo_trajectories` (combined LEO + GEO trajectories),
  `demo_constellation`, `demo_observatories_only`, `run_observatory_animation`.
- **Pointing (`'pointing'`)**: `demo_pointing_plot`, `demo_sky_scan`, `demo_pointing_sequence`,
  `demo_show_geo_search`, `demo_exclusion_pointing`.
- **Radiometry (`'radiometry'`)**: `demo_fixedpoints`, `demo_exclusion_table`, `demo_lambertian`,
  `demo_requiredIntegrationTime`, `demo_vector_resorting_plot`, `demoFixed`,
  `demo_gap_time_histogram`.

## Module Reference

### all_demos.py

Contains shared demo fixtures (`STANDARD_TLE_DATA`, `FOUR_GEO_TLE_DATA`, `get_sample_geo_tles`,
`set_exclusion_deg`, `initialize_standard_simulation`, `run_trajectory_demo`), all consolidated
demo functions across `DEMO_GROUPS`, `list_demo_groups()`, and
`run_all_demos(group='all', demos=None, save_html=False)`. To prevent browser WebGL context
limits (~16 max per page), `group='all'` defaults to `save_html=True`.

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

### cislunar.py

#### `cr3bp_eom(_t, state, mu=MU_EARTH_MOON)` / `propagate_cr3bp(state0, dt_seconds, ...)`
Dimensionless Earth-Moon CR3BP equations of motion and `DOP853` numerical propagator.

#### `emb_rotating_to_gcrs(...)` / `gcrs_to_emb_rotating(...)`
Converts between dimensionless Earth-Moon barycentric rotating coordinates and GCRS meters.

#### `add_cislunar_satellites(sim_data, states_nd, epoch=None, sat_category='cislunar')`
Populates `sim_data[sat_category]` (`CislunarState`) from CR3BP rotating states and propagates.

#### `propagate_cislunar(sim_data, time_date=None, sat_category='cislunar')`
Propagates `sim_data[sat_category]` CR3BP states and stores GCRS positions/velocities.

#### `demo_cislunar() -> Tuple[go.Figure, go.Figure]`
Propagates four cislunar orbit families (`CISLUNAR_ORBITS`) and plots EMB and GCRS views.

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
Vectorized exclusion and Earth-limb/local-horizon check.

#### `check_sim_exclusions`
```python
def check_sim_exclusions(
    sim_data, u_pointing, mask=None,
    earth_only=False, use_arcsin=False, return_angles=False
)
```
Packs observer positions, ground flags, exclusion angles, and celestial bodies from `sim_data`
and delegates to `compute_exclusions`. Shared by `exclusion`, `update_exclusion_table`, and
`scandetectors`.

#### `exclusion(data_struct, satellite_index, print_debug=False)`
Checks if a satellite or observatory detector's view is obstructed by the Sun, Moon, or Earth
(or below local horizon for observatories).

#### `update_exclusion_table(data_struct, print_debug_for_sat=None)`
Updates the exclusion table for all detectors against all fixed points.

### lambertian.py

#### `lambertiansphere(angle_light_observer, albedo, radius, base_brightness, debug=0)`
Calculates emitted brightness from multiple Lambertian spheres.

#### `simple_lambertian(diameter, distance, albedo, angle, base_brightness)`
Scalar version of Lambertian brightness calculation.

#### `includedAngle(vectors1, vectors2)`
Calculates the angle between corresponding vectors in two arrays.

### minimalsimulation.py

#### `SimulationState.add_component(key: str, state: ComponentState)`
Initializes or appends `state` into `sim_data[key]` along axis 0.

#### `SimulationState.step_to(time_date: datetime)`
Advances simulation time and propagates all active components (`satellites`, `cislunar`,
`celestial`, `observatories`) to `time_date`.

#### `SimulationState.iter_detector_components(category=None)`
Yields `(name, comp, det, offset)` across active `'satellites'`, `'cislunar'`, and
`'observatories'` detector components.

#### `create_empty_simulation(start_time: datetime, delta_time: float = 60.0)`
Initializes the core `sim_data` structure (`SimulationState`).

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

#### `plot_pointing_vectors(data_struct, title, plot_time=None)`
Creates a 3D plot of active observers with their current pointing vectors.

### pointing.py

#### `pointing_vectors(n: int)` / `resort_vectors_by_proximity(unit_vectors)`
Generates and proximity-sorts `n` Fibonacci-lattice unit vectors on a sphere.

#### `plot_vectors_on_sphere(vectors, title)` / `test_vector_resorting()`
3D visualization and distance comparison for sorted vs. unsorted pointing grids.

#### `generate_pointing_sphere(sim_data, n_points, debug=False)`
Generates and caches a set of pointing vectors on a sphere in `sim_data.pointing_spheres`.

#### `update_detector_pointing(sim_data, sat_category=None, debug=False)`
Vectorized pointing update via `check_sim_exclusions`, skipping excluded directions. If
`sat_category` is provided, updates only that component (`'satellites'`, `'cislunar'`, or
`'observatories'`).

#### `detectorPointingInitialize(sim_data, grid_points)`
Initializes `(n, 2)` `pointing_state` and `(n, 3)` `pointing` arrays for all detectors and adds a
pointing sphere to `sim_data`. Calls `update_detector_pointing` to set an initial valid direction.

### propagation.py

#### `add_satellites_from_tle(sim_data, tle_source, sat_category='satellites')`
Initializes satellite structures from a TLE file path or raw multi-line TLE string.

#### `readtle(tle_source)`
Extracts orbital elements and epochs from a TLE file path or raw multi-line TLE string.

#### `propagate_satellites(data_struct, time_date, sat_category='satellites')`
Updates satellite positions to the specified `time_date`.

### radiometry_calcs.py

Contains filter photometric constants (`FILTER_DATA`, `AU_M`, `RSUN_M`) alongside lookup helpers:

#### `fluxes(band: str)`
Returns solar, sky, and space fluxes for a given astronomical band.

#### `mag(x)`
Calculates magnitude from a linear ratio.

#### `amag(x)`
Calculates linear ratio from a magnitude.

#### `ground_sky_brightness`
```python
def ground_sky_brightness(
    zenith_vec: np.ndarray,
    pointing_vec: np.ndarray,
    sun_vec: np.ndarray,
    moon_vec: np.ndarray,
    band: str | None = None,
    return_mag: bool = False,
) -> float | dict[str, float]
```
Computes ground-based sky background brightness due to dark-sky airglow, scattered moonlight
(Krisciunas & Schaefer 1991), and daytime/twilight sunlight (Schaefer 1993; Patat et al. 2006),
scaled across wavelength (Noll et al. 2012; Jones et al. 2013) for `OPTICAL_NIR_BANDS` (`U`–`K`,
`g`–`z`). Returns `photons/s/m^2/sr` (or `mag/arcsec^2` when `return_mag=True`). Currently
standalone and not yet hooked into `scandetectors`.

#### `_unit_vec(v: np.ndarray) -> np.ndarray`
Normalizes a non-zero 3D direction vector.

#### `_airmass(cos_z: float) -> float`
Computes Krisciunas & Schaefer (1991) scattering airmass `(1 - 0.96 * sin^2 Z)^(-1/2)`.

#### `_scattered_nl(p_dot_src, x_src, mag_plus_1657, w_r, w_m, k_ext, xz) -> float`
Computes Rayleigh + Mie scattered surface brightness in nanoLamberts for Sun or Moon.

### scandetectors.py

#### `get_spherical_coords(arr)`
Converts an `(N, 3)` Cartesian array to spherical angles `(theta, phi)`.

#### `scandetectors(sim_data, print_output=0, mask=None)`
Orchestrates target detection, calculating signal, noise, and SNR for visible targets.
Observer positions are resolved uniformly via `sim_data.get_detector_positions(mask)`,
ensuring correct positions regardless of the order assets were added to `sim_data`.

### sim_report.py

#### `print_simulation_state(sim_data)`
Prints a formatted tabular summary of satellites, observatories, and exclusion parameters.

#### `sim_check(sim_data)`
Prints a diagnostic summary of the contents of a `sim_data` structure.

### targets.py

#### `generate_log_spherical_points(num_points, inner_radius, outer_radius, seed=None)`
Generates 3D points with logarithmic radial spacing and Fibonacci angular distribution.

#### `add_fixed_points(sim_data, num_points, size, innerRadius, outerRadius)`
Generates and adds fixed reference targets to the simulation.

#### `fixedSat(sim_data, x, y, z, fov=10*DEGREE)`
Adds a stationary V-band test satellite at `(x, y, z)` m pointing radially outward.

#### `fixedTarget(sim_data, size, x, y, z)` / `fixSun(sim_data)`
Helpers to add a stationary Lambertian target at `(x, y, z)` m or fix the Sun at `-1 AU`.

## Dependencies

- `numpy`: Numerical operations and array handling.
- `astropy`: Coordinate transformations and time handling.
- `jplephem`: Ephemeris data for celestial bodies.
- `sgp4`: Satellite propagation from TLEs.
- `plotly`: 3D visualization and interactive plots.
- `scipy`: Numerical integration and constants.
- `ipython`: Inline plot display in notebooks.
