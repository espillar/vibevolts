# VibeVolts Documentation

## Project Overview

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of
tools to initialize, propagate, and analyze the state of various space-based and ground-based
assets. These are intended be evolved in a discrete event simulation.

## Data Structure

The current state of the simulation is stored in a dictionary typically called `sim_data` which is
passed between the routines that initialize and operate on the components to initialize, evolve,
and interrogate the overall system. The different components of the system are typically dealt
with by different modules.

### `sim_data` dictionary structure:

*   `start_time`: `datetime`
    *   `minimalsimulation.create_empty_simulation`: Initializes.
*   `time`: `datetime`
    *   `minimalsimulation.create_empty_simulation`: Initializes.
    *   `celestialbodies.celestial_update`: Updates.
    *   `propagation.propagate_satellites`: Updates.
    *   `cadenceController.nextIntegration`: Advanced to the next scheduled event time.
*   `delta_time`: `float`
    *   `minimalsimulation.create_empty_simulation`: Initializes.
*   `counts`: `Dict[str, int]`
    *   `minimalsimulation.create_empty_simulation`: Initializes.
    *   `celestialbodies.add_celestial_bodies`: Adds `'celestial'`.
    *   `constellation.geos`: Adds `'satellites'`.
    *   `constellation.geosmod`: Adds `'satellites'`.
    *   `propagation.add_satellites_from_tle`: Adds `sat_category`.
    *   `observatories.add_observatories`: Adds `'observatories'`.
    *   `targets.add_fixed_points`: Adds `'fixedpoints'`.
    *   `radiometry_test.fixedSat`: Adds/increments `'satellites'`.
    *   `radiometry_test.fixedTarget`: Adds/increments `'fixedpoints'`.
*   `pointing_spheres`: `Dict[int, np.ndarray]`
    *   `minimalsimulation.create_empty_simulation`: Initializes.
    *   `pointing.generate_pointing_sphere`: Adds a new pointing sphere.
*   `celestial`: `Dict[str, np.ndarray]`
    *   `celestialbodies.add_celestial_bodies`: Initializes `position`, `velocity`, `acceleration`.
    *   `celestialbodies.celestial_update`: Updates `position`.
    *   `radiometry_test.fixSun`: Updates `position` of the sun.
*   `satellites`: `Dict[str, any]`
    *   `constellation.geos`: Initializes `position`, `velocity`, `acceleration`, etc.
    *   `constellation.geosmod`: Initializes `position`, `velocity`, `acceleration`, etc.
    *   `propagation.add_satellites_from_tle`: Initializes `position`, `velocity`, etc.
    *   `propagation.propagate_satellites`: Updates `position`.
    *   `radiometry_test.fixedSat`: Initializes or appends.
*   `detector`: `SimpleNamespace`
    *   `constellation.geos`: Initializes with `makeBlankDetector`.
    *   `constellation.geosmod`: Initializes with `makeDetector`.
    *   `propagation.add_satellites_from_tle`: Initializes with `makeBlankDetector`.
    *   `observatories.add_observatories`: Initializes with `makeBlankDetector`.
    *   `detector.makeBlankDetector`: Creates a blank detector object.
    *   `detector.makeDetector`: Creates and initializes a detector object.
    *   `detector.setDetectorFOV`: Updates `fov`.
    *   `detector.setDetectorIntegrationTime`: Updates `integrationTime`.
    *   `detector.detectorPointingInitialize`: Initializes `pointing_state` and `pointing`.
    *   `pointing.update_detector_pointing`: Updates `pointing` and `pointing_state`.
    *   `radiometry_test.fixedSat`: Initializes or appends.
*   `fixedpoints`: `Dict[str, np.ndarray]`
    *   `targets.add_fixed_points`: Initializes `position`, `size`, `albedo`.
    *   `radiometry_test.fixedTarget`: Initializes or appends to `position`, `size`, `albedo`.
*   `observatories`: `Dict[str, np.ndarray]`
    *   `observatories.add_observatories`: Initializes `position`, `velocity`, `acceleration`, etc.
*   `cadenceStructure`: `Dict[str, any]`
    *   `cadenceController.initCadence`: Initializes the schedule based on detector integration.
*   `initial_detector_params`: `Dict[str, any]`
    *   `radiometry_test.fixedSat`: Stores baseline parameters for creating consistent detectors.

## Demos and Tests

This section describes the demo scripts and testing utilities available in the toolkit.

### Core Demos
*   `all_demos.py`: Central script to run major demos and optionally save to HTML.
*   `generate_report.py`: Runs plotting demos and generates `demo_plots.html`.
*   `demo_constellation.py`: Visualizes the creation of satellite constellations.
*   `demo_fixedpoints.py`: Plots the fixed target data structure in 3D.
*   `demo_lambertian.py`: Demonstrates the Lambertian sphere brightness model.
*   `demo_sky_scan.py`: Simulates a full sky scan with multiple detectors.
*   `demo_pointing_vectors.py`: Visualizes satellite pointing directions in 3D.
*   `demo_pointing_sequence.py`: Shows a sequence of pointing updates over time.
*   `demo_pointing_plot.py`: Comprehensive visualization of pointing history.
*   `demo_exclusion_table.py`: Generates a table showing visibility exclusions (Sun/Moon/Earth).
*   `demo_show_geo_search.py`: Demonstrates searching for GEO satellites from ground.
*   `demo1.py`, `demo2.py`, `demo3.py`: Basic demonstration scripts for satellite trajectories.
*   `demogeo.py`: Specific demo for Geostationary orbit visualization.
*   `plot_satellite_brightness.py`: Plots apparent magnitude of satellites over time.

### Verification and Tests
*   `verify_cadence.py`: Verifies `cadenceController` logic and collects results via `DataHandler`.
*   `tests/test_detector.py`: Unit tests for detector initialization and property setting.
*   `tests/test_lambertian.py`: Unit tests for the Lambertian radiance model.
*   `tests/test_minimalsimulation.py`: Unit tests for core simulation data structure.

## Python Files and Functions

### `__init__.py`
Marks the directory as a Python package.

### `all_demos.py`
*   `demo_vector_resorting_plot() -> go.Figure`: Runs resorting test and returns the figure.
*   `run_all_demos(save_html=False)`: Orchestrates execution of all registered demo functions.

### `cadenceController.py`
*   `initCadence(sim_data: dict)`: Groups detectors by integration time and schedules events.
*   `nextIntegration(sim_data: dict, print_output: int = 0)`: Advances time, propagates
    satellites, and performs a targeted scan for the group.
*   `_update_next_schedule(sim_data: dict)`: Internal helper to identify the next event.

### `celestialbodies.py`
*   `add_celestial_bodies(sim_data: Dict[str, Any]) -> None`: Adds Sun and Moon structures.
*   `celestial_update(data_struct: Dict[str, Any], time_date: Optional[datetime] = None) -> ...`:
    Calculates and updates the GCRS positions of the Sun and Moon.

### `constants.py`
Defines physical constants (e.g., `EARTH_RADIUS`, `DEGREE`) and array indices.

### `constellation.py`
*   `geos(sim_data, n, fov) -> None`: Creates $n$ equally spaced GEO satellites.
*   `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None`: Advanced GEO creation.

### `dataHandling.py`
*   `class DataHandler`: Manages collection of simulation results and export.
    *   `__init__(self)`: Initializes empty collection.
    *   `add_results(self, results: dict)`: Appends dict of scan results to collection.
    *   `get_dataframe(self) -> pd.DataFrame`: Combines results into a single Pandas DataFrame.
    *   `save_to_csv(self, filename: str)`: Saves results to a CSV file.
    *   `save_to_parquet(self, filename: str)`: Saves results to a Parquet file.
    *   `clear(self)`: Clears results to start a new simulation run.

### `detector.py`
*   `setDetectorFOV(sim_data, fovSize)`: Global update of all detector fields-of-view.
*   `setDetectorIntegrationTime(sim_data, itime)`: Global update of all integration times.
*   `makeBlankDetector(n)`: Creates a minimal detector SimpleNamespace placeholder.
*   `makeDetector(n, band, fov, ifov, aper, intTime: float = 1.0, category=[''], asset_index=[0], ...)`: Creates a detailed
    detector object with filter zero-points, quantum efficiency, and exclusion angles.
*   `detectorPointingInitialize(sim_data, grid_points)`: Initializes pointing grids and state.
*   `requiredIntegrationTime(limitingMag, SNR, d, debug = 0)`: Exposure time calculation.
*   `testdetector()`: Diagnostic example comparing values against theoretical curves.

### `exclusion.py`
*   `exclusion(data_struct: Dict[str, Any], satellite_index: int, ...)`: Core logic for
    Sun/Moon/Earth exclusion checks.
*   `update_exclusion_table(data_struct: Dict[str, Any], ...)`: Updates exclusion table for all
    satellites against all fixed points.

### `fibonacciSearch.py`
*   `pointing_vectors(n: int) -> np.ndarray`: Generates uniform points on a unit sphere.
*   `resort_vectors_by_proximity(unit_vectors: np.ndarray) -> np.ndarray`: Reorders vectors
    to minimize slew distance during scans.
*   `plot_vectors_on_sphere(vectors: np.ndarray, title: str) -> go.Figure`: 3D plotly visualization.
*   `test_vector_resorting()`: Tests vector reordering and plots resulting distances.

### `fluxes.py`
*   `fluxes(band)`: Returns solar, space, and sky background fluxes for a given filter band.

### `generate_log_spherical_points.py`
*   `generate_log_spherical_points(num_points, inner_radius, outer_radius, seed=None)`:
    Generates 3D points with logarithmic radial and uniform angular distribution.

### `generate_report.py`
*   `generate_demo_html_report()`: Aggregates multiple demo plots into `demo_plots.html`.

### `lambertian.py`
*   `lambertiansphere(angle_light_observer, albedo, radius, base_brightness, debug=0) -> ...`:
    Vectorized calculation of sphere surface radiance.
*   `simple_lambertian(diameter, distance, albedo, angle, base_brightness) -> float`: Apparent
    brightness of a single Lambertian sphere.
*   `includedAngle(vectors1, vectors2) -> np.ndarray`: Vectorized included angle in radians.

### `minimalsimulation.py`
*   `create_empty_simulation(start_time, delta_time=60.0) -> Dict[str, Any]`: Bootstraps empty
    `sim_data` dictionary.

### `observatories.py`
*   `add_observatories(sim_data, num_observatories) -> None`: Adds ground-based assets.

### `plotting_3d.py`
*   `plot_3d_scatter(positions, title, plot_time, ...)`: General-purpose 3D plotting utility.

### `plotting_vectors.py`
*   `plot_pointing_vectors(data_struct, title, plot_time) -> go.Figure`: Pointing vector plot.

### `pointing.py`
*   `generate_pointing_sphere(sim_data, n_points, debug=False) -> None`: Populates sphere cache.
*   `update_detector_pointing(sim_data, debug=False) -> None`: Progresses pointing direction.
*   `demo_exclusion_pointing()`: Demonstrates satellite pointing with solar exclusion.
*   `jerk(sim_data, satellite_indices) -> Dict[str, Any]`: Randomly perturbs pointing.

### `propagation.py`
*   `add_satellites_from_tle(sim_data, tle_file_path, sat_category) -> None`: Loads from TLE.
*   `readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]`: Extracts elements from TLE.
*   `propagate_satellites(data_struct, time_date, sat_category=None) -> Dict[str, Any]`: Updates
    satellite positions using SGP4 or analytical models.

### `radiometry_calcs.py`
*   `fluxes(band)`: Looks up astronomical band filter parameters and returns flux values.
*   `mag(x: float) -> float`: Calculates linear ratio to magnitude.
*   `amag(x: float) -> float`: Calculates magnitude to linear ratio.
*   `_planck_law(wav_m, temp_k) -> float`: Planck's law helper.
*   `blackbody_flux(temperature, lambda_short, lambda_long) -> float`: Integrated Planck radiance.
*   `stefan_boltzmann_law(temperature: float) -> float`: Stefan-Boltzmann power.
*   `plot_blackbody_spectrum(temperature: float)`: Plots spectrum from 0.5 to 30 microns.
*   `plot_blackbody_spectrum_visible_nir(temperature: float)`: Plots visible/NIR spectrum.

### `radiometry_data.py`
Contains astronomical filter definitions and zero-point data.

### `radiometry_test.py`
*   `fixedSat(sim_data, x, y, z, fov=...)`: Places a static satellite and detector.
*   `fixedTarget(sim_data, size, x, y, z)`: Places a static Lambertian target.
*   `fixSun(sim_data) -> None`: Fixes the Sun at 1 AU on the negative X-axis.
*   `demoFixed()`: End-to-end setup demonstration for radiometry validation.

### `scandetectors.py`
*   `get_spherical_coords(arr)`: Converts Cartesian coordinates to spherical angles.
*   `scandetectors(sim_data, print_output=0, mask=None)`: High-performance vectorized scan
    calculating visibility, fluxes, and signal-to-noise ratios.

### `sim_check.py`
*   `sim_check(sim_data)`: Prints diagnostic summary of simulation data structures.

### `simulationTemplate.py`
*   `run_simulation_template()`: Demonstrates template lifecycle using Controller & Handler.

### `targets.py`
*   `add_fixed_points(sim_data, num_points=100, size=1.0, innerRadius=..., outerRadius=...)`:
    Populates static reference points.

### `verify_cadence.py`
*   `run_verification()`: End-to-end verification script for event cadence.

## Dependencies

VibeVolts requires the following Python libraries:

*   `numpy`
*   `astropy`
*   `jplephem`
*   `sgp4`
*   `plotly`
*   `scipy`
*   `ipython`
*   `pandas` (required for `dataHandling.py` and `verify_cadence.py`)

You can install them using pip:

```bash
pip install numpy astropy jplephem sgp4 plotly scipy ipython pandas
```
