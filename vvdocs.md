# VibeVolts Documentation

## Project Overview

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets. These are intended be evolved in a discrete event simulation.

## Data Structure

The current state of the simulation is stored in a dictionary typically called `sim_data` which is passed between the routines that initialize and operate on the components to initialize, evolve, and interrogate the overall system. The different components of the system are typically dealt with by different modules.

### `sim_data` dictionary structure:

*   `start_time`: `datetime`
    *   `minimalsimulation.create_empty_simulation`: Initializes.
*   `time`: `datetime`
    *   `minimalsimulation.create_empty_simulation`: Initializes.
    *   `propagation.propagate_satellites`: Updated.
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
    *   `constellation.geos`: Initializes `position`, `velocity`, `acceleration`, `orbital_elements`, `epochs`.
    *   `constellation.geosmod`: Initializes or appends to `position`, `velocity`, `acceleration`, `orbital_elements`, `epochs`.
    *   `propagation.add_satellites_from_tle`: Initializes `position`, `velocity`, `acceleration`, `orbital_elements`, `epochs`, `pointing`, `pointing_state`.
    *   `propagation.propagate_satellites`: Updates `position`.
    *   `radiometry_test.fixedSat`: Initializes or appends to `position`, `velocity`, `acceleration`, `orbital_elements`, `epochs`.
*   `detector`: `SimpleNamespace`
    *   `constellation.geos`: Initializes with `makeBlankDetector`.
    *   `constellation.geosmod`: Initializes with `makeDetector`.
    *   `propagation.add_satellites_from_tle`: Initializes with `makeBlankDetector`.
    *   `observatories.add_observatories`: Initializes with `makeBlankDetector`.
    *   `detector.makeBlankDetector`: Creates a blank detector object.
    *   `detector.makeDetector`: Creates and initializes a detector object. Includes `integrationTime` (NumPy array).
    *   `detector.setDetectorFOV`: Updates `fov`.
    *   `detector.setDetectorIntegrationTime`: Updates `integrationTime`.
    *   `detector.detectorPointingInitialize`: Initializes `pointing_state` and updates `pointing`.
    *   `pointing.update_detector_pointing`: Updates `pointing` and `pointing_state`.
    *   `pointing.jerk`: Updates `pointing`.
    *   `radiometry_test.fixedSat`: Initializes or appends to the detector object.
*   `fixedpoints`: `Dict[str, np.ndarray]`
    *   `targets.add_fixed_points`: Initializes `position`, `size`, `albedo`.
    *   `radiometry_test.fixedTarget`: Initializes or appends to `position`, `size`, `albedo`.
*   `observatories`: `Dict[str, np.ndarray]`
    *   `observatories.add_observatories`: Initializes `position`, `velocity`, `acceleration`, `pointing`.
*   `cadenceStructure`: `Dict[str, any]`
    *   `cadenceController.initCadence`: Initializes the simulation schedule based on detector integration times. Contains `cadenceList` (groups with `scanInterval`, `scanMask`, `scanNext`), `nextTime`, and `nextGroup`.
*   `initial_detector_params`: `Dict[str, any]`
    *   `radiometry_test.fixedSat`: Stores baseline parameters for creating consistent detectors.

## Demos and Tests

This section describes the demo scripts and testing utilities available in the toolkit.

### Core Demos
*   `all_demos.py`: Central script to run major demos and optionally save results to `all_demo_plots.html`.
*   `generate_report.py`: Runs a subset of plotting demos and generates a structured HTML report `demo_plots.html`.
*   `demo_constellation.py`: Visualizes the creation of satellite constellations.
*   `demo_fixedpoints.py`: Plots the fixed target data structure in 3D.
*   `demo_lambertian.py`: Demonstrates the Lambertian sphere brightness model.
*   `demo_sky_scan.py`: Simulates a full sky scan with multiple detectors.
*   `demo_pointing_vectors.py`: Visualizes satellite pointing directions in 3D.
*   `demo_pointing_sequence.py`: Shows a sequence of pointing updates over time.
*   `demo_pointing_plot.py`: Comprehensive visualization of pointing history.
*   `demo_exclusion_table.py`: Generates a table showing visibility exclusions (Sun/Moon/Earth).
*   `demo_show_geo_search.py`: Demonstrates searching for GEO satellites from ground observatories.
*   `demo1.py`, `demo2.py`, `demo3.py`: Basic demonstration scripts for satellite positioning and trajectories.
*   `demogeo.py`: Specific demo for Geostationary orbit visualization.
*   `plot_satellite_brighness.py`: Plots apparent magnitude of satellites over time.

### Verification and Tests
*   `verify_cadence.py`: A specialized script that verifies the `cadenceController` logic by running multiple integration steps and collecting results via `DataHandler`.
*   `tests/test_detector.py`: Unit tests for detector initialization and property setting.
*   `tests/test_lambertian.py`: Unit tests for the Lambertian radiance model.
*   `tests/test_minimalsimulation.py`: Unit tests for the core simulation data structure initialization.

## Python Files and Functions

### `__init__.py`
Marks the directory as a Python package.

### `all_demos.py`
*   `run_all_demos(save_html=False)`: Orchestrates execution of all registered demo functions.

### `cadenceController.py`
*   `initCadence(sim_data: dict)`: Groups detectors by integration time and initializes the simulation schedule.
*   `nextIntegration(sim_data: dict, print_output: int = 0)`: Advances time to the next event, propagates satellites, and performs a targeted scan.
*   `_update_next_schedule(sim_data: dict)`: Internal helper to identify the next chronological event.

### `celestialbodies.py`
*   `add_celestial_bodies(sim_data: dict)`: Adds Sun anad Moon structures.
*   `celestial_update(data_struct: dict, time_date: datetime = None)`: Updates celestial body positions (GCRS).

### `constants.py`
Defines physical constants (e.g., `EARTH_RADIUS`, `DEGREE`, `ARCSEC`) and array indices for orbital elements and pointing states.

### `constellation.py`
*   `geos(sim_data, n, fov)`: Creates $n$ equally spaced GEO satellites.
*   `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag)`: Advanced GEO constellation creation with detector parameters.

### `dataHandling.py`
*   `class DataHandler`: Manages collection of simulation results and export to Pandas DataFrames, CSV, or Parquet.

### `detector.py`
*   `makeDetector(n, band, fov, ifov, aper, ...)`: Creates a detailed detector object with radiometric properties.
*   `makeBlankDetector(n)`: Creates a minimal detector placeholder.
*   `setDetectorFOV(sim_data, fovSize)`: Global update of all detector fields-of-view.
*   `setDetectorIntegrationTime(sim_data, itime)`: Global update of all detector integration times.
*   `detectorPointingInitialize(sim_data, grid_points)`: Initializes pointing grids and state.
*   `requiredIntegrationTime(limitingMag, SNR, d, debug=0)`: Analytical calculation of necessary exposure time.

### `exclusion.py`
*   `exclusion(data_struct: dict, satellite_index: int, ...)`: Core logic for Sun/Moon/Earth exclusion checks.

### `fibonacciSearch.py`
*   `pointing_vectors(n: int)`: Generates $n$ uniform points on a sphere using Fibonacci lattice.
*   `resort_vectors_by_proximity(unit_vectors)`: Optimizes pointing sequences to minimize "slew" distance.

### `fluxes.py`
*   `fluxes(band)`: Returns solar, space, and sky background fluxes for a given filter band.

### `generate_log_spherical_points.py`
*   `generate_log_spherical_points(num_points, inner_radius, outer_radius, ...)`: Generates 3D points with logarithmic radial distribution.

### `generate_report.py`
*   `generate_demo_html_report()`: Aggregates multiple demo plots into a single HTML file.

### `lambertian.py`
*   `lambertiansphere(angle_light_observer, albedo, radius, base_brightness, ...)`: Vectorized calculation of sphere surface radiance.
*   `simple_lambertian(diameter, distance, albedo, angle, base_brightness)`: Scalar version for single-object calculations.
*   `includedAngle(vectors1, vectors2)`: Vectorized calculation of angles between vector pairs.

### `minimalsimulation.py`
*   `create_empty_simulation(start_time, delta_time=60.0)`: Bootstraps the `sim_data` dictionary.

### `observatories.py`
*   `add_observatories(sim_data, num_observatories)`: Adds ground-based assets.

### `plotting_3d.py`
*   `plot_3d_scatter(positions, title, plot_time, ...)`: General-purpose 3D plotting utility.

### `plotting_vectors.py`
*   `plot_pointing_vectors(data_struct, title, plot_time)`: Specialized 3D plot for pointing vectors.

### `pointing.py`
*   `generate_pointing_sphere(sim_data, n_points, ...)`: Populates the pointing sphere cache.
*   `update_detector_pointing(sim_data, debug=False)`: Progresses detectors to their next valid pointing direction.
*   `jerk(sim_data, satellite_indices)`: Perturbs satellite pointing for testing.

### `propagation.py`
*   `add_satellites_from_tle(sim_data, tle_file_path, sat_category)`: Loads satellites from TLE files.
*   `propagate_satellites(data_struct, time_date, sat_category=None)`: Updates positions using SGP4/analytical models.

### `radiometry_calcs.py`
*   `mag(x)`, `amag(x)`: Linear to magnitude (and vice versa) conversions.
*   `blackbody_flux(temperature, lambda_short, lambda_long)`: Integrated Planck radiance.

### `radiometry_data.py`
Contains astronomical filter definitions and zero-point data.

### `radiometry_test.py`
*   `fixedSat(sim_data, x, y, z, fov=...)`: Places a static satellite with a detector.
*   `fixedTarget(sim_data, size, x, y, z)`: Places a static Lambertian target.
*   `fixSun(sim_data)`: Fixes the Sun at 1 AU on the -X axis.

### `scandetectors.py`
*   `scandetectors(sim_data, print_output=0, mask=None)`: High-performance vectorized scan to find target/detector intersections and calculate SNR.

### `sim_check.py`
*   `sim_check(sim_data)`: Diagnostic utility to print `sim_data` contents.

### `targets.py`
*   `add_fixed_points(sim_data, num_points=100, size=1.0)`: Populates the simulation with static GCRS targets.

### `verify_cadence.py`
*   `run_verification()`: End-to-end simulation run to verify cadence and data collection.

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
