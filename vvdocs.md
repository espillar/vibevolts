# VibeVolts Documentation

## Project Overview

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets. These are intended be evolved in a discrete event simulation.

## Data Structure

The current state of the simulation is stored in a dictionary typically called `sim_data` which is passed between the routines that initialize and operate on the components to initialize, evolve, and interrogate the overall system. The different components of the system are typically dealt with by different modules.

### `sim_data` dictionary structure:

*   `start_time`: `datetime`
    *   `simulation.create_empty_simulation`: Initializes.
*   `time`: `datetime`
    *   `simulation.create_empty_simulation`: Initializes.
    *   `propagation.propagate_satellites`: Updated.
*   `delta_time`: `float`
    *   `simulation.create_empty_simulation`: Initializes.
*   `counts`: `Dict[str, int]`
    *   `simulation.create_empty_simulation`: Initializes.
    *   `celestialbodies.add_celestial_bodies`: Adds `'celestial'`.
    *   `constellation.geos`: Adds `'satellites'`.
    *   `constellation.geosmod`: Adds `'satellites'`.
    *   `propagation.add_satellites_from_tle`: Adds `sat_category`.
    *   `observatories.add_observatories`: Adds `'observatories'`.
    *   `targets.add_fixed_points`: Adds `'fixedpoints'`.
    *   `testObjects.fixedSat`: Adds/increments `'satellites'`.
    *   `testObjects.fixedTarget`: Adds/increments `'fixedpoints'`.
*   `pointing_spheres`: `Dict[int, np.ndarray]`
    *   `simulation.create_empty_simulation`: Initializes.
    *   `pointing.generate_pointing_sphere`: Adds a new pointing sphere.
*   `celestial`: `Dict[str, np.ndarray]`
    *   `celestialbodies.add_celestial_bodies`: Initializes `position`, `velocity`, `acceleration`.
    *   `celestialbodies.celestial_update`: Updates `position`.
    *   `testObjects.fixSun`: Updates `position` of the sun.
*   `satellites`: `Dict[str, any]`
    *   `constellation.geos`: Initializes `position`, `velocity`, `acceleration`, `orbital_elements`, `epochs`.
    *   `constellation.geosmod`: Initializes or appends to `position`, `velocity`, `acceleration`, `orbital_elements`, `epochs`.
    *   `propagation.add_satellites_from_tle`: Initializes `position`, `velocity`, `acceleration`, `orbital_elements`, `epochs`, `pointing`, `pointing_state`.
    *   `propagation.propagate_satellites`: Updates `position`.
    *   `testObjects.fixedSat`: Initializes or appends to `position`, `velocity`, `acceleration`, `orbital_elements`, `epochs`.
*   `detector`: `SimpleNamespace`
    *   `constellation.geos`: Initializes with `makeBlankDetector`.
    *   `constellation.geosmod`: Initializes with `makeDetector`.
    *   `propagation.add_satellites_from_tle`: Initializes with `makeBlankDetector`.
    *   `observatories.add_observatories`: Initializes with `makeBlankDetector`.
    *   `detector.makeBlankDetector`: Creates a blank detector object.
    *   `detector.makeDetector`: Creates and initializes a detector object.
    *   `detector.setDetectorFOV`: Updates `fov`.
    *   `detector.setDetectorIntegrationTime`: Updates `itime`.
    *   `detector.detectorPointingInitialize`: Initializes `pointing_state` and updates `pointing`.
    *   `pointing.update_detector_pointing`: Updates `pointing` and `pointing_state`.
    *   `pointing.jerk`: Updates `pointing`.
    *   `testObjects.fixedSat`: Initializes or appends to the detector object.
*   `fixedpoints`: `Dict[str, np.ndarray]`
    *   `targets.add_fixed_points`: Initializes `position`, `exclusion`, `size`, `albedo`.
    *   `testObjects.fixedTarget`: Initializes or appends to `position`, `exclusion`, `size`, `albedo`.
*   `observatories`: `Dict[str, np.ndarray]`
    *   `observatories.add_observatories`: Initializes `position`, `velocity`, `acceleration`, `pointing`.

## Demos

This section describes the demo scripts that are available in the toolkit.

*   `all_demos.py`:
    *   `run_all_demos(save_html=False)`: Runs all demo functions, and either shows them inline or saves them to a single HTML file.
*   `demo_constellation.py`:
    *   `demo_constellation() -> go.Figure`: Runs a demonstration of the constellation creation tools.
*   `demo_exclusion_pointing.py` (in `pointing.py`):
    *   `demo_exclusion_pointing()`: Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere.
*   `demo_fixedpoints.py`:
    *   `demo_fixedpoints() -> go.Figure`: Demonstrates the fixedpoints data structure by plotting it in 3D.
*   `detector.py`:
    *   `testdetector()`: testdetector creates an example that can be compared with some of the stuff in Curio.
*   `fibonacciSearch.py`:
    *   `test_vector_resorting()`: Tests the vector resorting and plots the Euclidean distance between subsequent vectors.
*   `generate_log_spherical_points.py`:
    *   The `if __name__ == '__main__':` block contains a demo of the point generation and visualization.
*   `radiometry_calcs.py`:
    *   `plot_blackbody_spectrum(temperature: float)`: Plots the spectral radiance of a blackbody from 0.5 to 30 microns.
    *   `plot_blackbody_spectrum_visible_nir(temperature: float)`: Plots the spectral radiance of a blackbody from 0.1 to 1 micron.
*   `sim_check.py`:
    *   The `if __name__ == '__main__':` block contains a demo of the `sim_check` function.
*   `testObjects.py`:
    *   `demoFixed()`: Demonstrates the use of the fixedSat and fixedTarget functions.

## Python Files and Functions

### `__init__.py`
This file is empty and marks the directory as a Python package.

### `all_demos.py`

*   `demo_vector_resorting_plot() -> go.Figure`: Runs the test_vector_resorting function and returns its figure.
*   `run_all_demos(save_html=False)`: Runs all demo functions, and either shows them inline or saves them to a single HTML file.

### `celestialbodies.py`

*   `add_celestial_bodies(sim_data: Dict[str, Any]) -> None`: Adds celestial body structures (for Sun and Moon) to the simulation data.
*   `celestial_update(data_struct: Dict[str, Any], time_date: Optional[datetime] = None) -> Dict[str, Any]`: Calculates and updates the positions of the Sun and Moon.

### `constants.py`
This file defines global constants for array indices and physical constants.

### `constellation.py`

*   `geos(sim_data, n,  fov) -> None`: Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.
*   `geosmod(sim_data, n, band,fov,ifov, aper, limitingmag) -> None`: Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

### `detector.py`

*   `setDetectorFOV(sim_data, fovSize)`: setDetectorFOV goes through the detectors in sim_data and changes the FOVs of all of them to size (radians).
*   `setDetectorIntegrationTime(sim_data, itime)`: setDetectorIntegrationTime goes through the detectors in sim_data and changes the integration times of all of them to `itime`.
*   `makeBlankDetector(n)`: makeBlankDetector makes and returns a detector SipleNamespace.
*   `makeDetector(n, band, fov, ifov, aper, qe = 0.5, photfrac=0.7, solarex= 20.0 * DEGREE,   lunarex= 10.0 * DEGREE,  earthex= 15.0 * DEGREE)`: makeDetector takes parameters of a sensor and stuffs a filter array and a detector array, which it returns.
*   `detectorPointingInitialize(sim_data, grid_points)`: We assume that sim_data['detectors'] loaded, but the pointing part of detectors is currently empty.
*   `requiredIntegrationTime(limitingMag, SNR, d, debug = 0)`: Calculates the required integration time to achieve a given limiting magnitude with a specified signal-to-noise ratio (SNR).
*   `testdetector()`: testdetector creates an example that can be compared with some of the stuff in Curio.

### `exclusion.py`

*   `exclusion(data_struct: Dict[str, Any], satellite_index: int, print_debug: bool = False) -> int`: Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.

### `fibonacciSearch.py`

*   `pointing_vectors(n: int) -> np.ndarray`: Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.
*   `resort_vectors_by_proximity(unit_vectors: np.ndarray) -> np.ndarray`: Resorts a list of vectors by making each subsequent vector the closest one in the remaining set to the previous one.
*   `plot_vectors_on_sphere(vectors: np.ndarray, title: str) -> go.Figure`: Creates a 3D plot of vectors on a sphere.
*   `test_vector_resorting()`: Tests the vector resorting and plots the Euclidean distance between subsequent vectors.

### `fluxes.py`
*   `fluxes(band)`: uses the FILTER_DATA table from radiometry_data.py for data.

### `generate_log_spherical_points.py`

*   `generate_log_spherical_points(num_points: int, inner_radius: float, outer_radius: float, seed: int = None) -> tuple[np.ndarray, np.ndarray]`: Generates 3D points with logarithmic radial and uniform angular distribution.

### `lambertian.py`

*   `lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: np.ndarray, radius: np.ndarray, base_brightness: np.ndarray) -> np.ndarray`: Calculates the illuminance of multiple lambertian spheres.
*   `simple_lambertian(diameter: float, distance: float, albedo: float, angle: float, base_brightness: float) -> float`: Calculates the apparent brightness of a Lambertian sphere.

### `observatories.py`

*   `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None`: Adds observatory data structures to the simulation data.

### `plotting_3d.py`
*   `plot_3d_scatter(positions: np.ndarray, title: str, plot_time: datetime, labels: Optional[List[str]] = None, marker_size: int = 1, trace_name: str = 'Points') -> go.Figure`: Creates a 3D plot of object positions.

### `plotting_vectors.py`
*   `plot_pointing_vectors(data_struct: Dict[str, Any], title: str, plot_time: datetime) -> go.Figure`: Creates a 3D plot of satellites with pointing vectors.

### `pointing.py`

*   `generate_pointing_sphere(sim_data: Dict[str, Any], n_points: int, debug: bool = False) -> None`: Generates a pointing sphere with n_points and stores it in the sim_data['pointing_spheres'][n_points].
*   `update_detector_pointing(sim_data: Dict[str, Any], debug: bool = False) -> None`: Updates the pointing vector for each detector, skipping excluded pointing directions.
*   `demo_exclusion_pointing()`: Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere.
*   `jerk(sim_data: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]`: Moves the pointing vector of specific satellites by 0.3 radians in a random direction.

### `propagation.py`

*   `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None`: Adds and initializes a category of satellites from a TLE file.
*   `readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]`: Reads a TLE file and extracts orbital elements and epochs for each satellite.
*   `propagate_satellites(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]`: Updates satellite positions based on their orbital elements.

### `radiometry_calcs.py`

*   `fluxes(band)`: uses the FILTER_DATA table from radiometry_data.py for data.
*   `mag(x: float) -> float`: Calculates a magnitude value from a linear ratio.
*   `amag(x: float) -> float`: Calculates the linear ratio from a magnitude value.
*   `_planck_law(wav_m: float, temp_k: float) -> float`: Helper function for Planck's law for spectral radiance.
*   `blackbody_flux(temperature: float, lambda_short: float, lambda_long: float) -> float`: Numerically computes the integrated spectral radiance of a blackbody over a given wavelength band.
*   `stefan_boltzmann_law(temperature: float) -> float`: Calculates the total power radiated per unit area by a blackbody using the Stefan-Boltzmann law.
*   `plot_blackbody_spectrum(temperature: float)`: Plots the spectral radiance of a blackbody from 0.5 to 30 microns.
*   `plot_blackbody_spectrum_visible_nir(temperature: float)`: Plots the spectral radiance of a blackbody from 0.1 to 1 micron.

### `radiometry_data.py`
This file contains radiometric data for standard astronomical filters.

### `scandetectors.py`

*   `get_spherical_coords(arr)`: given a n by 3 d array, return two 1D arrays with the theta and phi angles in radians.
*   `scandetectors(sim_data: dict)`: Scans for and processes detector data within the simulation.
*   `findVectorMask(values: np.ndarray, floorValue: float) -> np.ndarray`: Compares values in a 1D numpy array to a floorValue and returns a boolean mask.

### `sim_check.py`
*   `sim_check(sim_data)`: Prints a brief summary of what's present in a sim_data structure.

### `simulation.py`

*   `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]`: Initializes a minimal, empty data structure for a space simulation.

### `targets.py`

*   `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None`: Adds a structure for fixed reference points in the GCRS frame.

### `testObjects.py`

*   `fixedSat(sim_data: Dict[str, Any], x: float, y: float, z: float)`: Creates a single satellite fixed at the given x, y, z coordinates.
*   `fixedTarget(sim_data: Dict[str, Any], size: float, x: float, y: float, z: float)`: Places a fixed target at the given x, y, z coordinates.
*   `fixSun(sim_data: Dict[str, Any]) -> None`: Fixes the sun's position on the negative x-axis at 1 AU.
*   `demoFixed()`: Demonstrates the use of the fixedSat and fixedTarget functions.

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
