# VibeVolts Documentation

## Project Overview

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets. These are intended be evolved in a discrete event simulation. 

## Data Structure

The current state of the simulation is stored in a dictionary typically called `sim_data` which is passed between the routines that initialize and operate on the components to initialize, evolve, and interrogate the overall system.  The different components of the system are typically dealt with by different modules.   

In order to maximize the use of numpy and other parallel tools,  components like satellite elements are typically stored in arrays comprising all of the satellites in one numpy array, in this way efficient parallel numpy routines can easily be leveraged.

### Top-Level Keys

- `start_time`: `datetime`
  - **Description**: The starting time and date of the simulation. Must be a timezone-aware datetime object set to UTC.

- `delta_time`: `float`
  - **Description**: The time step for the simulation in seconds.

- `counts`: `dict`
  - **Description**: A dictionary holding counts of various simulation objects.

- `pointing_spheres`: `dict`
  - **Description**: A dictionary to hold pre-computed pointing spheres. The keys are the number of points in the sphere.

- `celestial`: `dict`
  - **Description**: Holds data for celestial bodies (Sun and Moon).
  - **Sub-keys**:
    - `position`: `np.ndarray` (2, 3) - Position vectors (x, y, z) in meters.
    - `velocity`: `np.ndarray` (2, 3) - Velocity vectors in m/s.
    - `acceleration`: `np.ndarray` (2, 3) - Acceleration vectors in m/s^2.

- `fixedpoints`: `dict`
  - **Description**: Holds data for fixed reference points in the GCRS frame.
  - **Sub-keys**:
    - `position`: `np.ndarray` (n, 3) - Position vectors of fixed points.
    - `visibility`: `np.ndarray` (n, m) - Visibility table where rows are fixed points and columns are satellites. Value is 1 if visible, 0 otherwise.

- `satellites` (and other satellite categories like `red_satellites`): `dict`
  - **Description**: Holds data for a category of satellites.
  - **Sub-keys**:
    - `position`: `np.ndarray` (n, 3) - Position vectors in meters.
    - `velocity`: `np.ndarray` (n, 3) - Velocity vectors in m/s.
    - `acceleration`: `np.ndarray` (n, 3) - Acceleration vectors in m/s^2.
    - `orbital_elements`: `np.ndarray` (n, 6) - Keplerian orbital elements.
    - `epochs`: `list[datetime]` - Epoch for each satellite's orbital elements.
    - `pointing`: `np.ndarray` (n, 3) - Pointing direction vector.
    - `pointing_state`: `np.ndarray` (n, 2) - State of the pointing sequence for each satellite.
    - `detector`: `SimpleNamespace`
      - **Description**: A SimpleNamespace object containing the detector's properties.
      - **Sub-keys**:
        - `apertureArea`: `np.ndarray` (n) - Aperture area in square meters.
        - `pixelArea`: `np.ndarray` (n) - Pixel area in square arcseconds.
        - `qe`: `np.ndarray` (n) - Quantum efficiency (0.0 to 1.0).
        - `photoEff`: `np.ndarray` (n) - Fraction of photons in photometry bucket.
        - `pixCount`: `np.ndarray` (n) - Total number of pixels in the detector.
        - `solarEx`: `np.ndarray` (n) - Solar exclusion angle in radians.
        - `lunarex`: `np.ndarray` (n) - Lunar exclusion angle in radians.
        - `earthEx`: `np.ndarray` (n) - Earth exclusion angle in radians.
        - `skyBack`: `np.ndarray` (n) - Sky background in photons per square steradian.
        - `zpCal`: `np.ndarray` (n) - Filter calibration zero point in photons/m^2/s.
        - `integrationTime`: `np.ndarray` (n) - Integration time to reach limiting magnitude.
        - `fov`: `np.ndarray` (n) - Field of view in radians.
        - `ifov`: `np.ndarray` (n) - Instantaneous field of view in radians.
        - `filt`: `list[str]` (n) - Filter name.

- `observatories`: `dict`
  - **Description**: Holds data for ground-based observatories.
  - **Sub-keys**:
    - `position`: `np.ndarray` (n, 3) - Position vectors in meters.
    - `velocity`: `np.ndarray` (n, 3) - Velocity vectors in m/s.
    - `acceleration`: `np.ndarray` (n, 3) - Acceleration vectors in m/s^2.
    - `pointing`: `np.ndarray` (n, 3) - Pointing direction vector.
    - `detector`: `SimpleNamespace`
      - **Description**: A SimpleNamespace object containing the detector's properties.
      - **Sub-keys**:
        - `apertureArea`: `np.ndarray` (n) - Aperture area in square meters.
        - `pixelArea`: `np.ndarray` (n) - Pixel area in square arcseconds.
        - `qe`: `np.ndarray` (n) - Quantum efficiency (0.0 to 1.0).
        - `photoEff`: `np.ndarray` (n) - Fraction of photons in photometry bucket.
        - `pixCount`: `np.ndarray` (n) - Total number of pixels in the detector.
        - `solarEx`: `np.ndarray` (n) - Solar exclusion angle in radians.
        - `lunarex`: `np.ndarray` (n) - Lunar exclusion angle in radians.
        - `earthEx`: `np.ndarray` (n) - Earth exclusion angle in radians.
        - `skyBack`: `np.ndarray` (n) - Sky background in photons per square steradian.
        - `zpCal`: `np.ndarray` (n) - Filter calibration zero point in photons/m^2/s.
        - `integrationTime`: `np.ndarray` (n) - Integration time to reach limiting magnitude.
        - `fov`: `np.ndarray` (n) - Field of view in radians.
        - `ifov`: `np.ndarray` (n) - Instantaneous field of view in radians.
        - `filt`: `list[str]` (n) - Filter name.

## Modules

### `simulation.py`
*   `create_empty_simulation(start_time: datetime, delta_time: float = 60.0)`: Initializes the top-level `sim_data` dictionary with `start_time`, `delta_time`, `counts`, and `pointing_spheres`.

### `celestialbodies.py`
*   `add_celestial_bodies(sim_data: Dict[str, Any])`: Adds the `celestial` dictionary to `sim_data`.
*   `celestial_update(data_struct: Dict[str, Any], time_date: datetime)`: Calculates and updates the positions of the Sun and Moon.

### `propagation.py`
*   `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str)`: Adds and initializes a category of satellites from a TLE file.
*   `celestial_update(data_struct: Dict[str, Any], time_date: datetime)`: Calculates and updates the positions of the Sun and Moon.
*   `readtle(tle_file_path: str)`: Reads a TLE file and extracts orbital elements and epochs for each satellite.
*   `propagate_satellites_new(data_struct: Dict[str, Any], time_date: datetime, sat_category: str=None)`: Updates satellite positions and pointing vectors based on their orbital elements.

### `observatories.py`
*   `add_observatories(sim_data: Dict[str, Any], num_observatories: int)`: Adds observatory data structures to the simulation data.

### `constellation.py`
*   `geos(sim_data, n, fov)`: Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.
*   `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag)`: Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

### `targets.py`
*   `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100)`: Adds a structure for fixed reference points in the GCRS frame.

### `visibility.py`
*   `is_visible(r1, r2, R_body)`: Checks if two points are visible to each other with a body in between.

### `pointing.py`
*   `pointing_place_update(data_struct: Dict[str, Any])`: Increments the pointing place for all satellites, wrapping around if necessary.
*   `jerk(data_struct: Dict[str, Any], satellite_indices: np.ndarray)`: Moves the pointing vector of a specific satellite by 0.3 radians in a random direction.
*   `generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int)`: Generates a pointing sphere with n_points and stores it in the data_struct.
*   `update_satellite_pointing(data_struct: Dict[str, Any])`: Updates the pointing vector for each satellite based on its pointing state.
*   `find_and_jerk_blind_satellites(data_struct: Dict[str, Any])`: Finds satellites with no visibility and applies the 'jerk' function to them.

### `detector.py`
*   `makeBlankDetector(n)`: Creates a blank detector array.
*   `makeDetector(n, band, fov, ifov, aper, qe=0.5, photfrac=0.7, solarex=20 * DEGREE, lunarex=10 * DEGREE, earthex=15 * DEGREE)`: Creates a detector array with the given parameters.
*   `requiredIntegrationTime(limitingMag, SNR, d, debug=0)`: Calculates the required integration time to achieve a given limiting magnitude.
*   `testdetector()`: A test function for the detector module.

### `lambertian.py`
*   `simple_lambertian(diameter: float, distance: float, albedo: float, angle: float, base_brightness: float)`: Calculates the apparent brightness of a Lambertian sphere.
*   `lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: float, radius: float)`: Calculates the effective brightness of a Lambertian sphere.

### `radiometry_data.py`
* This module contains radiometric data for standard astronomical filters.

### `radiometry_calcs.py`
*   `mag(x: float)`: Calculates a magnitude value from a linear ratio.
*   `amag(x: float)`: Calculates the linear ratio from a magnitude value.
*   `_planck_law(wav_m: float, temp_k: float)`: Helper function for Planck's law for spectral radiance.
*   `blackbody_flux(temperature: float, lambda_short: float, lambda_long: float)`: Numerically computes the integrated spectral radiance of a blackbody over a given wavelength band.
*   `stefan_boltzmann_law(temperature: float)`: Calculates the total power radiated per unit area by a blackbody using the Stefan-Boltzmann law.
*   `plot_blackbody_spectrum(temperature: float)`: Plots the spectral radiance of a blackbody from 0.5 to 30 microns.
*   `plot_blackbody_spectrum_visible_nir(temperature: float)`: Plots the spectral radiance of a blackbody from 0.1 to 1 micron.
*   `sat_magnitude(size: float, range: float, angle: float, band: str)`: given a satellite size and a waveband and range pull the brightness of the sun and the calibration from radiometry_data

### `plotting_3d.py`
*   `plot_3d_scatter(positions: np.ndarray, title: str, plot_time: datetime, labels: Optional[List[str]]=None, marker_size: int=1, trace_name: str='Points')`: Creates a 3D plot of object positions.

### `plotting_vectors.py`
*   `plot_pointing_vectors(data_struct: Dict[str, Any], title: str, plot_time: datetime)`: Creates a 3D plot of satellites with pointing vectors.

### `pointing_vectors.py`
* This module is empty.

### `generate_log_spherical_points.py`
*   `generate_log_spherical_points(num_points: int, inner_radius: float, outer_radius: float, object_size_m: float=1.0, seed: int=None)`: Generates 3D points with logarithmic radial and uniform angular distribution.

### `sim_check.py`
*   `sim_check(sim_data)`: Prints a brief summary of what's present in a sim_data structure.

### `fibonacciSearch.py`
*   `pointing_vectors(n: int)`: Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.
*   `resort_vectors_by_proximity(unit_vectors: np.ndarray)`: Resorts a list of vectors by making each subsequent vector the closest one in the remaining set to the previous one.
*   `plot_vectors_on_sphere(vectors: np.ndarray, title: str)`: Creates a 3D plot of vectors on a sphere.
*   `test_vector_resorting()`: Tests the vector resorting and plots the Euclidean distance between subsequent vectors.

### `exclusion.py`
*   `exclusion(data_struct: Dict[str, Any], satellite_index: int, print_debug: bool = False)`: Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.

### `all_demos.py`
*   `run_all_demos(save_html=False)`: Runs all demo functions, and either shows them inline or saves them to a single HTML file.
*   `demo_vector_resorting_plot()`: Runs the test_vector_resorting function and returns its figure.

## Demos

This section describes the demo scripts that are available in the toolkit.

### `demo1.py`
*   `demo1()`: Runs a full demonstration of the simulation tools.

### `demo2.py`
*   `demo2()`: Runs a demonstration plotting satellite and celestial positions.

### `demo3.py`
*   `demo3()`: Runs a demonstration plotting a single LEO satellite trajectory.

### `demo4.py`
*   `demo4()`: Runs a demonstration plotting a single GEO satellite trajectory.

### `demo_common.py`
*   `initialize_standard_simulation(start_time: datetime)`: Initializes a standard simulation with a predefined set of satellites.

### `demo_constellation.py`
*   `demo_constellation()`: Runs a demonstration of the constellation creation tools.

### `demo_exclusion_debug_print.py`
*   `demo_exclusion_debug_print()`: Demonstrates the debug printing feature of the exclusion function.

### `demo_exclusion_table.py`
*   `demo_exclusion_table()`: Demonstrates the creation and visualization of the exclusion table.

### `demo_fixedpoints.py`
*   `demo_fixedpoints()`: Demonstrates the fixedpoints data structure by plotting it in 3D.

### `demo_lambertian.py`
*   `demo_lambertian()`: Runs a demonstration of the lambertiansphere function, including example calculations and a plot.

### `demo_pointing_plot.py`
*   `demo_pointing_plot()`: Demonstrates the plot_pointing_vectors function.

### `demo_pointing_sequence.py`
*   `demo_pointing_sequence()`: Demonstrates the satellite pointing sequence functionality.

### `demo_pointing_vectors.py`
*   `demo_pointing_vectors()`: Demonstrates the generation and plotting of pointing vectors.

### `demo_requiredIntegrationTime.py`
*   `demo_requiredIntegrationTime()`: Demonstrates the requiredIntegrationTime function.

### `demo_show_geo_search.py`
*   `demo_show_geo_search()`: This demo initializes a simulation, adds a GEO constellation, and then generates several plots to visualize the satellite pointing updates and the RA/Dec history of one satellite.
*   `record_ra_dec()`: Records the RA and Dec of a satellite.

### `demo_sky_scan.py`
*   `demo_sky_scan()`: Performs a sky scan from a GEO satellite to map celestial exclusion zones.

### `generate_report.py`
*   `generate_demo_html_report()`: Runs all plotting demos and saves the output to a single HTML file.

### `plot_satellite_brighness.py`
*   `plot_satellite_brightness()`: Plots the apparent V-band photon flux and magnitude of satellites with various diameters over a range of distances.



## Building and Running

### Dependencies

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

### Running the Demos

The `all_demos.py` script provides a comprehensive demonstration of the toolkit's features. You can run the script directly from your terminal to see the plots displayed in your browser:

```bash
python all_demos.py
```

You can also import and run the `run_all_demos` function from a Jupyter Notebook to display all plots inline.

```python
import all_demos
all_demos.run_all_demos()
```

To save all the demo plots to a single HTML file named `all_demo_plots.html`, run the following from a Python script or notebook:

```python
import all_demos
all_demos.run_all_demos(save_html=True)
```


## Development Conventions
*  **Text Files**: All purely text file have lines that are no more than 100 charcters long to aid reading.
*   **Data Structures**: The simulation state is managed in a central dictionary. This dictionary is initialized as a minimal structure using `create_empty_simulation` from `simulation.py`. Components like satellites, observatories, and celestial bodies are then added incrementally using dedicated functions (e.g., `add_satellites_from_tle`, `add_observatories`), making the structure highly modular and flexible.
*   **Modularity**: The code is organized into modules, each with a specific responsibility. This makes the code easy to understand, maintain, and extend.
*   **Vectorization**: The code makes extensive use of NumPy for vectorized operations, which provides a significant performance improvement over iterating through lists.
*   **Type Hinting**: The code uses type hints to improve readability and allow for static analysis.
*   **Docstrings**: All functions have docstrings that explain their purpose, arguments, and return values.
*   **Constants**: Constants are defined in `constants.py` to avoid magic numbers in the code.
