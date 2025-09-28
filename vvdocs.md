# VibeVolts Documentation

This document provides an overview of the data structures, functions, and
dependencies for the VibeVolts simulation toolkit.

## PDF Report

A comprehensive PDF report with a summary of the project and a full listing of the source code is available in [report.pdf](report.pdf).

## HTML Wiki

A pure HTML version of this documentation is available in the file [wiki.html](wiki.html).

## 1. Common Data Structures

The toolkit uses two primary data structures to manage simulation state and physical constants.

### 1.1. Simulation State Dictionary (`simulation_data`)

This is the central data structure, created by composition. A minimal simulation is created with `create_empty_simulation` and then populated by functions like `add_satellites_from_tle`, `add_observatories`, etc.

```python
{
    'start_time': datetime,
    'counts': {
        'celestial': 2,
        'satellites': num_satellites,
        'observatories': num_observatories,
        'red_satellites': num_red_satellites
    },
    'celestial': {
        'position': np.zeros((2, 3)),
        'velocity': np.zeros((2, 3)),
        'acceleration': np.zeros((2, 3)),
    },
    'satellites': {
        'position': np.zeros((num_satellites, 3)),
        'velocity': np.zeros((num_satellites, 3)),
        'acceleration': np.zeros((num_satellites, 3)),
        'orbital_elements': np.zeros((num_satellites, 6)),
        'epochs': [],
        'pointing': np.zeros((num_satellites, 3)),
        'pointing_state': np.zeros((num_satellites, 2), dtype=int),
        'detector': np.zeros((num_satellites, 7)),
    },
    'observatories': { ... },
    'red_satellites': { ... },
    'fixedpoints': {
        'position': np.zeros((num_points, 3)),
        'visibility': np.zeros((num_points, num_satellites), dtype=int)
    },
    'pointing_spheres': {},
    'delta_time': 60.0
}
```

#### Key Components:

*   **`orbital_elements`**: A NumPy array (`n x 6`) containing the classical
    orbital elements for each satellite. The column indices for this array are
    defined as constants in the `constants.py` module (e.g., `ORBITAL_A_IDX`,
    `ORBITAL_E_IDX`).

*   **`detector`**: A NumPy array (`n x 7`) containing the properties of each
    sensor. The column indices for this array are defined as constants in the
    `constants.py` module (e.g., `DETECTOR_APERTURE_IDX`,
    `DETECTOR_PIXEL_SIZE_IDX`).

*   **`fixedpoints`**: A dictionary containing the properties of the static
    points in space used as observation targets. By default, 100 points are
    generated.
    *   `position`: A NumPy array (`num_points x 3`) of static 3D points in
        the GCRS frame.
    *   `visibility`: A NumPy array (`num_points x num_satellites`) used to
        store whether a point is visible to a satellite.

*   **`pointing_state`**: A NumPy array (`n x 2`) containing the pointing
    control state for each satellite. The column indices are defined in
    `constants.py` (`POINTING_COUNT_IDX`, `POINTING_PLACE_IDX`).

*   **`pointing_spheres`**: A dictionary used to cache pre-generated pointing
    vector grids, indexed by the number of points in the grid. This prevents
    redundant calculations.

### 1.2. Radiometric Filter Data (`FILTER_DATA`)

This dictionary, located in `radiometry_data.py`, provides standard data for a
variety of astronomical filters, including Johnson-Cousins, SDSS, and JWST.

*   **`sun`**: The apparent magnitude of the Sun in the given filter.
*   **`sky`**: The typical dark sky brightness in magnitudes per square
    arcsecond.
*   **`central_wavelength`**: The central wavelength of the filter passband in
    nanometers (nm).
*   **`bandwidth`**: The effective width of the filter passband in nanometers
    (nm).
*   **`zero_point`**: The photon flux (in photons/sec/m²) corresponding to a
    0-magnitude star.

```python
{
    'U': {
        'sun': -26.03,
        'sky': 22.0,
        'central_wavelength': 365.0,
        'bandwidth': 66.0,
        'zero_point': 4.96e9,
    },
    'B': { ... },
    # ... and so on for V, R, I, J, H, K, g, r, i, z, L, M, N, and JWST filters.
}
```

### 1.3. Physical Constants

The `radiometry_data.py` module also defines the following physical constants:

*   **`AU_M`**: The astronomical unit in meters (`1.496e+11 m`).
*   **`RSUN_M`**: The radius of the Sun in meters (`6.957e+08 m`).

## 2. Existing Functions

This section describes the functions available in the toolkit, organized by module.

### 2.1. `simulation.py`

*   **`create_empty_simulation(start_time: datetime, delta_time: float = 60.0)`**: Creates a minimal, empty data structure for a space simulation.
*   **`add_celestial_bodies(sim_data: Dict[str, Any])`**: Adds celestial body structures (for Sun and Moon) to the simulation data.
*   **`add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100)`**: Adds a structure for fixed reference points in the GCRS frame.

### 2.2. `propagation.py`

*   **`add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str)`**: Adds and initializes a category of satellites from a TLE file.
*   **`celestial_update(data_struct: Dict[str, Any], time_date: datetime)`**: Updates the positions of
    the Sun and Moon for a given time using the `astropy` library.
*   **`readtle(tle_file_path: str)`**: Reads a Two-Line Element (TLE) file and
    returns a NumPy array of orbital elements and a list of epoch datetimes.
*   **`propagate_satellites(data_struct: Dict[str, Any], time_date: datetime)`**: Updates satellite
    positions based on their orbital elements to a new time using a vectorized
    Keplerian propagator.

### 2.3. `visibility.py`

*   **`solarexclusion(data_struct: Dict[str, Any])`**: Calculates solar exclusion for all
    satellites based on their pointing vectors. Returns a tuple containing an
    `exclusion_vector` (1 for excluded, 0 for clear) and an `angle_vector`
    (the calculated angle in radians for each satellite).
*   **`exclusion(data_struct: Dict[str, Any], satellite_index: int, print_debug: bool = False)`**: The
    primary function that checks for viewing exclusion. It takes the main
    simulation data structure and a satellite index and returns `0` if the
    satellite's view is excluded, and `1` otherwise. The optional `print_debug`
    flag enables detailed console output.
*   **`update_visibility_table(data_struct: Dict[str, Any], print_debug_for_sat: Optional[int] = None)`**:
    Creates a 2D NumPy array where rows correspond to satellites and columns
    correspond to fixed points. A cell value of 1 means the view is clear, and
    0 means it is excluded. The optional `print_debug_for_sat` argument can be
    used to enable debug printing for a specific satellite.

### 2.4. `pointing.py`

*   **`jerk(data_struct: Dict[str, Any], satellite_number: int)`**: Moves the pointing vector of a
    specific satellite by 0.3 radians in a random direction.
*   **`find_and_jerk_blind_satellites(data_struct: Dict[str, Any])`**: Finds satellites with no
    visibility and applies the 'jerk' function to them.
*   **`pointing_place_update(data_struct: Dict[str, Any])`**: Increments the pointing place for
    all satellites, wrapping around if necessary.
*   **`generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int)`**: Generates a pointing
    sphere with n_points and stores it in the data_struct.
*   **`update_satellite_pointing(data_struct: Dict[str, Any])`**: Updates the pointing vector
    for each satellite based on its pointing state.

### 2.5. Plotting Modules

This module contains functions for creating interactive 3D plots of the
simulation state using the `plotly` library.

*   **`plotting_3d.plot_3d_scatter(positions: np.ndarray, title: str, plot_time: datetime, labels: Optional[List[str]] = None, marker_size: int = 1, trace_name: str = 'Points')`**: The primary function for creating 3D scatter
    plots. It displays object positions with Earth references and allows for
    customization of the marker size and trace name.
*   **`plotting_vectors.plot_pointing_vectors(data_struct: Dict[str, Any], title: str, plot_time: datetime)`**:
    Displays a 3D plot of satellites along with vectors indicating their
    pointing direction.

### 2.6. `pointing_vectors.py`

*   **`pointing_vectors(n: int)`**: Generates `n` equally spaced points on a unit
    sphere using the Fibonacci lattice algorithm.
*   **`plot_vectors_on_sphere(vectors: np.ndarray, title: str)`**: Creates a 3D plot of vectors
    on a sphere.

### 2.7. Demos

The `demo*.py` scripts showcase the toolkit's capabilities:
*   **`demo1`**: Initializes a standard simulation, propagates all satellites by
    1.5 hours, and plots their final positions.
*   **`demo2`**: Plots satellite positions at T=0 and T=300s, and includes
    vectors indicating the direction to the Sun and Moon at both times.
*   **`demo3`**: Plots the trajectory of a single LEO satellite over 90
    minutes.
*   **`demo4`**: Plots the trajectory of a single GEO satellite over 23 hours.
*   **`demo_constellation`**: Demonstrates the creation of a GEO satellite
    constellation.
*   **`demo_exclusion_table`**: Calculates the visibility of fixed points for
    all satellites and displays the result as a heatmap.

*   **`demo_fixedpoints`**: Visualizes the distribution of the generated "fixed
    points" (observation targets) in a 3D scatter plot.
*   **`show_geo_search`**: A demo that performs a geometric search for satellites using a GEO constellation, and generates several plots to visualize the satellite pointing updates and the RA/Dec history of one satellite.
*   **`demo_lambertian`**: Demonstrates the `lambertiansphere` brightness
    calculation and plots brightness vs. phase angle.
*   **`demo_pointing_plot`**: Shows a 3D plot of all satellites with their
    pointing vectors.
*   **`demo_pointing_vectors`**: Generates 1000 uniformly distributed pointing
    vectors and plots them on a sphere.
*   **`demo_sky_scan`**: Simulates a sky scan from a GEO satellite, mapping out
    the celestial exclusion zones as a heatmap.
*   **`demo_pointing_sequence`**: Demonstrates the satellite pointing sequence
    functionality, showing how satellites can step through a pre-defined grid of
    pointing vectors.

### 2.8. How to Run Demos

The `all_demos.py` script provides a comprehensive demonstration of the toolkit's features.

#### 1. Run Demos from the Command Line

You can run the script directly from your terminal to see the plots displayed in your browser:

```bash
python all_demos.py
```

#### 2. Run Demos in a Jupyter Notebook

You can import and run the `run_all_demos` function from a Jupyter Notebook to display all plots inline.

```python
import all_demos
all_demos.run_all_demos()
```

To save all the demo plots to a single HTML file named `all_demo_plots.html`, run the following from a Python script or notebook:

```python
import all_demos
all_demos.run_all_demos(save_html=True)
```

### 2.9. `demo_common.py`

*   **`initialize_standard_simulation(start_time)`**: A helper function that
    sets up a standard simulation scenario. It loads a predefined set of TLEs
    from `standard_tle.txt`, initializes the data structures, and propagates
    all satellites to the given start time. This is the recommended starting
    point for most simulations and is used by all demo scripts.

### 2.10. `constants.py`

This module centralizes the definition of numerical constants used throughout
the toolkit, particularly for array indexing. This improves readability and
maintainability by replacing "magic numbers" with descriptive names. Key
constants include:
*   `EARTH_RADIUS`, `MOON_RADIUS`: Radii in meters.
*   `DETECTOR_*_IDX`: Column indices for the `detector` NumPy array.
*   `ORBITAL_*_IDX`: Column indices for the `orbital_elements` NumPy array.
*   `POINTING_*_IDX`: Column indices for the `pointing_state` NumPy array.

### 2.11. `radiometry_calcs.py`

*   **`mag(x: float)`**: Converts a linear flux ratio to an astronomical magnitude.
*   **`amag(x: float)`**: Converts an astronomical magnitude back to a linear flux
    ratio.
*   **`blackbody_flux(temperature: float, lambda_short: float, lambda_long: float)`**: Computes the
    integrated spectral radiance of a blackbody over a wavelength band.
*   **`stefan_boltzmann_law(temperature: float)`**: Calculates the total power radiated
    per unit area by a blackbody.
*   **`plot_blackbody_spectrum(temperature: float)`**: Plots the spectral radiance of a
    blackbody from 0.5 to 30 microns.
*   **`plot_blackbody_spectrum_visible_nir(temperature: float)`**: Plots the spectral
    radiance of a blackbody from 0.1 to 1 micron.

### 2.12. `lambertian.py`

*   **`lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: float, radius: float)`**: Calculates the effective
    brightness cross-section (in square meters) of a diffusely reflecting
    (Lambertian) sphere based on illumination geometry, albedo, and size.

### 2.13. `generate_log_spherical_points.py`

*   **`generate_log_spherical_points(num_points: int, inner_radius: float, outer_radius: float, object_size_m: float = 1.0, seed: int = None)`**: Generates a set of 3D points with logarithmic
    radial and uniform angular distribution. Returns a tuple containing the
    points array and a sizes array.

### 2.14. `constellation.py`

*   **`geos(sim_data: Dict[str, Any], n: int, fov: float)`**: Creates `n` equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

### 2.15. `generate_report.py`

*   This script generates a PDF report of the project.

## 3. Dependencies

To run the VibeVolts code, the following Python modules must be installed. You
can install them using pip.

*   **`numpy`**: For numerical operations and array manipulation.
*   **`astropy`**: For astronomical calculations and coordinate transformations.
*   **`jplephem`**: Used by `astropy` for planetary ephemeris calculations.
*   **`sgp4`**: For parsing TLE satellite data.
*   **`plotly`**: For creating interactive 3D plots.
*   **`scipy`**: For scientific computations, specifically numerical integration
    in `radiometry.py`.
*   **`ipython`**: For displaying plots inline in Jupyter notebooks.

Example installation command:
```bash
pip install numpy astropy jplephem sgp4 plotly scipy ipython
```
