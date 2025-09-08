# VibeVolts Documentation

This document provides an overview of the data structures, functions, and dependencies for the VibeVolts simulation toolkit.

## HTML Wiki

A pure HTML version of this documentation is available in the file [wiki.html](wiki.html).

## 1. Common Data Structures

The toolkit uses two primary data structures to manage simulation state and physical constants.

### 1.1. Simulation State Dictionary (`simulation_data`)

This is the central data structure, created by the `initializeStructures` function in `simulation.py`. It is a Python dictionary that organizes all simulation entities into categories.

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
        'detector': np.zeros((num_satellites, 7)),
    },
    'observatories': { ... },
    'red_satellites': { ... },
    'fixedpoints': {
        'position': np.zeros((num_points, 3)),
        'size': np.zeros((num_points,))
    },
    'pointing_spheres': {},
    'delta_time': 60.0
}
```

#### Key Components:

*   **`orbital_elements`**: A NumPy array (`n x 6`) containing the classical orbital elements for each satellite. The columns are:
    *   `0`: Semi-major axis (meters)
    *   `1`: Eccentricity
    *   `2`: Inclination (radians)
    *   `3`: Right Ascension of the Ascending Node (radians)
    *   `4`: Argument of Perigee (radians)
    *   `5`: Mean Anomaly (radians)

*   **`detector`**: A NumPy array (`n x 7`) containing the properties of each sensor. The columns are:
    *   `0`: Aperture size (meters)
    *   `1`: Pixel size (radians)
    *   `2`: Quantum Efficiency (fraction)
    *   `3`: Pixel count
    *   `4`: Solar exclusion angle (radians)
    *   `5`: Lunar exclusion angle (radians)
    *   `6`: Earth exclusion angle (radians)

*   **`fixedpoints`**: A dictionary containing the properties of the static points in space used as observation targets.
    *   `position`: A NumPy array (`num_points x 3`) of static 3D points in the GCRS frame.
    *   `size`: A NumPy array (`num_points`,) of the size of each object in meters.
*   **`pointing_state`**: A NumPy array (`n x 2`) containing the pointing control state for each satellite.
    *   `0`: Pointing Count (number of points in the pointing grid)
    *   `1`: Pointing Place (current index in the pointing sequence)

### 1.2. Radiometric Filter Data (`FILTER_DATA`)

This dictionary, located in `radiometry_data.py`, provides standard data for a variety of astronomical filters, including Johnson-Cousins, SDSS, and JWST.

*   **`sun`**: The apparent magnitude of the Sun in the given filter.
*   **`sky`**: The typical dark sky brightness in magnitudes per square arcsecond.
*   **`central_wavelength`**: The central wavelength of the filter passband in nanometers (nm).
*   **`bandwidth`**: The effective width of the filter passband in nanometers (nm).
*   **`zero_point`**: The photon flux (in photons/sec/m²) corresponding to a 0-magnitude star.

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

*   **`initializeStructures(num_satellites, num_observatories, num_red_satellites, start_time)`**: Creates and returns the main `simulation_data` dictionary.

### 2.2. `propagation.py`

*   **`celestial_update(data_struct, time_date)`**: Updates the positions of the Sun and Moon for a given time using the `astropy` library.
*   **`readtle(tle_file_path)`**: Reads a Two-Line Element (TLE) file and returns a NumPy array of orbital elements and a list of epoch datetimes.
*   **`propagate_satellites(data_struct, time_date)`**: Updates satellite positions based on their orbital elements to a new time using a vectorized Keplerian propagator.

### 2.3. `visibility.py`

*   **`solarexclusion(data_struct)`**: Calculates solar exclusion for all satellites based on their pointing vectors. Returns a tuple containing an `exclusion_vector` (1 for excluded, 0 for clear) and an `angle_vector` (the calculated angle in radians for each satellite).
*   **`exclusion(data_struct, satellite_index)`**: The primary function that checks for viewing exclusion. It takes the main simulation data structure and a satellite index and returns `0` if the satellite's view is excluded, and `1` otherwise.
*   **`update_visibility_table(data_struct)`**: Creates a 2D NumPy array where rows correspond to satellites and columns correspond to fixed points. A cell value of 1 means the view is clear, and 0 means it is excluded.

### 2.4. `pointing.py`

*   **`jerk(data_struct, satellite_number)`**: Moves the pointing vector of a specific satellite by 0.3 radians in a random direction.
*   **`find_and_jerk_blind_satellites(data_struct)`**: Finds satellites with no visibility and applies the 'jerk' function to them.
*   **`pointing_place_update(data_struct)`**: Increments the pointing place for all satellites, wrapping around if necessary.
*   **`generate_pointing_sphere(data_struct, n_points)`**: Generates a pointing sphere with n_points and stores it in the data_struct.
*   **`update_satellite_pointing(data_struct)`**: Updates the pointing vector for each satellite based on its pointing state.

### 2.5. Plotting Modules

This module contains functions for creating interactive 3D plots of the simulation state using the `plotly` library.

*   **`plotting_3d.plot_3d_scatter(positions, title, plot_time, labels, marker_size, trace_name)`**: The primary function for creating 3D scatter plots. It displays object positions with Earth references and allows for customization of the marker size and trace name.
*   **`plotting_vectors.plot_pointing_vectors(data_struct, title, plot_time)`**: Displays a 3D plot of satellites along with vectors indicating their pointing direction.

### 2.6. `pointing_vectors.py`

*   **`pointing_vectors(n)`**: Generates `n` equally spaced points on a unit sphere using the Fibonacci lattice algorithm.
*   **`plot_vectors_on_sphere(vectors, title)`**: Creates a 3D plot of vectors on a sphere.

### 2.7. Demos

The `demo*.py` scripts showcase the toolkit's capabilities:
*   **`demo1`**: Initializes a standard simulation, propagates all satellites by 1.5 hours, and plots their final positions.
*   **`demo2`**: Plots satellite positions at T=0 and T=300s, and includes vectors indicating the direction to the Sun and Moon at both times.
*   **`demo3`**: Plots the trajectory of a single LEO satellite over 90 minutes.
*   **`demo4`**: Plots the trajectory of a single GEO satellite over 23 hours.
*   **`demo_exclusion_table`**: Calculates the visibility of fixed points for all satellites and displays the result as a heatmap.
*   **`demo_exclusion_debug_print`**: A non-plotting demo that shows the detailed debug output of the `exclusion` function for a single satellite.
*   **`demo_fixedpoints`**: Visualizes the distribution of the generated "fixed points" (observation targets) in a 3D scatter plot.
*   **`demo_lambertian`**: Demonstrates the `lambertiansphere` brightness calculation and plots brightness vs. phase angle.
*   **`demo_pointing_plot`**: Shows a 3D plot of all satellites with their pointing vectors.
*   **`demo_pointing_vectors`**: Generates 1000 uniformly distributed pointing vectors and plots them on a sphere.
*   **`demo_sky_scan`**: Simulates a sky scan from a GEO satellite, mapping out the celestial exclusion zones as a heatmap.
*   **`demo_pointing_sequence`**: Demonstrates the satellite pointing sequence functionality, showing how satellites can step through a pre-defined grid of pointing vectors.

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

### 2.9. `radiometry_calcs.py`

*   **`mag(x)`**: Converts a linear flux ratio to an astronomical magnitude.
*   **`amag(x)`**: Converts an astronomical magnitude back to a linear flux ratio.
*   **`blackbody_flux(temperature, lambda_short, lambda_long)`**: Computes the integrated spectral radiance of a blackbody over a wavelength band.
*   **`stefan_boltzmann_law(temperature)`**: Calculates the total power radiated per unit area by a blackbody.
*   **`plot_blackbody_spectrum(temperature)`**: Plots the spectral radiance of a blackbody from 0.5 to 30 microns.
*   **`plot_blackbody_spectrum_visible_nir(temperature)`**: Plots the spectral radiance of a blackbody from 0.1 to 1 micron.

### 2.10. `lambertian.py`

*   **`lambertiansphere(vec_from_sphere_to_light, vec_from_sphere_to_observer, albedo, radius)`**: Calculates the effective brightness cross-section (in square meters) of a diffusely reflecting (Lambertian) sphere based on illumination geometry, albedo, and size.

### 2.11. `generate_log_spherical_points.py`

*   **`generate_log_spherical_points(num_points, inner_radius, outer_radius, object_size_m, seed)`**: Generates a set of 3D points with logarithmic radial and uniform angular distribution. Returns a tuple containing the points array and a sizes array.

## 3. Dependencies

To run the VibeVolts code, the following Python modules must be installed. You can install them using pip.

*   **`numpy`**: For numerical operations and array manipulation.
*   **`astropy`**: For astronomical calculations and coordinate transformations.
*   **`jplephem`**: Used by `astropy` for planetary ephemeris calculations.
*   **`sgp4`**: For parsing TLE satellite data.
*   **`plotly`**: For creating interactive 3D plots.
*   **`scipy`**: For scientific computations, specifically numerical integration in `radiometry.py`.

Example installation command:
```bash
pip install numpy astropy jplephem sgp4 plotly scipy
```
