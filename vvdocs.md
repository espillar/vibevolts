# VibeVolts Documentation

This document provides an overview of the data structures, functions, and dependencies for the VibeVolts simulation toolkit.

## 1. Common Data Structures

The toolkit uses two primary data structures to manage simulation state and physical constants.

### 1.1. Simulation State Dictionary (`simulation_data`)

This is the central data structure, created by the `initializeStructures` function in `vibevolts.py`. It is a Python dictionary that organizes all simulation entities into categories.

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
        'position': np.zeros((num_points, 3))
    }
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

*   **`fixedpoints`**: A dictionary containing a single key, `position`, which holds a NumPy array (`num_points x 3`) of static 3D points in the GCRS frame. These points are generated with a logarithmic radial distribution and are intended for fixed-point calculations or as a reference grid.

### 1.2. Radiometric Filter Data (`FILTER_DATA`)

This dictionary from `radiometry.py` contains standard data for various astronomical filters.

```python
{
    'U': {
        'sun': -26.03,              # Apparent magnitude of the Sun
        'sky': 22.0,                # Sky brightness (mag/arcsec^2)
        'central_wavelength': 365.0, # in nm
        'bandwidth': 66.0,          # in nm
        'zero_point': 4.96e9,       # Photon flux for mag=0 object
    },
    'B': { ... },
    # ... and so on for V, R, I, J, H, K, g, r, i, z, L, M, N, and JWST filters.
}
```

## 2. Existing Functions

This section describes the functions available in the toolkit, organized by module.

### 2.1. `vibevolts.py`

*   **`initializeStructures(num_satellites, num_observatories, num_red_satellites, start_time)`**: Creates and returns the main `simulation_data` dictionary.
*   **`celestial_update(data_struct, time_date)`**: Updates the positions of the Sun and Moon for a given time using the `astropy` library.
*   **`readtle(tle_file_path)`**: Reads a Two-Line Element (TLE) file and returns a NumPy array of orbital elements and a list of epoch datetimes.
*   **`propagate_satellites(data_struct, time_date)`**: Updates satellite positions based on their orbital elements to a new time using a vectorized Keplerian propagator.
*   **`plot_positions_3d(positions, title, plot_time, labels)`**: Displays an interactive 3D plot of object positions using `plotly`.
*   **`solarexclusion(data_struct)`**: Calculates solar exclusion for all satellites based on their pointing vectors.
*   **`demo1()`, `demo2()`, `demo3()`, `demo4()`, `demo5()`**: Demonstration functions that run pre-configured simulations and generate plots.
*   **`demo_fixedpoints()`**: Demonstrates the `fixedpoints` data structure by plotting it in 3D.

### 2.2. `radiometry.py`

*   **`mag(x)`**: Converts a linear flux ratio to an astronomical magnitude.
*   **`amag(x)`**: Converts an astronomical magnitude back to a linear flux ratio.
*   **`blackbody_flux(temperature, lambda_short, lambda_long)`**: Computes the integrated spectral radiance of a blackbody over a wavelength band.
*   **`stefan_boltzmann_law(temperature)`**: Calculates the total power radiated per unit area by a blackbody.
*   **`plot_blackbody_spectrum(temperature)`**: Plots the spectral radiance of a blackbody from 0.5 to 30 microns.
*   **`plot_blackbody_spectrum_visible_nir(temperature)`**: Plots the spectral radiance of a blackbody from 0.1 to 1 micron.

### 2.3. `lambertiansphere.py`

*   **`lambertiansphere(vec_from_sphere_to_light, vec_from_sphere_to_observer, albedo, radius)`**: Calculates the effective brightness cross-section (in square meters) of a diffusely reflecting (Lambertian) sphere based on illumination geometry, albedo, and size.

### 2.4. `exclusion.py`

This module provides functions to determine if a satellite's line of sight is obstructed by major celestial bodies (Sun, Moon, Earth).

*   **`exclusion(data_struct, satellite_index)`**: The primary function that checks for viewing exclusion. It takes the main simulation data structure and a satellite index and returns `True` if the satellite's pointing vector is within the exclusion zone of the Sun, Moon, or Earth, and `False` otherwise. The exclusion angles are retrieved from the satellite's `detector` properties.
*   **`test_exclusion_plot()`**: A demonstration and testing function that creates a scenario with 100 satellites with random orbits and pointing vectors. It runs the `exclusion` check on them and generates interactive 3D plots for the first 15 satellites to visually verify the results.

### 2.5. `generate_log_spherical_points.py`

*   **`generate_log_spherical_points(num_points, inner_radius, outer_radius, seed)`**: Generates a set of 3D points with logarithmic radial and uniform angular distribution.
*   **`visualize_point_distribution(points)`**: Visualizes the distribution of a 3D point cloud with four plots.

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
