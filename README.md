# VibeVolts

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets.

## Key Features

*   **Data Structures:** A comprehensive data structure to manage simulation state.
*   **Satellite Propagation:** Propagate satellite orbits from TLE data.
*   **Astronomical Calculations:** Accurate celestial body positioning using `astropy`.
*   **Radiometry:** Includes data for standard astronomical filters and functions for radiometric calculations.
*   **Exclusion Analysis:** Functions to determine if a satellite's line of sight is obstructed.
*   **3D Visualization:** Interactive 3D plotting of simulation scenarios using `plotly`.
*   **Point Generation:** Tools to generate 3D point clouds with specific distributions.

## Core Modules

The toolkit has been refactored into a modular structure:

*   `simulation.py`: Core data structures and initialization.
*   `propagation.py`: Orbit propagation and celestial mechanics.
*   `visibility.py`: Line-of-sight and exclusion calculations.
*   `pointing.py`: Satellite pointing control.
*   `lambertian.py`: Lambertian sphere brightness calculations.
*   `radiometry_data.py` & `radiometry_calcs.py`: Radiometric data and functions.
*   `plotting_3d.py` & `plotting_vectors.py`: 3D visualization functions.
*   `pointing_vectors.py`: Functions for generating and visualizing uniformly distributed vectors on a sphere.
*   `generate_log_spherical_points.py`: Tools for generating 3D point clouds.

## Demos

The `all_demos.py` script runs a series of demonstrations to showcase the toolkit's features. Here is a list of the available demos:

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

## Dependencies

VibeVolts requires the following Python libraries:

*   `numpy`
*   `astropy`
*   `jplephem`
*   `sgp4`
*   `plotly`
*   `scipy`

You can install them using pip:

```bash
pip install numpy astropy jplephem sgp4 plotly scipy
```

## Usage and Demos

The `all_demos.py` script provides a comprehensive demonstration of the toolkit's features.

### 1. Run Demos from the Command Line

You can run the script directly from your terminal to see the plots displayed in your browser:

```bash
python all_demos.py
```

### 2. Run Demos in a Jupyter Notebook

You can import and run the `run_all_demos` function from a Jupyter Notebook to display all plots inline.

```python
import all_demos
all_demos.run_all_demos()
```

## Documentation

For detailed documentation of all data structures, functions, and modules, please refer to `vvdocs.md` and `vvdocs.org`.
