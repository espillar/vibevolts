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
*   `generate_log_spherical_points.py`: Tools for generating 3D point clouds.

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
