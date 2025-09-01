# VibeVolts

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets.

## Key Features

*   **Data Structures:** A comprehensive data structure to manage simulation state, including celestial bodies, satellites, and observatories.
*   **Satellite Propagation:** Propagate satellite orbits from Two-Line Element (TLE) data using a vectorized Keplerian propagator.
*   **Astronomical Calculations:** Accurate celestial body positioning using `astropy`.
*   **Radiometry:** Includes data for standard astronomical filters and functions for radiometric calculations.
*   **Exclusion Analysis:** Functions to determine if a satellite's line of sight is obstructed by the Sun, Moon, or Earth.
*   **3D Visualization:** Interactive 3D plotting of simulation scenarios using `plotly`.
*   **Point Generation:** Tools to generate 3D point clouds with specific distributions for analysis.

## Core Modules

*   `vibevolts.py`: The main module containing core simulation functions.
*   `radiometry.py`: Radiometric data and conversion functions.
*   `lambertiansphere.py`: Functions for calculating the brightness of spherical objects.
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

The `vibevolts_demo.py` script provides a comprehensive demonstration of the toolkit's features. It is designed to be run in two ways:

### 1. As a Standalone Script

You can run the script directly from your terminal:

```bash
python vibevolts_demo.py
```

When executed, the script will:
- Run a series of demonstration scenarios.
- Print status information to the console.
- Generate a `demo_plots.html` file in the same directory. This is a self-contained HTML file with interactive 3D plots for each demonstration scenario.

### 2. As a Library in a Notebook or other Scripts

The demo functions (e.g., `demo1`, `demo2`) can be imported and used in other Python scripts or Jupyter notebooks. In this mode, the functions will **return** a `plotly.graph_objects.Figure` object but will **not** automatically display it.

This allows you to further customize the plot or integrate it into a larger analysis workflow.

Example in a Jupyter Notebook:

```python
from vibevolts_demo import demo1

# Generate the plot object
my_figure = demo1()

# You can now display it, modify it, or save it
my_figure.update_layout(title="My Custom Title")
my_figure.show()
```

## Documentation

For detailed documentation of all data structures, functions, and modules, please refer to `vvdocs.md` and `vvdocs.org`.
