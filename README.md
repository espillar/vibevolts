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

## Documentation

For detailed documentation of all data structures, functions, and modules, please refer to `vvdocs.md` and `vvdocs.org`.
