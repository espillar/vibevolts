# VibeVolts Gemini Documentation

## Project Overview

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets. The toolkit is highly modular, with a clear separation of concerns between different components of the simulation.

The core of the simulation is a data structure that represents the state of the simulation at a given time. This data structure is initialized and updated by a set of functions that are organized into the following modules:

*   **`simulation.py`**: Defines the core data structures and initialization functions.
*   **`propagation.py`**: Handles orbit propagation and celestial mechanics.
*   **`visibility.py`**: Performs line-of-sight and exclusion calculations.
*   **`pointing.py`**: Manages satellite pointing control.
*   **`lambertian.py`**: Calculates Lambertian sphere brightness.
*   **`radiometry_data.py` & `radiometry_calcs.py`**: Provide radiometric data and functions.
*   **`plotting_3d.py` & `plotting_vectors.py`**: Contain 3D visualization functions.
*   **`pointing_vectors.py`**: Includes functions for generating and visualizing uniformly distributed vectors on a sphere.
*   **`generate_log_spherical_points.py`**: Provides tools for generating 3D point clouds.

## Building and Running

### Dependencies

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

## Development Conventions

*   **Data Structures**: The simulation state is managed in a central dictionary, which is passed to and modified by the various functions. This dictionary is defined in `simulation.py`.
*   **Modularity**: The code is organized into modules, each with a specific responsibility. This makes the code easy to understand, maintain, and extend.
*   **Vectorization**: The code makes extensive use of NumPy for vectorized operations, which provides a significant performance improvement over iterating through lists.
*   **Type Hinting**: The code uses type hints to improve readability and allow for static analysis.
*   **Docstrings**: All functions have docstrings that explain their purpose, arguments, and return values.
*   **Constants**: Constants are defined in `constants.py` to avoid magic numbers in the code.
