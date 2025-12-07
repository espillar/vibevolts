# VibeVolts Gemini Documentation

## Project OvervieW



VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets. These are intended be evolved in a discrete event simulation. 



## Data Structure

There is a "global" data structure 

The current state of the simulation is stored in a dictionary typically called sim_data which is passed between the routines that initialize and operate on the components to initialize, evolve, and interrogate the overall system.  The different components of the system are typically dealt with by different modules.   

In order to maximize the use of numpy and other parallel tools,  components like satellite elements are typically stored in arrays comprising all of the satellites in one numpy array, in this way efficient parallel numpy routines can easily be leveraged.

Here is current trace of what the structure looks like and how it evolves.



 This data structure is initialized and updated by a set of functions that are organized into the following modules:

*   **`simulation.py`**: Defines the functions to create the basic simulation data structure and to add celestial bodies and fixed points.
*   **`propagation.py`**: Handles orbit propagation, celestial mechanics, and adding satellites from TLE files.
*   **`observatories.py`**: Defines functions to add ground-based observatories to the simulation.
*   **`constellation.py`**: Defines functions for creating satellite constellations. The `geos` function in this module creates a constellation of GEO satellites and adds them to the main 'satellites' group.
*   **`visibility.py`**: Performs line-of-sight and exclusion calculations.
*   **`pointing.py`**: Manages satellite pointing control.
*   **`lambertian.py`**: Calculates Lambertian sphere brightness.
*   **`radiometry_data.py` & `radiometry_calcs.py`**: Provide radiometric data and functions.
*   **`plotting_3d.py` & `plotting_vectors.py`**: Contain 3D visualization functions.
*   **`pointing_vectors.py`**: Includes functions for generating and visualizing uniformly distributed vectors on a sphere.
*   **`generate_log_spherical_points.py`**: Provides tools for generating 3D point clouds.
*   **`demo_common.py`**: A utility module that provides helper functions for the demo scripts.
*   **`demo_constellation.py`**: A demo script for creating and visualizing satellite constellations.
*   **`demo_show_geo_search.py`**: A demo script that demonstrates a geometric search for satellites using a GEO constellation, and generates several plots to visualize the satellite pointing updates and the RA/Dec history of one satellite.
*   **`demo_pointing_sequence.py`**: A demo script for demonstrating the satellite pointing sequence functionality.
*   **`demo_sky_scan.py`**: A demo script for simulating a sky scan from a satellite.
*   **`generate_report.py`**: A script for generating a PDF report of the project.

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


## Documentation

## Development Conventions
*  **Text Files**: All purely text file have lines that are no more than 100 charcters long to aid reading.
*   **Data Structures**: The simulation state is managed in a central dictionary. This dictionary is initialized as a minimal structure using `create_empty_simulation` from `simulation.py`. Components like satellites, observatories, and celestial bodies are then added incrementally using dedicated functions (e.g., `add_satellites_from_tle`, `add_observatories`), making the structure highly modular and flexible.
*   **Modularity**: The code is organized into modules, each with a specific responsibility. This makes the code easy to understand, maintain, and extend.
*   **Vectorization**: The code makes extensive use of NumPy for vectorized operations, which provides a significant performance improvement over iterating through lists.
*   **Type Hinting**: The code uses type hints to improve readability and allow for static analysis.
*   **Docstrings**: All functions have docstrings that explain their purpose, arguments, and return values.
*   **Constants**: Constants are defined in `constants.py` to avoid magic numbers in the code.
