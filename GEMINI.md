# VibeVolts Gemini Documentation

## Project Overview



VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets. These are intended be evolved in a discrete event simulation. 



## Documentation

Gemini should maintain a document file vibevolts.md as follows. 

The first section should list the main data structure dictionary, and all the python files should be reviewed for functions which add data to the dictionary if called.  The dictionary items added should be listed, the functions that add or modify each of them should be listed next to them, and any inline documentation should be listed along with a function signature.

The next section should list the demos in the python files.

The next section list each of the python files, in a subsection, and in subsubsections list each of the functions inthe python file along with a signature for each function and any documentaiton.

The next section should list dependencies of the code.



## Data Structure

There is a "global" data structure 

The current state of the simulation is stored in a dictionary typically called sim_data which is passed between the routines that initialize and operate on the components to initialize, evolve, and interrogate the overall system.  The different components of the system are typically dealt with by different modules.   

In order to maximize the use of numpy and other parallel tools,  components like satellite elements are typically stored in arrays comprising all of the satellites in one numpy array, in this way efficient parallel numpy routines can easily be leveraged.

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

### scansensors.py
#### scansensors(sim_data: dict)
```python
def scansensors(sim_data: dict)
```
Scans for and processes sensor data within the simulation.

Args:
    sim_data (dict): The main simulation data dictionary.
                     This dictionary is expected to contain all
                     relevant simulation state and parameters.

## Development Conventions
*  **Text Files**: All purely text file have lines that are no more than 100 charcters long to aid reading.
*   **Data Structures**: The simulation state is managed in a central dictionary. This dictionary is initialized as a minimal structure using `create_empty_simulation` from `simulation.py`. Components like satellites, observatories, and celestial bodies are then added incrementally using dedicated functions (e.g., `add_satellites_from_tle`, `add_observatories`), making the structure highly modular and flexible.
*   **Modularity**: The code is organized into modules, each with a specific responsibility. This makes the code easy to understand, maintain, and extend.
*   **Vectorization**: The code makes extensive use of NumPy for vectorized operations, which provides a significant performance improvement over iterating through lists.
*   **Type Hinting**: The code uses type hints to improve readability and allow for static analysis.
*   **Docstrings**: All functions have docstrings that explain their purpose, arguments, and return values.
*   **Constants**: Constants are defined in `constants.py` to avoid magic numbers in the code.
