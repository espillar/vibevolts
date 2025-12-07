# `sim_data` Data Structure Trace

This document traces the construction of the `sim_data` dictionary, a central data structure in the VibeVolts simulation toolkit. It outlines which modules contribute to this structure and what data they add. The dictionary is typically passed to functions under the variable name `sim_data`.

---

## `simulation.py`

**Description:** This module lays the foundation of the simulation data structure. It initializes the dictionary and adds fundamental simulation parameters, celestial bodies, and fixed ground points.

**Functions and Data Structure Additions:**

- **`create_empty_simulation(start_time_jd: float)`**: Initializes the main dictionary with core simulation parameters.
    - `sim_data['time']`: An `astropy.time.Time` object representing the simulation's current time.
    - `sim_data['time_jd']`: The Julian Date representation of the current time (`float`).
    - `sim_data['ephemeris']`: A `jplephem` object used for calculating planetary positions.
    - `sim_data['earth']`: A dictionary holding Earth-specific orientation and rotation data.

- **`add_celestial_bodies(sim_data: dict, body_names: list)`**: Adds major solar system bodies to the simulation.
    - `sim_data['celestial_bodies']`: A dictionary where each key is the name of a celestial body (e.g., `'sun'`, `'moon'`). The value is another dictionary containing the body's state vectors.
        - `body['pos']`: `numpy.ndarray` (3,) of the body's position in the GCRS frame.
        - `body['vel']`: `numpy.ndarray` (3,) of the body's velocity in the GCRS frame.

- **`add_fixed_points(sim_data: dict, points_lla: list)`**: Adds static points of interest on the Earth's surface.
    - `sim_data['fixed_points']`: A dictionary containing information about fixed locations.
        - `'ecf'`: `numpy.ndarray` (Nx3) of the points' Earth-Centered Fixed (ECF) coordinates.
        - `'names'`: `list` of strings with the names for each point.

---

## `propagation.py`

**Description:** This module is responsible for adding satellites to the simulation from Two-Line Element (TLE) sets and propagating their orbits.

**Functions and Data Structure Additions:**

- **`add_satellites_from_tle(sim_data: dict, tle_file: str)`**: Adds a group of satellites to the simulation based on data from a TLE file.
    - `sim_data['satellites']`: A dictionary that holds the state and properties of all satellites. For performance, the data is stored in `numpy` arrays where the first dimension `N` corresponds to the number of satellites.
        - `'names'`: `list` of satellite names.
        - `'epoch'`: An `astropy.time.Time` object for the TLE epoch of the satellite group.
        - `'pos'`: `numpy.ndarray` (Nx3) of satellite positions in the GCRS frame (ECI).
        - `'vel'`: `numpy.ndarray` (Nx3) of satellite velocities in the GCRS frame (ECI).
        - `'sgp4'`: `list` of `sgp4.api.Satrec` objects, one for each satellite, used for propagation.

---

## `observatories.py`

**Description:** This module adds ground-based observatories to the simulation.

**Functions and Data Structure Additions:**

- **`add_observatories(sim_data: dict, obs_lla: list)`**: Adds a set of ground observatories.
    - `sim_data['observatories']`: A dictionary containing the positions and names of the observatories.
        - `'names'`: `list` of observatory names.
        - `'ecf'`: `numpy.ndarray` (Nx3) of observatory positions in Earth-Centered Fixed (ECF) coordinates.
        - `'lla'`: `numpy.ndarray` (Nx3) of observatory positions in Latitude, Longitude, Altitude.

---

## `constellation.py`

**Description:** This module provides functions to create common satellite constellations. It uses functions from `propagation.py` to add satellites to the `sim_data` structure.

**Functions and Data Structure Additions:**

- **`geos(sim_data: dict)`**: This function generates a constellation of 24 geostationary satellites. It does not add new keys to `sim_data` but populates the `sim_data['satellites']` dictionary by calling `add_satellites_from_tle`.

---

## `pointing.py`

**Description:** This module manages the pointing control for satellites, updating their orientation in space.

**Functions and Data Structure Additions:**

- **`update_pointing(sim_data: dict, pointing_vectors: dict)`**: Updates the primary pointing vector for each satellite.
    - `sim_data['satellites']['primary_pointing_vector']`: An `numpy.ndarray` (Nx3) representing the direction each satellite's instrument is pointing in the GCRS frame. This key is added to the existing `satellites` dictionary.

---

## `targets.py`

**Description:** This module is responsible for adding a set of fixed target points within the simulation space.

**Functions and Data Structure Additions:**

- **`add_fixed_points(sim_data: dict, num_points: int)`**: Adds a group of fixed points (targets) to the simulation.
    - `sim_data['counts']['fixedpoints']`: An `int` representing the number of fixed points created.
    - `sim_data['fixedpoints']`: A dictionary holding the properties of the fixed points.
        - `'position'`: `numpy.ndarray` (Nx3) of target positions in the GCRS frame.
        - `'exclusion'`: `numpy.ndarray` (Nx0) an empty array that can be resized later to store exclusion data.

---

## Modules for Calculation (No `sim_data` Modification)

The following modules contain functions that primarily perform calculations using data from `sim_data` but do not modify the structure itself. Their outputs are typically returned directly to the caller.

- **`visibility.py`**: Calculates line-of-sight visibility between assets.
- **`lambertian.py`**: Calculates the Lambertian brightness of satellites.
- **`radiometry_calcs.py`**: Performs radiometric calculations.