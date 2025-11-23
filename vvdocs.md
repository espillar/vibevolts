# 1VibeVolts Documentation

This document provides an overview of the data structures, functions, and dependencies for the VibeVolts simulation toolkit.

## Project Overview

VibeVolts is a Python-based simulation toolkit for space environment modeling. It provides a set of tools to initialize, propagate, and analyze the state of various space-based and ground-based assets. The toolkit is highly modular, with a clear separation of concerns between different components of the simulation.

The core of the simulation is a data structure that represents the state of the simulation at a given time. This data structure is initialized and updated by a set of functions that are organized into the following modules:

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
*   **`show_geo_search.py`**: A demo script that demonstrates a geometric search for satellites using a GEO constellation, and generates several plots to visualize the satellite pointing updates and the RA/Dec history of one satellite.
*   **`demo_pointing_sequence.py`**: A demo script for demonstrating the satellite pointing sequence functionality.
*   **`demo_sky_scan.py`**: A demo script for simulating a sky scan from a satellite.
*   **`generate_report.py`**: A script for generating a PDF report of the project.

## Data Structure Initialization

The `sim_data` dictionary is built incrementally. The following functions are responsible for initializing the different parts of the data structure:

*   **`create_empty_simulation`** (in `simulation.py`): Initializes the top-level `sim_data` dictionary with `start_time`, `delta_time`, `counts`, and `pointing_spheres`.
*   **`add_celestial_bodies`** (in `simulation.py`): Adds the `celestial` dictionary to `sim_data`.
*   **`add_fixed_points`** (in `simulation.py`): Adds the `fixedpoints` dictionary to `sim_data`.
*   **`add_satellites_from_tle`** (in `propagation.py`): Adds a satellite category dictionary (e.g., `satellites`) to `sim_data`.
*   **`add_observatories`** (in `observatories.py`): Adds the `observatories` dictionary to `sim_data`.
*   **`geos`** (in `constellation.py`): Adds a `satellites` dictionary for a GEO constellation to `sim_data`.
*   **`generate_pointing_sphere`** (in `pointing.py`): Adds a pointing sphere to the `pointing_spheres` dictionary in `sim_data`.
*   **`makeBlankDetector`** (in `detector.py`): Creates a blank detector array.
*   **`makeDetector`** (in `detector.py`): Creates a detector array with specified parameters.


## Modules and Functions

*   **`all_demos.py`**: 
    *   `run_all_demos(save_html=False)`: Runs all demo functions, and either shows them inline or saves them to a single HTML file.
        ```python
        """
        Runs all demo functions, and either shows them inline or saves them to a single HTML file.
        
        Args:
            save_html (bool): If True, saves plots to HTML. If False, displays plots inline.
        """
        ```

*   **`constants.py`**: This module contains global constants for array indices and physical constants.

*   **`constellation.py`**: 
    *   `geos(sim_data, n, fov)`: Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.
        ```python
        """
        Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

    Args:
        sim_data: The main simulation data dictionary.
        n: The number of satellites to create.
        fov: The diameter of the field of view of the satellite in radians.
    """
        ```
    *   `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag)`: Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.
        ```python
        """
        Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

    Args:
        sim_data: The main simulation data dictionary.
        n: The number of satellites to create.
        fov: The diameter of the field of view of the satellite in radians.
    """
        ```

*   **`demo_common.py`**: 
    *   `initialize_standard_simulation(start_time: datetime)`: Initializes a standard simulation with a predefined set of satellites.
        ```python
        """
        Initializes a standard simulation with a predefined set of satellites.

    This function consolidates TLE data, initializes the main data structure,
    and propagates satellites to their initial positions at the specified start
    time. This ensures the returned simulation state is fully populated with
    positions and radially-outward pointing vectors.

    Args:
        start_time: The timezone-aware datetime object for the simulation start.

    Returns:
        The fully initialized and propagated simulation data dictionary.
    """
        ```

*   **`demo_constellation.py`**: 
    *   `demo_constellation()`: Runs a demonstration of the constellation creation tools.
        ```python
        """
        Runs a demonstration of the constellation creation tools.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object for the satellite positions plot.
    """
        ```

*   **`demo_exclusion_debug_print.py`**: 
    *   `demo_exclusion_debug_print()`: Demonstrates the debug printing feature of the exclusion function.
        ```python
        """
        Demonstrates the debug printing feature of the exclusion function.

    This function runs the exclusion check for the first satellite against
    the first 100 fixed points and prints the detailed debug output for
    each check, as enabled by the `print_debug_for_sat` parameter.
    """
        ```

*   **`demo_exclusion_table.py`**: 
    *   `demo_exclusion_table()`: Demonstrates the creation and visualization of the exclusion table.
        ```python
        """
        Demonstrates the creation and visualization of the exclusion table.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
        ```

*   **`demo_fixedpoints.py`**: 
    *   `demo_fixedpoints()`: Demonstrates the fixedpoints data structure by plotting it in 3D.
        ```python
        """
        Demonstrates the fixedpoints data structure by plotting it in 3D.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
        ```

*   **`demo_lambertian.py`**: 
    *   `demo_lambertian()`: Runs a demonstration of the lambertiansphere function, including example calculations and a plot.
        ```python
        """
        Runs a demonstration of the lambertiansphere function,
        including example calculations and a plot.
        """
        ```

*   **`demo_pointing_plot.py`**: 
    *   `demo_pointing_plot()`: Demonstrates the plot_pointing_vectors function.
        ```python
        """
        Demonstrates the plot_pointing_vectors function.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
        ```

*   **`demo_pointing_sequence.py`**: 
    *   `demo_pointing_sequence()`: Demonstrates the satellite pointing sequence functionality.
        ```python
        """
        Demonstrates the satellite pointing sequence functionality.
        """
        ```

*   **`demo_pointing_vectors.py`**: 
    *   `demo_pointing_vectors()`: Demonstrates the generation and plotting of pointing vectors.
        ```python
        """
        Demonstrates the generation and plotting of pointing vectors.

    Returns:
        A Plotly figure object.
    """
        ```

*   **`demo_requiredIntegrationTime.py`**: 
    *   `demo_requiredIntegrationTime()`: Demonstrates the requiredIntegrationTime function.
        ```python
        """
        Demonstrates the requiredIntegrationTime function.
        Returns a graph
        """
        ```

*   **`demo_sky_scan.py`**: 
    *   `demo_sky_scan()`: Performs a sky scan from a GEO satellite to map celestial exclusion zones.
        ```python
        """
        Performs a sky scan from a GEO satellite to map celestial exclusion zones.
        """
        ```

*   **`demo1.py`**: 
    *   `demo1()`: Runs a full demonstration of the simulation tools.
        ```python
        """
        Runs a full demonstration of the simulation tools.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object for the satellite positions plot.
    """
        ```

*   **`demo2.py`**: 
    *   `demo2()`: Runs a demonstration plotting satellite and celestial positions.
        ```python
        """
        Runs a demonstration plotting satellite and celestial positions.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
        ```

*   **`demo3.py`**: 
    *   `demo3()`: Runs a demonstration plotting a single LEO satellite trajectory.
        ```python
        """
        Runs a demonstration plotting a single LEO satellite trajectory.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
        ```

*   **`demo4.py`**: 
    *   `demo4()`: Runs a demonstration plotting a single GEO satellite trajectory.
        ```python
        """
        Runs a demonstration plotting a single GEO satellite trajectory.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
        ```

*   **`detector.py`**: 
    *   `makeBlankDetector(n)`: Creates a blank detector array.
    *   `makeDetector(n, band, fov, ifov, aper, qe=0.5, photfrac=0.7, solarex=20 * DEGREE, lunarex=10 * DEGREE, earthex=15 * DEGREE)`: Creates a detector array with the given parameters.
        ```python
        '''
        makeDetector takes parameters of a sensor and stuffs a filter array and a detector array, which it returns.
        n is the number of sensors to produce
        band is the band the measurement takes place in (see radiometry_data)
        fov is the field of view- assumed square- in radians
        ifov is the pixel fov - assumed square - in radians
        aper is the aperture diameter - assumed round - in meters
        qe is th quantum efficiency of the system from entrance aperture
        to detectro
        photfrac is the fraction of the light captured in the photometry aperture
        solarex is the solar exclusion angle in radians
        lunarex is the lunar exclusion angle in radians
        earthex is the earth exclusion angle in radians

    This function is called when a new satellite is created.
    It uses the data from FILTER_DATA in radiometry_data.py, which is
    often in units of magnitudes,and 

    THIS VERSION IS FOR A GROUND OBSERVAOTRY
    '''
        ```
    *   `requiredIntegrationTime(limitingMag, SNR, filt, d, debug=0)`: Calculates the required integration time to achieve a given limiting magnitude.
        ```python
        '''
        requiredIntegrationTime(limitingMag, d)
        takes a two dimensional detector array ("detect")and calculates
          all the integration tiemes
        and returns those as a vector.
        For comparison with the equations paper, we first extract the variables
        to the conventional names used in that paper.
        '''
        ```
    *   `testdetector()`: A test function for the detector module.
        ```python
        '''
        testdetector creates an example that can be compared
        with some of the stuff in Curio.
        '''
        ```

*   **`fibonacciSearch.py`**:
    *   `searchStruct(detect)`: Creates the data structure for each of the satellite detectors.
        ```python
        '''
        creates the data structure for each of the satellite detectors,
        adds the structure to the detector
        '''
        ```

*   **`generate_log_spherical_points.py`**: 
    *   `generate_log_spherical_points(num_points: int, inner_radius: float, outer_radius: float, object_size_m: float=1.0, seed: int=None)`: Generates 3D points with logarithmic radial and uniform angular distribution.
        ```python
        """
        Generates 3D points with logarithmic radial and uniform angular distribution.

    This function creates a point cloud where point distances from the origin are
    logarithmically spaced. On any given spherical shell, points are distributed
    uniformly using the Fibonacci lattice method. Each point is associated with a
    specified object size.

    Args:
        num_points: The total number of points to generate.
        inner_radius: The minimum distance from the origin (must be positive).
        outer_radius: The maximum distance from the origin (must be >= inner_radius).
        object_size_m: The size in meters to be associated with each point.
                       Defaults to 1.0.
        seed: An optional integer to seed the random number generator for
              reproducible shuffling.

    Returns:
        A tuple containing:
        - A NumPy array of shape (num_points, 3) for the Cartesian coordinates.
        - A NumPy array of shape (num_points,) for the object size in meters.
    """
        ```

*   **`generate_report.py`**: 
    *   `generate_demo_html_report()`: Runs all plotting demos and saves the output to a single HTML file.
        ```python
        """
        Runs all plotting demos and saves the output to a single HTML file.
        """
        ```

*   **`lambertian.py`**: 
    *   `simple_lambertian(diameter: float, distance: float, albedo: float, angle: float, base_brightness: float)`: Calculates the apparent brightness of a Lambertian sphere.
        ```python
        """
        Calculates the apparent brightness of a Lambertian sphere.

    This function computes the apparent brightness of a diffusely
    reflecting sphere based on its physical properties, viewing
    geometry, and a given base incident brightness. It simplifies
    the calculation by taking the phase angle directly, rather than
    calculating it from vectors.

    Args:
        diameter: The diameter of the sphere in meters.
        distance: The distance from the sphere to the observer
            in meters.
        albedo: The fraction of incident light that is
            reflected (0.0 to 1.0).
        angle: The phase angle in radians. This is the angle
            between the light source and the observer as seen
            from the sphere's center (expected to be between 0 and pi).
        base_brightness: The incident flux or brightness of the
            light source at the sphere's location (e.g., in
            Watts per square meter or photons / s / m^2).

    Returns:
        The apparent brightness of the sphere as observed from
        the specified distance (e.g., in Watts per square meter
        or photons / s / m^2).
    """
        ```
    *   `lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: float, radius: float)`: Calculates the effective brightness of a Lambertian sphere.
        ```python
        """
        Calculates the effective brightness of a
        Lambertian sphere.

    This function determines the apparent brightness of
    a diffusely reflecting sphere based on the angle
    between the light source and the observer, the
    sphere's albedo (reflectivity), and its size.

    Args:
        vec_from_sphere_to_light: A 3-element NumPy
            array representing the direction vector from
            the sphere to the light source.
        vec_from_sphere_to_observer: A 3-element NumPy
            array representing the direction vector from
            the sphere to the observer.
        albedo: The fraction of incident light that is
            reflected (0.0 to 1.0).
        radius: The radius of the sphere in meters.

    Returns:
        The effective brightness cross-section in
        square meters. This value is proportional to
        the total light reflected towards the observer.
    """
        ```

*   **`observatories.py`**: 
    *   `add_observatories(sim_data: Dict[str, Any], num_observatories: int)`: Adds observatory data structures to the simulation data.
        ```python
        """
        Adds observatory data structures to the simulation data.

    Args:
        sim_data: The main simulation data dictionary.
        num_observatories: The number of observatories to add.
    """
        ```

*   **`plot_satellite_brighness.py`**: 
    *   `plot_satellite_brightness()`: Plots the apparent V-band photon flux and magnitude of satellites with various diameters over a range of distances.
        ```python
        """
        Plots the apparent V-band photon flux and magnitude of satellites
        with various diameters over a range of distances.

    This function calculates and plots two figures: one showing photon flux
    on a log-log scale, and another showing apparent magnitude (V-band)
    on a linear y-axis, for satellites of different sizes illuminated by
    the sun at a 90-degree phase angle.
    """
        ```

*   **`plotting_3d.py`**: 
    *   `plot_3d_scatter(positions: np.ndarray, title: str, plot_time: datetime, labels: Optional[List[str]]=None, marker_size: int=1, trace_name: str='Points')`: Creates a 3D plot of object positions.
        ```python
        """
        Creates a 3D plot of object positions.
        """
        ```

*   **`plotting_vectors.py`**: 
    *   `plot_pointing_vectors(data_struct: Dict[str, Any], title: str, plot_time: datetime)`: Creates a 3D plot of satellites with pointing vectors.
        ```python
        """
        Creates a 3D plot of satellites with pointing vectors.
        """
        ```

*   **`pointing.py`**: 
    *   `pointing_place_update(data_struct: Dict[str, Any])`: Increments the pointing place for all satellites, wrapping around if necessary.
        ```python
        """
        Increments the pointing place for all satellites, wrapping around if necessary.
        Basically this usges data_struct['satellites']['pointing_state'] and wraps to
        0 if its     POINTING_COUNT_IDX or greater.
        """
        ```
    *   `jerk(data_struct: Dict[str, Any], satellite_number: int)`: Moves the pointing vector of a specific satellite by 0.3 radians in a random direction.
        ```python
        """
        Moves the pointing vector of a specific satellite by 0.3 radians in a
        random direction.

    This function applies a random rotation to the satellite's pointing vector
    using a simplified version of Rodrigues' rotation formula.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_number: The index of the satellite to modify.

    Returns:
        The modified data_struct with the updated pointing vector.
    """
        ```
    *   `generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int)`: Generates a pointing sphere with n_points and stores it in the data_struct.
        ```python
        """
        Generates a pointing sphere with n_points and stores it in the data_struct.
        The index is ['pointing_spheres'][n] 
        If a sphere with the same number of points already exists, this function does nothing.
        """
        ```
    *   `update_satellite_pointing(data_struct: Dict[str, Any])`: Updates the pointing vector for each satellite based on its pointing state.
        ```python
        """
        Updates the pointing vector for each satellite based on its pointing state.
        """
        ```
    *   `find_and_jerk_blind_satellites(data_struct: Dict[str, Any])`: Finds satellites with no visibility and applies the 'jerk' function to them.
        ```python
        """
        Finds satellites with no visibility and applies the 'jerk' function to them.

    This function iterates through the visibility table. If any satellite (column)
    has no visible fixed points (i.e., the column sum is 0), the `jerk`
    function is called to randomly adjust its pointing vector.

    Args:
        data_struct: The main simulation data dictionary.

    Returns:
        The modified data_struct.
    """
        ```

*   **`pointing_vectors.py`**: 
    *   `pointing_vectors(n: int)`: Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.
        ```python
        """
        Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.

    Args:
        n: The number of points to generate.

    Returns:
        A NumPy array of shape (n, 3) for the Cartesian coordinates of the points.
    """
        ```
    *   `plot_vectors_on_sphere(vectors: np.ndarray, title: str)`: Creates a 3D plot of vectors on a sphere.
        ```python
        """
        Creates a 3D plot of vectors on a sphere.

    Args:
        vectors: A NumPy array of shape (n, 3) representing the vectors.
        title: The title of the plot.

    Returns:
        A Plotly figure object.
    """
        ```

*   **`propagation.py`**: 
    *   `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str)`: Adds and initializes a category of satellites from a TLE file.
        ```python
        """
        Adds and initializes a category of satellites from a TLE file.
        Mote although the TLEs are loaded, positions etc. are not.

    Args:
        sim_data: The main simulation data dictionary.
        tle_file_path: Path to the TLE file.
        sat_category: The key for this satellite category (e.g., 'satellites').

    Data added to the set_category element of sim_data
    position
    velocity
    acceleration
    orbital_elements
    epochs
    pointing
    """
        ```
    *   `celestial_update(data_struct: Dict[str, Any], time_date: datetime)`: Calculates and updates the positions of the Sun and Moon.
        ```python
        """
        Calculates and updates the positions of the Sun and Moon.
        """
        ```
    *   `readtle(tle_file_path: str)`: Reads a TLE file and extracts orbital elements and epochs for each satellite.
        ```python
        """
        Reads a TLE file and extracts orbital elements and epochs for each satellite.

    The array returned ahd the orbital elements in "canonical" order.
    """
        ```
    *   `propagate_satellites_new(data_struct: Dict[str, Any], time_date: datetime, sat_category: str=None)`: Updates satellite positions and pointing vectors based on their orbital elements.
        ```python
        """
        Updates satellite positions and pointing vectors based on their orbital elements.
        """
        ```

*   **`radiometry_calcs.py`**: 
    *   `mag(x: float)`: Calculates a magnitude value from a linear ratio.
        ```python
        """
        Calculates a magnitude value from a linear ratio.
        Uses the formula: magnitude = -2.5 * log10(ratio)
        """
        ```
    *   `amag(x: float)`: Calculates the linear ratio from a magnitude value.
        ```python
        """
        Calculates the linear ratio from a magnitude value.
        This is the inverse of the mag() function.
        Uses the formula: ratio = 10**(-0.4 * magnitude)
        """
        ```
    *   `_planck_law(wav_m: float, temp_k: float)`: Helper function for Planck's law for spectral radiance.
        ```python
        """
        Helper function for Planck's law for spectral radiance.
        Args:
        wav_m: Wavelength in meters.
        temp_k: Temperature in Kelvin.
        Returns:
        Spectral radiance in W / (m^2 * sr * m).
        """
        ```
    *   `blackbody_flux(temperature: float, lambda_short: float, lambda_long: float)`: Numerically computes the integrated spectral radiance of a blackbody over a given wavelength band.
        ```python
        """
        Numerically computes the integrated spectral radiance of
        a blackbody over a given wavelength band.

    Requires the scipy library.

    Args:
        temperature: The temperature of the blackbody in Kelvin.
        lambda_short: The short wavelength of the band in microns.
        lambda_long: The long wavelength of the band in microns.

    Returns:
        The integrated spectral radiance in units of
        Watts / (m^2 * steradian).
    """
        ```
    *   `stefan_boltzmann_law(temperature: float)`: Calculates the total power radiated per unit area by a blackbody using the Stefan-Boltzmann law.
        ```python
        """
        Calculates the total power radiated per unit area by a
        blackbody using the Stefan-Boltzmann law.

    Args:
        temperature: The temperature of the blackbody in Kelvin.

    Returns:
        The total radiated power per unit area in W / m^2.
    """
        ```
    *   `plot_blackbody_spectrum(temperature: float)`: Plots the spectral radiance of a blackbody from 0.5 to 30 microns.
        ```python
        """
        Plots the spectral radiance of a blackbody from
        0.5 to 30 microns.

    Args:
        temperature: The temperature of the blackbody in Kelvin.
    """
        ```
    *   `plot_blackbody_spectrum_visible_nir(temperature: float)`: Plots the spectral radiance of a blackbody from 0.1 to 1 micron.
        ```python
        """
        Plots the spectral radiance of a blackbody from
        0.1 to 1 micron.

    Args:
        temperature: The temperature of the blackbody in Kelvin.
    """
        ```
    *   `sat_magnitude(size: float, range: float, angle: float, band: str)`: given a satellite size and a waveband and range pull the brightness of the sun and the calibration from radiometry_data
        ```python
        """
        given a satellite size and a waveband and range
        pull the brightness of the sun and the calibration from radiometry_data
    
    """
        ```

*   **`radiometry_data.py`**: This module contains radiometric data for standard astronomical filters.

*   **`show_geo_search.py`**: 
    *   `show_geo_search()`: This demo initializes a simulation, adds a GEO constellation, and then generates several plots to visualize the satellite pointing updates and the RA/Dec history of one satellite.
        ```python
        """
        This demo initializes a simulation, adds a GEO constellation, and then
        generates several plots to visualize the satellite pointing updates and
        the RA/Dec history of one satellite.
        """
        ```
    *   `record_ra_dec()`: Records the RA and Dec of a satellite.

*   **`simulation.py`**: 
    *   `create_empty_simulation(start_time: datetime, delta_time: float=60.0)`: Initializes a minimal, empty data structure for a space simulation.
        ```python
        """
        Initializes a minimal, empty data structure for a space simulation.

    Args:
        start_time: The starting time and date of the simulation. This must be a
                    timezone-aware datetime object set to UTC.
        delta_time: The time step for the simulation in seconds.

    Returns:
        A dictionary representing the basic simulation state.
     This includes s
    
    """
        ```
    *   `add_celestial_bodies(sim_data: Dict[str, Any])`: Adds celestial body structures (for Sun and Moon) to the simulation data.
        ```python
        """
        Adds celestial body structures (for Sun and Moon) to the simulation data.

    Args:
        sim_data: The simulation data dictionary.
    """
        ```
    *   `add_fixed_points(sim_data: Dict[str, Any], num_points: int=100)`: Adds a structure for fixed reference points in the GCRS frame.
        ```python
        """
        Adds a structure for fixed reference points in the GCRS frame.

    Args:
        sim_data: The simulation data dictionary.
        num_points: The number of fixed points to generate.
    """
        ```

*   **`tests/test_simulation.py`**: 
    *   `test_create_empty_simulation_structure()`: Tests that create_empty_simulation returns a dictionary with the expected keys and initial values.
        ```python
        """
        Tests that create_empty_simulation returns a dictionary
        with the expected keys and initial values.
        """
        ```
    *   `test_create_empty_simulation_raises_errors()`: Tests that create_empty_simulation raises appropriate errors for invalid input.
        ```python
        """
        Tests that create_empty_simulation raises appropriate errors for invalid input.
        """
        ```

*   **`visibility.py`**: 
    *   `solarexclusion(data_struct: Dict[str, Any])`: Calculates solar exclusion for all satellites based on their pointing vectors.
        ```python
        """
        Calculates solar exclusion for all satellites based on their pointing vectors.

    This function operates in a vectorized manner on all satellites in the
    'satellites' category. It computes the angle between each satellite's
    pointing vector and the vector from the satellite to the Sun.

    Args:
        data_struct: The main simulation data dictionary.

    Returns:
        A tuple containing:
        - exclusion_vector (np.ndarray): An array of the same length as the
          number of satellites. An element is 1 if the satellite is within
          the solar exclusion angle, 0 otherwise.
        - angle_vector (np.ndarray): An array containing the calculated angle
          in radians for each satellite.
    """
        ```
    *   `exclusion(data_struct: Dict[str, Any], satellite_index: int, print_debug: bool=False)`: Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.
        ```python
        """
        Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_index: The index of the satellite to check.
        print_debug: If True, prints detailed debug information for the calculation.

    Returns:
        0 if the satellite's view is excluded, 1 otherwise.
    """
        ```
    *   `update_visibility_table(data_struct: Dict[str, Any], print_debug_for_sat: Optional[int]=None)`: Updates the visibility table for each satellite against each fixed point.
        ```python
        """
        Updates the visibility table for each satellite against each fixed point.

    Args:
        data_struct: The main simulation data dictionary.
        print_debug_for_sat: If an integer is provided, the `exclusion` function's
                             debug printout will be enabled for that satellite index.
    """
        ```

## Common Data Structures

The `sim_data` dictionary is the central data structure in the VibeVolts simulation toolkit.

### Top-Level Keys

- `start_time`: `datetime`
  - **Description**: The starting time and date of the simulation. Must be a timezone-aware datetime object set to UTC.

- `delta_time`: `float`
  - **Description**: The time step for the simulation in seconds.

- `counts`: `dict`
  - **Description**: A dictionary holding counts of various simulation objects.

- `pointing_spheres`: `dict`
  - **Description**: A dictionary to hold pre-computed pointing spheres. The keys are the number of points in the sphere.

- `celestial`: `dict`
  - **Description**: Holds data for celestial bodies (Sun and Moon).
  - **Sub-keys**:
    - `position`: `np.ndarray` (2, 3) - Position vectors (x, y, z) in meters.
    - `velocity`: `np.ndarray` (2, 3) - Velocity vectors in m/s.
    - `acceleration`: `np.ndarray` (2, 3) - Acceleration vectors in m/s^2.

- `fixedpoints`: `dict`
  - **Description**: Holds data for fixed reference points in the GCRS frame.
  - **Sub-keys**:
    - `position`: `np.ndarray` (n, 3) - Position vectors of fixed points.
    - `visibility`: `np.ndarray` (n, m) - Visibility table where rows are fixed points and columns are satellites. Value is 1 if visible, 0 otherwise.

- `satellites` (and other satellite categories like `red_satellites`): `dict`
  - **Description**: Holds data for a category of satellites.
  - **Sub-keys**:
    - `position`: `np.ndarray` (n, 3) - Position vectors in meters.
    - `velocity`: `np.ndarray` (n, 3) - Velocity vectors in m/s.
    - `acceleration`: `np.ndarray` (n, 3) - Acceleration vectors in m/s^2.
    - `orbital_elements`: `np.ndarray` (n, 6) - Keplerian orbital elements.
    - `epochs`: `list[datetime]` - Epoch for each satellite's orbital elements.
    - `pointing`: `np.ndarray` (n, 3) - Pointing direction vector.
    - `pointing_state`: `np.ndarray` (n, 2) - State of the pointing sequence for each satellite.
    - `detector`: `SimpleNamespace`
      - **Description**: A SimpleNamespace object containing the detector's properties.
      - **Sub-keys**:
        - `aperture`: `np.ndarray` (n) - Aperture area in square meters.
        - `pixelArea`: `np.ndarray` (n) - Pixel area in square arcseconds.
        - `qe`: `np.ndarray` (n) - Quantum efficiency (0.0 to 1.0).
        - `photoEff`: `np.ndarray` (n) - Fraction of photons in photometry bucket.
        - `pixCount`: `np.ndarray` (n) - Total number of pixels in the detector.
        - `solarEx`: `np.ndarray` (n) - Solar exclusion angle in radians.
        - `lunarex`: `np.ndarray` (n) - Lunar exclusion angle in radians.
        - `earthEx`: `np.ndarray` (n) - Earth exclusion angle in radians.
        - `skyBack`: `np.ndarray` (n) - Sky background in photons per square steradian.
        - `zpCal`: `np.ndarray` (n) - Filter calibration zero point in photons/m^2/s.
        - `itime`: `np.ndarray` (n) - Integration time to reach limiting magnitude.
        - `fov`: `np.ndarray` (n) - Field of view in radians.
        - `ifov`: `np.ndarray` (n) - Instantaneous field of view in radians.
        - `filt`: `list[str]` (n) - Filter name.

- `observatories`: `dict`
  - **Description**: Holds data for ground-based observatories.
  - **Sub-keys**:
    - `position`: `np.ndarray` (n, 3) - Position vectors in meters.
    - `velocity`: `np.ndarray` (n, 3) - Velocity vectors in m/s.
    - `acceleration`: `np.ndarray` (n, 3) - Acceleration vectors in m/s^2.
    - `pointing`: `np.ndarray` (n, 3) - Pointing direction vector.
    - `detector`: `SimpleNamespace`
      - **Description**: A SimpleNamespace object containing the detector's properties.
      - **Sub-keys**:
        - `aperture`: `np.ndarray` (n) - Aperture area in square meters.
        - `pixelArea`: `np.ndarray` (n) - Pixel area in square arcseconds.
        - `qe`: `np.ndarray` (n) - Quantum efficiency (0.0 to 1.0).
        - `photoEff`: `np.ndarray` (n) - Fraction of photons in photometry bucket.
        - `pixCount`: `np.ndarray` (n) - Total number of pixels in the detector.
        - `solarEx`: `np.ndarray` (n) - Solar exclusion angle in radians.
        - `lunarex`: `np.ndarray` (n) - Lunar exclusion angle in radians.
        - `earthEx`: `np.ndarray` (n) - Earth exclusion angle in radians.
        - `skyBack`: `np.ndarray` (n) - Sky background in photons per square steradian.
        - `zpCal`: `np.ndarray` (n) - Filter calibration zero point in photons/m^2/s.
        - `itime`: `np.ndarray` (n) - Integration time to reach limiting magnitude.
        - `fov`: `np.ndarray` (n) - Field of view in radians.
        - `ifov`: `np.ndarray` (n) - Instantaneous field of view in radians.
        - `filt`: `list[str]` (n) - Filter name.

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

## Development Conventions
*  **Text Files**: All purely text file have lines that are no more than 100 charcters long to aid reading.
*   **Data Structures**: The simulation state is managed in a central dictionary. This dictionary is initialized as a minimal structure using `create_empty_simulation` from `simulation.py`. Components like satellites, observatories, and celestial bodies are then added incrementally using dedicated functions (e.g., `add_satellites_from_tle`, `add_observatories`), making the structure highly modular and flexible.
*   **Modularity**: The code is organized into modules, each with a specific responsibility. This makes the code easy to understand, maintain, and extend.
*   **Vectorization**: The code makes extensive use of NumPy for vectorized operations, which provides a significant performance improvement over iterating through lists.
*   **Type Hinting**: The code uses type hints to improve readability and allow for static analysis.
*   **Docstrings**: All functions have docstrings that explain their purpose, arguments, and return values.
*   **Constants**: Constants are defined in `constants.py` to avoid magic numbers in the code.