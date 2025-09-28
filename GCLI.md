# GCLI.md

# all_demos.py
## run_all_demos()
Runs all demo functions, and either shows them inline or saves them to a single HTML file.

# demo_common.py
## initialize_standard_simulation(start_time: datetime)
Initializes a standard simulation with a predefined set of satellites. It consolidates TLE data, initializes the main data structure, and propagates satellites to their initial positions.

# constants.py
This file defines global constants used as indices for NumPy arrays throughout the simulation. This avoids using "magic numbers" and makes the code more readable. It includes:
- Radii for Earth and Moon in meters.
- Detector array indices for properties like aperture, pixel size, quantum efficiency, etc.
- Orbital elements array indices for semi-major axis, eccentricity, inclination, etc.
- Pointing state array indices for pointing count and place.

# demo1.py
## demo1()
Runs a full demonstration of the simulation tools. It generates and returns a Plotly figure object for the satellite positions plot.

# demo2.py
## demo2()
Runs a demonstration plotting satellite and celestial positions at two different times. It returns a Plotly figure object.

# demo3.py
## demo3()
Runs a demonstration plotting a single LEO satellite trajectory over 90 minutes. It returns a Plotly figure object.

# demo4.py
## demo4()
Runs a demonstration plotting a single GEO satellite trajectory over 23 hours. It returns a Plotly figure object.

# demo_exclusion_debug_print.py
## demo_exclusion_debug_print()
Demonstrates the debug printing feature of the exclusion function. It runs an exclusion check and prints detailed debug output.

# demo_exclusion_table.py
## demo_exclusion_table()
Demonstrates the creation and visualization of the exclusion table. It generates and returns a Plotly heatmap figure.

# demo_fixedpoints.py
## demo_fixedpoints()
Demonstrates the fixedpoints data structure by plotting it in 3D. It returns a Plotly figure object.

# demo_lambertian.py
## demo_lambertian()
Runs a demonstration of the lambertiansphere function, including example calculations and a plot of brightness vs. phase angle.

# demo_pointing_plot.py
## demo_pointing_plot()
Demonstrates the plot_pointing_vectors function. It returns a Plotly figure object showing satellite positions with pointing vectors.

# demo_pointing_sequence.py
## demo_pointing_sequence()
Demonstrates the satellite pointing sequence functionality. It animates the pointing vectors of satellites over several time steps.

# demo_pointing_vectors.py
## demo_pointing_vectors()
Demonstrates the generation and plotting of pointing vectors. It generates a number of vectors and plots them on a sphere.

# demo_sky_scan.py
## demo_sky_scan()
Performs a sky scan from a GEO satellite to map celestial exclusion zones. It generates a heatmap of the sky showing clear and excluded areas.

# show_geo_search.py
## show_geo_search()
A demo that performs a geometric search for satellites using a GEO constellation, and generates several plots to visualize the satellite pointing updates and the RA/Dec history of one satellite.

# generate_log_spherical_points.py
## generate_log_spherical_points(num_points: int, inner_radius: float, outer_radius: float, object_size_m: float = 1.0, seed: int = None)
Generates 3D points with logarithmic radial and uniform angular distribution. It creates a point cloud for use as fixed points in the simulation.

# generate_report.py
## generate_demo_html_report()
Runs all plotting demos and saves the output to a single HTML file named demo_plots.html.

# lambertian.py
## lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: float, radius: float)
Calculates the effective brightness of a Lambertian sphere. It determines the apparent brightness based on the angle between the light source and the observer.

# plotting_3d.py
## plot_3d_scatter(positions: np.ndarray, title: str, plot_time: datetime, labels: Optional[List[str]] = None, marker_size: int = 1, trace_name: str = 'Points')
Creates a 3D plot of object positions. It can plot any set of 3D points and includes a representation of the Earth.

# plotting_vectors.py
## plot_pointing_vectors(data_struct: Dict[str, Any], title: str, plot_time: datetime)
Creates a 3D plot of satellites with their pointing vectors. It shows the direction each satellite is pointing.

# pointing.py
## pointing_place_update(data_struct: Dict[str, Any])
Increments the pointing place for all satellites, wrapping around if necessary.
## jerk(data_struct: Dict[str, Any], satellite_number: int)
Moves the pointing vector of a specific satellite by 0.3 radians in a random direction.
## generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int)
Generates a pointing sphere with n_points and stores it in the data_struct.
## update_satellite_pointing(data_struct: Dict[str, Any])
Updates the pointing vector for each satellite based on its pointing state.
## find_and_jerk_blind_satellites(data_struct: Dict[str, Any])
Finds satellites with no visibility and applies the 'jerk' function to them.

# pointing_vectors.py
## pointing_vectors(n: int)
Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.
## plot_vectors_on_sphere(vectors: np.ndarray, title: str)
Creates a 3D plot of vectors on a sphere.

# propagation.py
## celestial_update(data_struct: Dict[str, Any], time_date: datetime)
Calculates and updates the positions of the Sun and Moon for a given time.
## readtle(tle_file_path: str)
Reads a TLE file and extracts orbital elements and epochs for each satellite.
## propagate_satellites(data_struct: Dict[str, Any], time_date: datetime)
Updates satellite positions and pointing vectors based on their orbital elements to a specified time.

# radiometry_calcs.py
## mag(x: float)
Calculates a magnitude value from a linear ratio.
## amag(x: float)
Calculates the linear ratio from a magnitude value.
## _planck_law(wav_m: float, temp_k: float)
Helper function for Planck's law for spectral radiance.
## blackbody_flux(temperature: float, lambda_short: float, lambda_long: float)
Numerically computes the integrated spectral radiance of a blackbody over a given wavelength band.
## stefan_boltzmann_law(temperature: float)
Calculates the total power radiated per unit area by a blackbody.
## plot_blackbody_spectrum(temperature: float)
Plots the spectral radiance of a blackbody from 0.5 to 30 microns.
## plot_blackbody_spectrum_visible_nir(temperature: float)
Plots the spectral radiance of a blackbody from 0.1 to 1 micron.

# radiometry_data.py
This file contains radiometric data and physical constants.
- `AU_M`: Astronomical Unit in meters.
- `RSUN_M`: Radius of the Sun in meters.
- `FILTER_DATA`: A dictionary containing data for standard astronomical filters (Johnson-Cousins, NIR, SDSS, Mid-IR, JWST MIRI).
  - For each filter, it provides:
    - `sun`: Apparent magnitude of the sun.
    - `sky`: Sky brightness in magnitudes per square arcsecond.
    - `central_wavelength`: Central wavelength in nanometers.
    - `bandwidth`: Bandwidth in nanometers.
    - `zero_point`: Photon flux for a 0-magnitude object.

# simulation.py
## create_empty_simulation(start_time: datetime, delta_time: float = 60.0)
Initializes a minimal, empty data structure for a space simulation.

### add_celestial_bodies(sim_data: Dict[str, Any]) -> None:
Adds celestial body structures (for Sun and Moon) to the simulation data.

### add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100) -> None:
Adds a structure for fixed reference points in the GCRS frame.

# visibility.py
## solarexclusion(data_struct: Dict[str, Any])
Calculates solar exclusion for all satellites based on their pointing vectors.
## exclusion(data_struct: Dict[str, Any], satellite_index: int, print_debug: bool = False)
Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.
## update_visibility_table(data_struct: Dict[str, Any], print_debug_for_sat: Optional[int] = None)
Updates the visibility table for each satellite against each fixed point.