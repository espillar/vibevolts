# VibeVolts Documentation

## Data Structure

The main simulation data dictionary (`sim_data`) is initially created by `create_empty_simulation` in `simulation.py`.

### Initial `sim_data` Structure

```python
# Initial structure from create_empty_simulation in simulation.py
{
    'start_time': <datetime>,
    'delta_time': <float>,
    'counts': {},
    'pointing_spheres': {},
}
```

### `sim_data` Dictionary Items and Modifying Functions

| Key | Modifying Function | File | Signature | Documentation | Type of Modification |
|---|---|---|---|---|---|
| `start_time` | `create_empty_simulation` | `simulation.py` | `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]` | Initializes a minimal, empty data structure for a space simulation. | initialization |
| `start_time` | `initialize_standard_simulation` | `demo_common.py` | `initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]` | Initializes a standard simulation with a predefined set of satellites. | initialization_via_function_call |
| `start_time` | `demo_constellation` | `demo_constellation.py` | `demo_constellation() -> go.Figure` | Runs a demonstration of the constellation creation tools. | initialization_via_function_call |
| `start_time` | `demo_exclusion_debug_print` | `demo_exclusion_debug_print.py` | `demo_exclusion_debug_print()` | Demonstrates the debug printing feature of the exclusion function. | initialization_via_function_call |
| `start_time` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | initialization_via_function_call |
| `start_time` | `demo_fixedpoints` | `demo_fixedpoints.py` | `demo_fixedpoints() -> go.Figure` | Demonstrates the fixedpoints data structure by plotting it in 3D. | initialization_via_function_call |
| `start_time` | `demo_pointing_plot` | `demo_pointing_plot.py` | `demo_pointing_plot() -> go.Figure` | Demonstrates the plot_pointing_vectors function. | initialization_via_function_call |
| `start_time` | `demo_pointing_sequence` | `demo_pointing_sequence.py` | `demo_pointing_sequence() -> go.Figure` | Demonstrates the satellite pointing sequence functionality. | initialization_via_function_call |
| `start_time` | `demo_show_geo_search` | `demo_show_geo_search.py` | `demo_show_geo_search()` | This demo initializes a simulation, adds a GEO constellation, and then\n    generates several plots to visualize the satellite pointing updates and\n    the RA/Dec history of one satellite. | initialization_via_function_call |
| `start_time` | `demo_sky_scan` | `demo_sky_scan.py` | `demo_sky_scan() -> go.Figure` | Performs a sky scan from a GEO satellite to map celestial exclusion zones. | initialization_via_function_call |
| `start_time` | `demo1` | `demo1.py` | `demo1() -> go.Figure` | Runs a full demonstration of the simulation tools. | initialization_via_function_call |
| `start_time` | `demo2` | `demo2.py` | `demo2() -> go.Figure` | Runs a demonstration plotting satellite and celestial positions. | initialization_via_function_call |
| `start_time` | `demo3` | `demo3.py` | `demo3() -> go.Figure` | Runs a demonstration plotting a single LEO satellite trajectory. | initialization_via_function_call |
| `start_time` | `demo4` | `demo4.py` | `demo4() -> go.Figure` | Runs a demonstration plotting a single GEO satellite trajectory. | initialization_via_function_call |
| `start_time` | `demo_exclusion_pointing` | `pointing.py` | `demo_exclusion_pointing()` | Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere. | initialization_via_function_call |
| `delta_time` | `create_empty_simulation` | `simulation.py` | `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]` | Initializes a minimal, empty data structure for a space simulation. | initialization |
| `delta_time` | `initialize_standard_simulation` | `demo_common.py` | `initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]` | Initializes a standard simulation with a predefined set of satellites. | initialization_via_function_call |
| `delta_time` | `demo_constellation` | `demo_constellation.py` | `demo_constellation() -> go.Figure` | Runs a demonstration of the constellation creation tools. | initialization_via_function_call |
| `delta_time` | `demo_exclusion_debug_print` | `demo_exclusion_debug_print.py` | `demo_exclusion_debug_print()` | Demonstrates the debug printing feature of the exclusion function. | initialization_via_function_call |
| `delta_time` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | initialization_via_function_call |
| `delta_time` | `demo_fixedpoints` | `demo_fixedpoints.py` | `demo_fixedpoints() -> go.Figure` | Demonstrates the fixedpoints data structure by plotting it in 3D. | initialization_via_function_call |
| `delta_time` | `demo_pointing_plot` | `demo_pointing_plot.py` | `demo_pointing_plot() -> go.Figure` | Demonstrates the plot_pointing_vectors function. | initialization_via_function_call |
| `delta_time` | `demo_pointing_sequence` | `demo_pointing_sequence.py` | `demo_pointing_sequence() -> go.Figure` | Demonstrates the satellite pointing sequence functionality. | initialization_via_function_call |
| `delta_time` | `demo_show_geo_search` | `demo_show_geo_search.py` | `demo_show_geo_search()` | This demo initializes a simulation, adds a GEO constellation, and then\n    generates several plots to visualize the satellite pointing updates and\n    the RA/Dec history of one satellite. | initialization_via_function_call |
| `delta_time` | `demo_sky_scan` | `demo_sky_scan.py` | `demo_sky_scan() -> go.Figure` | Performs a sky scan from a GEO satellite to map celestial exclusion zones. | initialization_via_function_call |
| `delta_time` | `demo1` | `demo1.py` | `demo1() -> go.Figure` | Runs a full demonstration of the simulation tools. | initialization_via_function_call |
| `delta_time` | `demo2` | `demo2.py` | `demo2() -> go.Figure` | Runs a demonstration plotting satellite and celestial positions. | initialization_via_function_call |
| `delta_time` | `demo3` | `demo3.py` | `demo3() -> go.Figure` | Runs a demonstration plotting a single LEO satellite trajectory. | initialization_via_function_call |
| `delta_time` | `demo4` | `demo4.py` | `demo4() -> go.Figure` | Runs a demonstration plotting a single GEO satellite trajectory. | initialization_via_function_call |
| `delta_time` | `demo_exclusion_pointing` | `pointing.py` | `demo_exclusion_pointing()` | Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere. | initialization_via_function_call |
| `counts` | `create_empty_simulation` | `simulation.py` | `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]` | Initializes a minimal, empty data structure for a space simulation. | initialization |
| `counts` | `add_fixed_points` | `targets.py` | `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None` | Adds a structure for fixed reference points in the GCRS frame. | assignment |
| `counts` | `initialize_standard_simulation` | `demo_common.py` | `initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]` | Initializes a standard simulation with a predefined set of satellites. | initialization_via_function_call |
| `counts` | `add_celestial_bodies` | `celestialbodies.py` | `add_celestial_bodies(sim_data: Dict[str, Any]) -> None` | Adds celestial body structures (for Sun and Moon) to the simulation data. | assignment |
| `counts` | `geos` | `constellation.py` | `geos(sim_data, n, fov) -> None` | Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation. | assignment |
| `counts` | `geosmod` | `constellation.py` | `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None` | Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation. | assignment |
| `counts` | `add_observatories` | `observatories.py` | `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None` | Adds observatory data structures to the simulation data. | assignment |
| `counts` | `add_satellites_from_tle` | `propagation.py` | `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None` | Adds and initializes a category of satellites from a TLE file. | assignment_to_subkey_by_variable_key |
| `counts` | `demo_constellation` | `demo_constellation.py` | `demo_constellation() -> go.Figure` | Runs a demonstration of the constellation creation tools. | initialization_via_function_call |
| `counts` | `demo_exclusion_debug_print` | `demo_exclusion_debug_print.py` | `demo_exclusion_debug_print()` | Demonstrates the debug printing feature of the exclusion function. | initialization_via_function_call |
| `counts` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | initialization_via_function_call |
| `counts` | `demo_fixedpoints` | `demo_fixedpoints.py` | `demo_fixedpoints() -> go.Figure` | Demonstrates the fixedpoints data structure by plotting it in 3D. | initialization_via_function_call |
| `counts` | `demo_pointing_plot` | `demo_pointing_plot.py` | `demo_pointing_plot() -> go.Figure` | Demonstrates the plot_pointing_vectors function. | initialization_via_function_call |
| `counts` | `demo_pointing_sequence` | `demo_pointing_sequence.py` | `demo_pointing_sequence() -> go.Figure` | Demonstrates the satellite pointing sequence functionality. | initialization_via_function_call |
| `counts` | `demo_show_geo_search` | `demo_show_geo_search.py` | `demo_show_geo_search()` | This demo initializes a simulation, adds a GEO constellation, and then\n    generates several plots to visualize the satellite pointing updates and\n    the RA/Dec history of one satellite. | initialization_via_function_call |
| `counts` | `demo_sky_scan` | `demo_sky_scan.py` | `demo_sky_scan() -> go.Figure` | Performs a sky scan from a GEO satellite to map celestial exclusion zones. | initialization_via_function_call |
| `counts` | `demo1` | `demo1.py` | `demo1() -> go.Figure` | Runs a full demonstration of the simulation tools. | initialization_via_function_call |
| `counts` | `demo2` | `demo2.py` | `demo2() -> go.Figure` | Runs a demonstration plotting satellite and celestial positions. | initialization_via_function_call |
| `counts` | `demo3` | `demo3.py` | `demo3() -> go.Figure` | Runs a demonstration plotting a single LEO satellite trajectory. | initialization_via_function_call |
| `counts` | `demo4` | `demo4.py` | `demo4() -> go.Figure` | Runs a demonstration plotting a single GEO satellite trajectory. | initialization_via_function_call |
| `counts` | `demo_exclusion_pointing` | `pointing.py` | `demo_exclusion_pointing()` | Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere. | initialization_via_function_call |
| `pointing_spheres` | `create_empty_simulation` | `simulation.py` | `create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]` | Initializes a minimal, empty data structure for a space simulation. | initialization |
| `pointing_spheres` | `initialize_standard_simulation` | `demo_common.py` | `initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]` | Initializes a standard simulation with a predefined set of satellites. | modification_via_function_call |
| `pointing_spheres` | `geos` | `constellation.py` | `geos(sim_data, n, fov) -> None` | Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation. | implicit_assignment_via_function_call |
| `pointing_spheres` | `geosmod` | `constellation.py` | `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None` | Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation. | implicit_assignment_via_function_call |
| `pointing_spheres` | `generate_pointing_sphere` | `pointing.py` | `generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int, debug: bool = False) -> None` | Generates a pointing sphere with n_points and stores it in the data_struct['pointing_sphers'][n_points] | assignment_to_subkey_by_key |
| `pointing_spheres` | `demo_constellation` | `demo_constellation.py` | `demo_constellation() -> go.Figure` | Runs a demonstration of the constellation creation tools. | initialization_via_function_call |
| `pointing_spheres` | `demo_exclusion_debug_print` | `demo_exclusion_debug_print.py` | `demo_exclusion_debug_print()` | Demonstrates the debug printing feature of the exclusion function. | initialization_via_function_call |
| `pointing_spheres` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | initialization_via_function_call |
| `pointing_spheres` | `demo_fixedpoints` | `demo_fixedpoints.py` | `demo_fixedpoints() -> go.Figure` | Demonstrates the fixedpoints data structure by plotting it in 3D. | initialization_via_function_call |
| `pointing_spheres` | `demo_pointing_plot` | `demo_pointing_plot.py` | `demo_pointing_plot() -> go.Figure` | Demonstrates the plot_pointing_vectors function. | initialization_via_function_call |
| `pointing_spheres` | `demo_pointing_sequence` | `demo_pointing_sequence.py` | `demo_pointing_sequence() -> go.Figure` | Demonstrates the satellite pointing sequence functionality. | initialization_via_function_call |
| `pointing_spheres` | `demo_show_geo_search` | `demo_show_geo_search.py` | `demo_show_geo_search()` | This demo initializes a simulation, adds a GEO constellation, and then\n    generates several plots to visualize the satellite pointing updates and\n    the RA/Dec history of one satellite. | initialization_via_function_call |
| `pointing_spheres` | `demo_sky_scan` | `demo_sky_scan.py` | `demo_sky_scan() -> go.Figure` | Performs a sky scan from a GEO satellite to map celestial exclusion zones. | initialization_via_function_call |
| `fixedpoints` | `add_fixed_points` | `targets.py` | `add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None` | Adds a structure for fixed reference points in the GCRS frame. | assignment |
| `fixedpoints` | `initialize_standard_simulation` | `demo_common.py` | `initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]` | Initializes a standard simulation with a predefined set of satellites. | modification_via_function_call |
| `fixedpoints` | `demo_constellation` | `demo_constellation.py` | `demo_constellation() -> go.Figure` | Runs a demonstration of the constellation creation tools. | initialization_via_function_call |
| `fixedpoints` | `demo_exclusion_debug_print` | `demo_exclusion_debug_print.py` | `demo_exclusion_debug_print()` | Demonstrates the debug printing feature of the exclusion function. | initialization_via_function_call |
| `fixedpoints` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | initialization_via_function_call |
| `fixedpoints` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | potential_modification_via_function_call_if_active |
| `fixedpoints` | `demo_fixedpoints` | `demo_fixedpoints.py` | `demo_fixedpoints() -> go.Figure` | Demonstrates the fixedpoints data structure by plotting it in 3D. | initialization_via_function_call |
| `fixedpoints` | `demo_pointing_plot` | `demo_pointing_plot.py` | `demo_pointing_plot() -> go.Figure` | Demonstrates the plot_pointing_vectors function. | initialization_via_function_call |
| `fixedpoints` | `demo_pointing_sequence` | `demo_pointing_sequence.py` | `demo_pointing_sequence() -> go.Figure` | Demonstrates the satellite pointing sequence functionality. | initialization_via_function_call |
| `fixedpoints` | `demo_show_geo_search` | `demo_show_geo_search.py` | `demo_show_geo_search()` | This demo initializes a simulation, adds a GEO constellation, and then\n    generates several plots to visualize the satellite pointing updates and\n    the RA/Dec history of one satellite. | initialization_via_function_call |
| `fixedpoints` | `demo_sky_scan` | `demo_sky_scan.py` | `demo_sky_scan() -> go.Figure` | Performs a sky scan from a GEO satellite to map celestial exclusion zones. | initialization_via_function_call |
| `fixedpoints` | `demo1` | `demo1.py` | `demo1() -> go.Figure` | Runs a full demonstration of the simulation tools. | initialization_via_function_call |
| `fixedpoints` | `demo2` | `demo2.py` | `demo2() -> go.Figure` | Runs a demonstration plotting satellite and celestial positions. | initialization_via_function_call |
| `fixedpoints` | `demo3` | `demo3.py` | `demo3() -> go.Figure` | Runs a demonstration plotting a single LEO satellite trajectory. | initialization_via_function_call |
| `fixedpoints` | `demo4` | `demo4.py` | `demo4() -> go.Figure` | Runs a demonstration plotting a single GEO satellite trajectory. | initialization_via_function_call |
| `fixedpoints` | `demo_exclusion_pointing` | `pointing.py` | `demo_exclusion_pointing()` | Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere. | initialization_via_function_call |
| `celestial` | `initialize_standard_simulation` | `demo_common.py` | `initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]` | Initializes a standard simulation with a predefined set of satellites. | modification_via_function_call |
| `celestial` | `add_celestial_bodies` | `celestialbodies.py` | `add_celestial_bodies(sim_data: Dict[str, Any]) -> None` | Adds celestial body structures (for Sun and Moon) to the simulation data. | assignment |
| `celestial` | `celestial_update` | `celestialbodies.py` | `celestial_update(data_struct: Dict[str, Any], time_date: datetime) -> Dict[str, Any]` | Calculates and updates the positions of the Sun and Moon. | update_subkey_position |
| `celestial` | `demo_constellation` | `demo_constellation.py` | `demo_constellation() -> go.Figure` | Runs a demonstration of the constellation creation tools. | initialization_via_function_call |
| `celestial` | `demo_exclusion_debug_print` | `demo_exclusion_debug_print.py` | `demo_exclusion_debug_print()` | Demonstrates the debug printing feature of the exclusion function. | initialization_via_function_call |
| `celestial` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | initialization_via_function_call |
| `celestial` | `demo_fixedpoints` | `demo_fixedpoints.py` | `demo_fixedpoints() -> go.Figure` | Demonstrates the fixedpoints data structure by plotting it in 3D. | initialization_via_function_call |
| `celestial` | `demo_pointing_plot` | `demo_pointing_plot.py` | `demo_pointing_plot() -> go.Figure` | Demonstrates the plot_pointing_vectors function. | initialization_via_function_call |
| `celestial` | `demo_pointing_sequence` | `demo_pointing_sequence.py` | `demo_pointing_sequence() -> go.Figure` | Demonstrates the satellite pointing sequence functionality. | initialization_via_function_call |
| `celestial` | `demo_show_geo_search` | `demo_show_geo_search.py` | `demo_show_geo_search()` | This demo initializes a simulation, adds a GEO constellation, and then\n    generates several plots to visualize the satellite pointing updates and\n    the RA/Dec history of one satellite. | initialization_via_function_call |
| `celestial` | `demo_sky_scan` | `demo_sky_scan.py` | `demo_sky_scan() -> go.Figure` | Performs a sky scan from a GEO satellite to map celestial exclusion zones. | initialization_via_function_call |
| `celestial` | `demo1` | `demo1.py` | `demo1() -> go.Figure` | Runs a full demonstration of the simulation tools. | initialization_via_function_call |
| `celestial` | `demo2` | `demo2.py` | `demo2() -> go.Figure` | Runs a demonstration plotting satellite and celestial positions. | initialization_via_function_call |
| `celestial` | `demo3` | `demo3.py` | `demo3() -> go.Figure` | Runs a demonstration plotting a single LEO satellite trajectory. | initialization_via_function_call |
| `celestial` | `demo4` | `demo4.py` | `demo4() -> go.Figure` | Runs a demonstration plotting a single GEO satellite trajectory. | initialization_via_function_call |
| `celestial` | `demo_exclusion_pointing` | `pointing.py` | `demo_exclusion_pointing()` | Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere. | initialization_via_function_call |
| `satellites` | `initialize_standard_simulation` | `demo_common.py` | `initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]` | Initializes a standard simulation with a predefined set of satellites. | modification_via_function_call |
| `satellites` | `geos` | `constellation.py` | `geos(sim_data, n, fov) -> None` | Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation. | assignment |
| `satellites` | `geosmod` | `constellation.py` | `geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None` | Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation. | assignment |
| `satellites` | `add_satellites_from_tle` | `propagation.py` | `add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None` | Adds and initializes a category of satellites from a TLE file. | assignment_to_subkey_by_variable_key |
| `satellites` | `propagate_satellites_new` | `propagation.py` | `propagate_satellites_new(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]` | Updates satellite positions and pointing vectors based on their orbital elements. | direct_subkey_modification |
| `satellites` | `update_satellite_pointing` | `pointing.py` | `update_satellite_pointing(data_struct: Dict[str, Any], debug: bool = False) -> None` | Updates the pointing vector for each satellite, skipping excluded pointing directions. | direct_subkey_modification |
| `satellites` | `jerk` | `pointing.py` | `jerk(data_struct: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]` | Moves the pointing vector of specific satellites by 0.3 radians in a\n    random direction. | direct_subkey_modification |
| `satellites` | `demo_constellation` | `demo_constellation.py` | `demo_constellation() -> go.Figure` | Runs a demonstration of the constellation creation tools. | initialization_via_function_call |
| `satellites` | `demo_exclusion_debug_print` | `demo_exclusion_debug_print.py` | `demo_exclusion_debug_print()` | Demonstrates the debug printing feature of the exclusion function. | initialization_via_function_call |
| `satellites` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | initialization_via_function_call |
| `satellites` | `demo_fixedpoints` | `demo_fixedpoints.py` | `demo_fixedpoints() -> go.Figure` | Demonstrates the fixedpoints data structure by plotting it in 3D. | initialization_via_function_call |
| `satellites` | `demo_pointing_plot` | `demo_pointing_plot.py` | `demo_pointing_plot() -> go.Figure` | Demonstrates the plot_pointing_vectors function. | initialization_via_function_call |
| `satellites` | `demo_pointing_sequence` | `demo_pointing_sequence.py` | `demo_pointing_sequence() -> go.Figure` | Demonstrates the satellite pointing sequence functionality. | initialization_via_function_call |
| `satellites` | `demo_show_geo_search` | `demo_show_geo_search.py` | `demo_show_geo_search()` | This demo initializes a simulation, adds a GEO constellation, and then\n    generates several plots to visualize the satellite pointing updates and\n    the RA/Dec history of one satellite. | initialization_via_function_call |
| `satellites` | `demo_sky_scan` | `demo_sky_scan.py` | `demo_sky_scan() -> go.Figure` | Performs a sky scan from a GEO satellite to map celestial exclusion zones. | initialization_via_function_call |
| `satellites` | `demo1` | `demo1.py` | `demo1() -> go.Figure` | Runs a full demonstration of the simulation tools. | initialization_via_function_call |
| `satellites` | `demo2` | `demo2.py` | `demo2() -> go.Figure` | Runs a demonstration plotting satellite and celestial positions. | initialization_via_function_call |
| `satellites` | `demo3` | `demo3.py` | `demo3() -> go.Figure` | Runs a demonstration plotting a single LEO satellite trajectory. | initialization_via_function_call |
| `satellites` | `demo4` | `demo4.py` | `demo4() -> go.Figure` | Runs a demonstration plotting a single GEO satellite trajectory. | initialization_via_function_call |
| `satellites` | `demo_exclusion_pointing` | `pointing.py` | `demo_exclusion_pointing()` | Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere. | initialization_via_function_call |
| `observatories` | `initialize_standard_simulation` | `demo_common.py` | `initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]` | Initializes a standard simulation with a predefined set of satellites. | modification_via_function_call |
| `observatories` | `add_observatories` | `observatories.py` | `add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None` | Adds observatory data structures to the simulation data. | assignment |
| `observatories` | `demo_constellation` | `demo_constellation.py` | `demo_constellation() -> go.Figure` | Runs a demonstration of the constellation creation tools. | initialization_via_function_call |
| `observatories` | `demo_exclusion_debug_print` | `demo_exclusion_debug_print.py` | `demo_exclusion_debug_print()` | Demonstrates the debug printing feature of the exclusion function. | initialization_via_function_call |
| `observatories` | `demo_exclusion_table` | `demo_exclusion_table.py` | `demo_exclusion_table() -> go.Figure` | Demonstrates the creation and visualization of the exclusion table. | initialization_via_function_call |
| `observatories` | `demo_fixedpoints` | `demo_fixedpoints.py` | `demo_fixedpoints() -> go.Figure` | Demonstrates the fixedpoints data structure by plotting it in 3D. | initialization_via_function_call |
| `observatories` | `demo_pointing_plot` | `demo_pointing_plot.py` | `demo_pointing_plot() -> go.Figure` | Demonstrates the plot_pointing_vectors function. | initialization_via_function_call |
| `observatories` | `demo_pointing_sequence` | `demo_pointing_sequence.py` | `demo_pointing_sequence() -> go.Figure` | Demonstrates the satellite pointing sequence functionality. | initialization_via_function_call |
| `observatories` | `demo_show_geo_search` | `demo_show_geo_search.py` | `demo_show_geo_search()` | This demo initializes a simulation, adds a GEO constellation, and then\n    generates several plots to visualize the satellite pointing updates and\n    the RA/Dec history of one satellite. | initialization_via_function_call |
| `observatories` | `demo_sky_scan` | `demo_sky_scan.py` | `demo_sky_scan() -> go.Figure` | Performs a sky scan from a GEO satellite to map celestial exclusion zones. | initialization_via_function_call |
| `observatories` | `demo1` | `demo1.py` | `demo1() -> go.Figure` | Runs a full demonstration of the simulation tools. | initialization_via_function_call |
| `observatories` | `demo2` | `demo2.py` | `demo2() -> go.Figure` | Runs a demonstration plotting satellite and celestial positions. | initialization_via_function_call |
| `observatories` | `demo3` | `demo3.py` | `demo3() -> go.Figure` | Runs a demonstration plotting a single LEO satellite trajectory. | initialization_via_function_call |
| `observatories` | `demo4` | `demo4.py` | `demo4() -> go.Figure` | Runs a demonstration plotting a single GEO satellite trajectory. | initialization_via_function_call |
| `observatories` | `demo_exclusion_pointing` | `pointing.py` | `demo_exclusion_pointing()` | Demonstrates satellite pointing with a solar exclusion angle and a detector field of view, plotting the pointing history on a sphere. | initialization_via_function_call |


## Demos

The following Python files contain demonstrations of the VibeVolts toolkit:

*   `all_demos.py`: Contains `demo_vector_resorting_plot` and `run_all_demos`, which orchestrates other demos.
*   `demo1.py`: Contains `demo1`, a full demonstration of simulation tools.
*   `demo2.py`: Contains `demo2`, a demonstration plotting satellite and celestial positions.
*   `demo3.py`: Contains `demo3`, a demonstration plotting a single LEO satellite trajectory.
*   `demo4.py`: Contains `demo4`, a demonstration plotting a single GEO satellite trajectory.
*   `demo_common.py`: Contains `initialize_standard_simulation`, used by many other demos for setup.
*   `demo_constellation.py`: Contains `demo_constellation`, for constellation creation tools.
*   `demo_exclusion_debug_print.py`: Contains `demo_exclusion_debug_print`, demonstrating exclusion debug printing.
*   `demo_exclusion_table.py`: Contains `demo_exclusion_table`, demonstrating exclusion table visualization.
*   `demo_fixedpoints.py`: Contains `demo_fixedpoints`, demonstrating fixed points data structure.
*   `demo_lambertian.py`: Contains `demo_lambertian`, demonstrating the Lambertian sphere function.
*   `demo_pointing_plot.py`: Contains `demo_pointing_plot`, demonstrating pointing vector plotting.
*   `demo_pointing_sequence.py`: Contains `demo_pointing_sequence`, demonstrating satellite pointing sequence.
*   `demo_pointing_vectors.py`: Contains `demo_pointing_vectors`, demonstrating pointing vector generation and plotting.
*   `demo_requiredIntegrationTime.py`: Contains `demo_requiredIntegrationTime`, demonstrating required integration time calculation.
*   `demo_show_geo_search.py`: Contains `demo_show_geo_search`, for visualizing GEO satellite pointing and RA/Dec history.
*   `demo_sky_scan.py`: Contains `demo_sky_scan`, for performing a sky scan from a GEO satellite.
*   `fibonacciSearch.py`: Contains `test_vector_resorting`, which is called by `all_demos.py`.
*   `pointing.py`: Contains `demo_exclusion_pointing`, demonstrating satellite pointing with exclusion.


## Python Files and Functions

### `targets.py`

#### `add_fixed_points`

```python
add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None
```

Adds a structure for fixed reference points in the GCRS frame.


### `exclusion.py`

#### `exclusion`

```python
exclusion(data_struct: Dict[str, Any], satellite_index: int, print_debug: bool = False) -> int
```

Determines if a satellite\'s pointing vector is excluded by the Sun, Moon, or Earth.


### `all_demos.py`

#### `demo_vector_resorting_plot`

```python
demo_vector_resorting_plot() -> go.Figure
```

Runs the test_vector_resorting function and returns its figure.

#### `run_all_demos`

```python
run_all_demos(save_html=False)
```

Runs all demo functions, and either shows them inline or saves them to a single HTML file.


### `celestialbodies.py`

#### `add_celestial_bodies`

```python
add_celestial_bodies(sim_data: Dict[str, Any]) -> None
```

Adds celestial body structures (for Sun and Moon) to the simulation data.

#### `celestial_update`

```python
celestial_update(data_struct: Dict[str, Any], time_date: datetime) -> Dict[str, Any]
```

Calculates and updates the positions of the Sun and Moon.


### `constellation.py`

#### `geos`

```python
geos(sim_data, n, fov) -> None
```

Creates n equally spaced satellites in GEO and adds them to the \'satellites\' group in the simulation.

#### `geosmod`

```python
geosmod(sim_data, n, band, fov, ifov, aper, limitingmag) -> None
```

Creates n equally spaced satellites in GEO and adds them to the \'satellites\' group in the simulation.


### `demo_common.py`

#### `initialize_standard_simulation`

```python
initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]
```

Initializes a standard simulation with a predefined set of satellites.


### `demo_constellation.py`

#### `demo_constellation`

```python
demo_constellation() -> go.Figure
```

Runs a demonstration of the constellation creation tools.


### `demo_exclusion_debug_print.py`

#### `demo_exclusion_debug_print`

```python
demo_exclusion_debug_print()
```

Demonstrates the debug printing feature of the exclusion function.


### `demo_exclusion_table.py`

#### `demo_exclusion_table`

```python
demo_exclusion_table() -> go.Figure
```

Demonstrates the creation and visualization of the exclusion table.


### `demo_fixedpoints.py`

#### `demo_fixedpoints`

```python
demo_fixedpoints() -> go.Figure
```

Demonstrates the fixedpoints data structure by plotting it in 3D.


### `demo_lambertian.py`

#### `demo_lambertian`

```python
demo_lambertian()
```

Runs a demonstration of the lambertiansphere function,
including example calculations and a plot.


### `demo_pointing_plot.py`

#### `demo_pointing_plot`

```python
demo_pointing_plot() -> go.Figure
```

Demonstrates the plot_pointing_vectors function.


### `demo_pointing_sequence.py`

#### `demo_pointing_sequence`

```python
demo_pointing_sequence() -> go.Figure
```

Demonstrates the satellite pointing sequence functionality.


### `demo_pointing_vectors.py`

#### `demo_pointing_vectors`

```python
demo_pointing_vectors() -> go.Figure
```

Demonstrates the generation and plotting of pointing vectors.


### `demo_requiredIntegrationTime.py`

#### `demo_requiredIntegrationTime`

```python
demo_requiredIntegrationTime()
```

Demonstrates the requiredIntegrationTime function.


### `demo_show_geo_search.py`

#### `demo_show_geo_search`

```python
demo_show_geo_search()
```

This demo initializes a simulation, adds a GEO constellation, and then
generates several plots to visualize the satellite pointing updates and
the RA/Dec history of one satellite.


### `demo_sky_scan.py`

#### `demo_sky_scan`

```python
demo_sky_scan() -> go.Figure
```

Performs a sky scan from a GEO satellite to map celestial exclusion zones.


### `demo1.py`

#### `demo1`

```python
demo1() -> go.Figure
```

Runs a full demonstration of the simulation tools.


### `demo2.py`

#### `demo2`

```python
demo2() -> go.Figure
```

Runs a demonstration plotting satellite and celestial positions.


### `demo3.py`

#### `demo3`

```python
demo3() -> go.Figure
```

Runs a demonstration plotting a single LEO satellite trajectory.


### `demo4.py`

#### `demo4`

```python
demo4() -> go.Figure
```

Runs a demonstration plotting a single GEO satellite trajectory.


### `detector.py`

#### `makeBlankDetector`

```python
makeBlankDetector(n)
```



#### `makeDetector`

```python
makeDetector(n, band, fov,ifov, aper, qe = 0.5, photfrac=0.7, solarex = 20 * DEGREE, lunarex = 10 * DEGREE, earthex= 15 * DEGREE)
```

makeDetector takes parameters of a sensor and stuffs a filter array and a detector array, which it returns.

#### `requiredIntegrationTime`

```python
requiredIntegrationTime(limitingMag, SNR, d, debug = 0)
```

requiredIntegrationTime(limitingMag, d)
takes a two dimensional detector array (\"detect\")and calculates
      all the integration tiemes
and returns those as a vector.

#### `testdetector`

```python
testdetector()
```

testdetector creates an example that can be compared
with some of the stuff in Curio.


### `fibonacciSearch.py`

#### `pointing_vectors`

```python
pointing_vectors(n: int) -> np.ndarray
```

Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.

#### `resort_vectors_by_proximity`

```python
resort_vectors_by_proximity(unit_vectors: np.ndarray) -> np.ndarray
```

Resorts a list of vectors by making each subsequent vector the closest one
in the remaining set to the previous one.

#### `plot_vectors_on_sphere`

```python
plot_vectors_on_sphere(vectors: np.ndarray, title: str) -> go.Figure
```

Creates a 3D plot of vectors on a sphere.

#### `test_vector_resorting`

```python
test_vector_resorting()
```

Tests the vector resorting and plots the Euclidean distance between subsequent vectors.


### `generate_log_spherical_points.py`

#### `generate_log_spherical_points`

```python
generate_log_spherical_points(num_points: int, inner_radius: float, outer_radius: float, object_size_m: float = 1.0, seed: int = None) -> tuple[np.ndarray, np.ndarray]
```

Generates 3D points with logarithmic radial and uniform angular distribution.


### `generate_report.py`

#### `generate_demo_html_report`

```python
generate_demo_html_report()
```

Runs all plotting demos and saves the output to a single HTML file.


### `lambertian.py`

#### `simple_lambertian`

```python
simple_lambertian(diameter: float, distance: float, albedo: float, angle: float, base_brightness: float) -> float
```

Calculates the apparent brightness of a Lambertian sphere.

#### `lambertiansphere`

```python
lambertiansphere(vec_from_sphere_to_light: np.ndarray, vec_from_sphere_to_observer: np.ndarray, albedo: float, radius: float) -> float
```

Calculates the effective brightness of a
Lambertian sphere.


### `observatories.py`

#### `add_observatories`

```python
add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None
```

Adds observatory data structures to the simulation data.


### `plot_satellite_brighness.py`

#### `plot_satellite_brightness`

```python
plot_satellite_brightness()
```

Plots the apparent V-band photon flux and magnitude of satellites
with various diameters over a range of distances.


### `plotting_3d.py`

#### `plot_3d_scatter`

```python
plot_3d_scatter(positions: np.ndarray, title: str, plot_time: datetime, labels: Optional[List[str]] = None, marker_size: int = 1, trace_name: str = 'Points') -> go.Figure
```

Creates a 3D plot of object positions.


### `plotting_vectors.py`

#### `plot_pointing_vectors`

```python
plot_pointing_vectors(data_struct: Dict[str, Any], title: str, plot_time: datetime) -> go.Figure
```

Creates a 3D plot of satellites with pointing vectors.


### `pointing.py`

#### `generate_pointing_sphere`

```python
generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int, debug: bool = False) -> None
```

Generates a pointing sphere with n_points and stores it in the data_struct[\'pointing_sphers\'][n_points]

#### `update_satellite_pointing`

```python
update_satellite_pointing(data_struct: Dict[str, Any], debug: bool = False) -> None
```

Updates the pointing vector for each satellite, skipping excluded pointing directions.

#### `demo_exclusion_pointing`

```python
demo_exclusion_pointing()
```

Demonstrates satellite pointing with a solar exclusion angle and a detector
field of view, plotting the pointing history on a sphere.

#### `jerk`

```python
jerk(data_struct: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]
```

Moves the pointing vector of specific satellites by 0.3 radians in a
random direction.


### `propagation.py`

#### `add_satellites_from_tle`

```python
add_satellites_from_tle(sim_data: Dict[str, Any], tle_file_path: str, sat_category: str) -> None
```

Adds and initializes a category of satellites from a TLE file.

#### `readtle`

```python
readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]
```

Reads a TLE file and extracts orbital elements and epochs for each satellite.

#### `propagate_satellites_new`

```python
propagate_satellites_new(data_struct: Dict[str, Any], time_date: datetime, sat_category: str = None) -> Dict[str, Any]
```

Updates satellite positions and pointing vectors based on their orbital elements.


### `radiometry_calcs.py`

#### `mag`

```python
mag(x: float) -> float
```

Calculates a magnitude value from a linear ratio.

#### `amag`

```python
amag(x: float) -> float
```

Calculates the linear ratio from a magnitude value.

#### `_planck_law`

```python
_planck_law(wav_m: float, temp_k: float) -> float
```

Helper function for Planck\'s law for spectral radiance.

#### `blackbody_flux`

```python
blackbody_flux(temperature: float, lambda_short: float, lambda_long: float) -> float
```

Numerically computes the integrated spectral radiance of
a blackbody over a given wavelength band.

#### `stefan_boltzmann_law`

```python
stefan_boltzmann_law(temperature: float) -> float
```

Calculates the total power radiated per unit area by a
blackbody using the Stefan-Boltzmann law.

#### `plot_blackbody_spectrum`

```python
plot_blackbody_spectrum(temperature: float)
```

Plots the spectral radiance of a blackbody from
0.5 to 30 microns.

#### `plot_blackbody_spectrum_visible_nir`

```python
plot_blackbody_spectrum_visible_nir(temperature: float)
```

Plots the spectral radiance of a blackbody from
0.1 to 1 micron.

#### `sat_magnitude`

```python
sat_magnitude(size: float, range: float, angle: float, band: str) -> float
```

given a satellite size and a waveband and range
pull the brightness of the sun and the calibration from radiometry_data

### `sim_check.py`

#### `sim_check`

```python
sim_check(sim_data)
```

Prints a brief summary of what\'s present in a sim_data structure.


### `simulation.py`

#### `create_empty_simulation`

```python
create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> Dict[str, Any]
```

Initializes a minimal, empty data structure for a space simulation.

## Dependencies

*   `numpy`
*   `astropy`
*   `jplephem`
*   `sgp4`
*   `plotly`
*   `scipy`
*   `ipython`
