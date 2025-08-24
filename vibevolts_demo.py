import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
from astropy.coordinates import solar_system_ephemeris

from vibevolts import (
    initializeStructures,
    celestial_update,
    readtle,
    propagate_satellites,
    create_exclusion_table,
    DETECTOR_SOLAR_EXCL_IDX,
    DETECTOR_LUNAR_EXCL_IDX,
    DETECTOR_EARTH_EXCL_IDX,
)
from visualization import plot_3d_scatter, plot_pointing_vectors


def initialize_standard_simulation(start_time: datetime) -> Dict[str, Any]:
    """
    Initializes a standard simulation with a predefined set of satellites.

    This function consolidates the TLE data used across various demos into
    a single setup function. It initializes the main data structure with a
    mix of LEO, GEO, and HEO satellites.

    Args:
        start_time: The timezone-aware datetime object for the simulation start.

    Returns:
        The fully initialized simulation data dictionary.
    """
    # Consolidated TLE data from all demos
    tle_data = """ISS (ZARYA)
1 25544U 98067A   25209.52203988  .00012111  00000+0  22159-3 0  9991
2 25544  51.6412 254.9961 0006733  98.4322 261.6813 15.49493393462383
NOAA 19
1 33591U 09005A   25209.38959223  .00000100  00000+0  97987-4 0  9993
2 33591  99.1533 244.3362 0013327 101.3725 258.7562 14.12510122810029
HST
1 20580U 90037B   25208.83160218  .00000113  00000+0  35999-4 0  9990
2 20580  28.4695 177.8391 0001259 138.5273 221.5822 15.09326468 23453
GEO-01
1 90001U 25001A   25209.50000000  .00000000  00000-0  00000-0 0  9991
2 90001   0.0500   0.0000 0001000   0.0000   0.0000  1.00270000    12
GEO-02
1 90002U 25001B   25209.50000000  .00000000  00000-0  00000-0 0  9992
2 90002   0.0500  36.0000 0001000   0.0000   0.0000  1.00270000    13
GEO-03
1 90003U 25001C   25209.50000000  .00000000  00000-0  00000-0 0  9993
2 90003   0.0500  72.0000 0001000   0.0000  45.0000  1.00270000    14
GEO-04
1 90004U 25001D   25209.50000000  .00000000  00000-0  00000-0 0  9994
2 90004   0.0500 108.0000 0001000   0.0000  90.0000  1.00270000    15
GEO-05
1 90005U 25001E   25209.50000000  .00000000  00000-0  00000-0 0  9995
2 90005   0.0500 144.0000 0001000   0.0000 135.0000  1.00270000    16
GEO-06
1 90006U 25001F   25209.50000000  .00000000  00000-0  00000-0 0  9996
2 90006   0.0500 180.0000 0001000   0.0000 180.0000  1.00270000    17
GEO-07
1 90007U 25001G   25209.50000000  .00000000  00000-0  00000-0 0  9997
2 90007   0.0500 216.0000 0001000   0.0000 225.0000  1.00270000    18
GEO-08
1 90008U 25001H   25209.50000000  .00000000  00000-0  00000-0 0  9998
2 90008   0.0500 252.0000 0001000   0.0000 270.0000  1.00270000    19
GEO-09
1 90009U 25001I   25209.50000000  .00000000  00000-0  00000-0 0  9999
2 90009   0.0500 288.0000 0001000   0.0000 315.0000  1.00270000    10
GEO-10
1 90010U 25001J   25209.50000000  .00000000  00000-0  00000-0 0  9990
2 90010   0.0500 324.0000 0001000   0.0000 360.0000  1.00270000    11
HEO-01 (MOLNIYA)
1 90011U 25002A   25209.50000000  .00000000  00000-0  00000-0 0  9995
2 90011  63.4000  50.0000 7500000  270.0000  45.0000  2.00560000    13
HEO-02 (MOLNIYA)
1 90012U 25002B   25209.50000000  .00000000  00000-0  00000-0 0  9996
2 90012  63.4000 110.0000 7500000  270.0000  90.0000  2.00560000    14
HEO-03 (MOLNIYA)
1 90013U 25002C   25209.50000000  .00000000  00000-0  00000-0 0  9997
2 90013  63.4000 170.0000 7500000  270.0000 135.0000  2.00560000    15
HEO-04 (MOLNIYA)
1 90014U 25002D   25209.50000000  .00000000  00000-0  00000-0 0  9998
2 90014  63.4000 230.0000 7500000  270.0000 180.0000  2.00560000    16
HEO-05 (MOLNIYA)
1 90015U 25002E   25209.50000000  .00000000  00000-0  00000-0 0  9999
2 90015  63.4000 290.0000 7500000  270.0000 225.0000  2.00560000    17
LEO-01 (POLAR)
1 90016U 25003A   25209.50000000  .00000000  00000-0  00000-0 0  9990
2 90016  98.0000 120.0000 0010000  90.0000  20.0000 14.50000000    18
LEO-02 (POLAR)
1 90017U 25003B   25209.50000000  .00000000  00000-0  00000-0 0  9991
2 90017  98.0000 240.0000 0010000  90.0000  40.0000 14.50000000    19
LEO-03
1 90018U 25003C   25209.50000000  .00000000  00000-0  00000-0 0  9992
2 90018  45.0000  80.0000 0010000  45.0000  60.0000 15.00000000    10
LEO-04
1 90019U 25003D   25209.50000000  .00000000  00000-0  00000-0 0  9993
2 90019  45.0000 200.0000 0010000  45.0000  80.0000 15.00000000    11
LEO-05
1 90020U 25003E   25209.50000000  .00000000  00000-0  00000-0 0  9994
2 90020  28.5000  10.0000 0010000  10.0000 100.0000 15.50000000    12
"""
    dummy_tle_path = "standard_tle.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    # Read the TLEs to determine the number of satellites
    orbital_elements, epochs = readtle(dummy_tle_path)
    num_sats = len(orbital_elements)

    print(f"Initializing standard simulation with {num_sats} satellites.")

    sim_data = initializeStructures(
        num_satellites=num_sats,
        num_observatories=0,
        num_red_satellites=0,
        start_time=start_time
    )

    # Populate the satellite orbital elements from the TLE file
    sim_data['satellites']['orbital_elements'] = orbital_elements
    sim_data['satellites']['epochs'] = epochs

    # Ensure the required ephemeris data is available
    solar_system_ephemeris.set('jpl')

    return sim_data

def demo1():
    """
    Runs a full demonstration of the simulation tools: initialization,
    celestial updates, TLE reading, satellite propagation, and 3D plotting.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    # --- Celestial Body Updates ---
    sim_data = celestial_update(sim_data, sim_start_time)
    print("\n--- Celestial Positions at Start Time ---")
    print(f"Time: {sim_start_time.isoformat()}")
    print(sim_data['celestial']['position'])

    # --- Satellite Propagation and Plotting ---
    time_t1 = sim_start_time + timedelta(hours=1, minutes=30)
    print(f"\n--- Propagating satellites to T1: {time_t1.isoformat()} ---")
    sim_data = propagate_satellites(sim_data, time_t1)

    plot_3d_scatter(
        positions=sim_data['satellites']['position'],
        title=f"Satellite Positions at {time_t1.isoformat()}",
        plot_time=time_t1,
        marker_size=2,
        trace_name='Satellites'
    )

def demo2():
    """
    Runs a second demonstration with 10 LEO satellites, plotting their
    positions and celestial vectors at 0, 60, and 300 seconds.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    # --- Satellite and Celestial Propagation ---
    time_t0 = sim_start_time
    sim_data = propagate_satellites(sim_data, time_t0)
    positions_t0 = sim_data['satellites']['position'].copy()
    sim_data = celestial_update(sim_data, time_t0)
    celestial_pos_t0 = sim_data['celestial']['position'].copy()

    time_t1 = sim_start_time + timedelta(seconds=300)
    sim_data = propagate_satellites(sim_data, time_t1)
    positions_t1 = sim_data['satellites']['position'].copy()
    sim_data = celestial_update(sim_data, time_t1)
    celestial_pos_t1 = sim_data['celestial']['position'].copy()

    # --- 3D Plotting of All Time Steps ---
    print("\n--- Generating 3D plot for Demo 2 ---")
    earth_radius = 6378137.0
    vector_scale = 10 * earth_radius
    fig = go.Figure()

    # Add positions at T0 and T1
    fig.add_trace(go.Scatter3d(
        x=positions_t0[:, 0], y=positions_t0[:, 1], z=positions_t0[:, 2],
        mode='markers', marker=dict(size=5, color='blue'), name='Sats (T=0s)'
    ))
    fig.add_trace(go.Scatter3d(
        x=positions_t1[:, 0], y=positions_t1[:, 1], z=positions_t1[:, 2],
        mode='markers', marker=dict(size=5, color='red'), name='Sats (T=300s)'
    ))

    # Add Celestial Vectors
    sun_vec_t0 = celestial_pos_t0[0] / np.linalg.norm(celestial_pos_t0[0]) * vector_scale
    moon_vec_t0 = celestial_pos_t0[1] / np.linalg.norm(celestial_pos_t0[1]) * vector_scale
    fig.add_trace(go.Scatter3d(x=[0, sun_vec_t0[0]], y=[0, sun_vec_t0[1]], z=[0, sun_vec_t0[2]], mode='lines', line=dict(color='orange', width=4), name='Sun (T=0s)'))
    fig.add_trace(go.Scatter3d(x=[0, moon_vec_t0[0]], y=[0, moon_vec_t0[1]], z=[0, moon_vec_t0[2]], mode='lines', line=dict(color='gray', width=4), name='Moon (T=0s)'))

    sun_vec_t1 = celestial_pos_t1[0] / np.linalg.norm(celestial_pos_t1[0]) * vector_scale
    moon_vec_t1 = celestial_pos_t1[1] / np.linalg.norm(celestial_pos_t1[1]) * vector_scale
    fig.add_trace(go.Scatter3d(x=[0, sun_vec_t1[0]], y=[0, sun_vec_t1[1]], z=[0, sun_vec_t1[2]], mode='lines', line=dict(color='yellow', width=4), name='Sun (T=300s)'))
    fig.add_trace(go.Scatter3d(x=[0, moon_vec_t1[0]], y=[0, moon_vec_t1[1]], z=[0, moon_vec_t1[2]], mode='lines', line=dict(color='lightgray', width=4), name='Moon (T=300s)'))

    # Add Earth
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))

    fig.update_layout(
        title=f"Satellite and Celestial Positions at 0 and 300 seconds",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(r=20, b=10, l=10, t=40)
    )
    fig.show()

def demo3():
    """
    Runs a third demonstration with a single LEO satellite, plotting its
    trajectory over 90 minutes.
    """
    # Define the simulation start time.
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)

    # --- TLE Reading and Initialization ---
    # Create a dummy TLE file for one LEO satellite
    tle_data = """LEO-TRAJECTORY
1 90201U 25005A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90201  51.6000  10.0000 0010000  90.0000  20.0000 15.50000000    11
"""
    dummy_tle_path = "dummy_tle3.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    orbital_elements_from_tle, epochs_from_tle = readtle(dummy_tle_path)
    num_sats = len(orbital_elements_from_tle)

    print(f"\n--- Starting Demo 3 ---")
    print(f"Initializing structures for {num_sats} LEO satellite.")

    sim_data = initializeStructures(
        num_satellites=num_sats,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )
    sim_data['satellites']['orbital_elements'] = orbital_elements_from_tle
    sim_data['satellites']['epochs'] = epochs_from_tle

    # --- Satellite Propagation ---
    positions_over_time = []
    time_steps = np.arange(0, 91, 10) # 0 to 90 minutes in 10 minute steps

    for minutes in time_steps:
        prop_time = sim_start_time + timedelta(minutes=int(minutes))
        sim_data = propagate_satellites(sim_data, prop_time)
        positions_over_time.append(sim_data['satellites']['position'][0])

    positions_array = np.array(positions_over_time)

    # --- 3D Plotting of the Trajectory ---
    print("\n--- Generating 3D plot for Demo 3 ---")
    earth_radius = 6378137.0
    fig = go.Figure()

    # Add the trajectory line
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='lines', line=dict(color='red', width=4), name='Trajectory'
    ))

    # Add markers for each time step
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='markers', marker=dict(size=5, color='blue'),
        text=[f'T={t} min' for t in time_steps], hoverinfo='text', name='Time Steps'
    ))

    # Add Earth and reference markers
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))

    fig.update_layout(
        title=f"Single LEO Satellite Trajectory over 90 Minutes",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Trace'
    )
    fig.show()

def demo4():
    """
    Runs a fourth demonstration with a single GEO satellite, plotting its
    trajectory over 23 hours.
    """
    # Define the simulation start time.
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)

    # --- TLE Reading and Initialization ---
    # Create a dummy TLE file for one GEO satellite
    tle_data = """GEO-TRAJECTORY
1 90301U 25006A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90301   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
"""
    dummy_tle_path = "dummy_tle4.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    orbital_elements_from_tle, epochs_from_tle = readtle(dummy_tle_path)
    num_sats = len(orbital_elements_from_tle)

    print(f"\n--- Starting Demo 4 ---")
    print(f"Initializing structures for {num_sats} GEO satellite.")

    sim_data = initializeStructures(
        num_satellites=num_sats,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )
    sim_data['satellites']['orbital_elements'] = orbital_elements_from_tle
    sim_data['satellites']['epochs'] = epochs_from_tle

    # --- Satellite Propagation ---
    positions_over_time = []
    time_steps = np.arange(0, 24, 1) # 0 to 23 hours in 1 hour steps

    for hours in time_steps:
        prop_time = sim_start_time + timedelta(hours=int(hours))
        sim_data = propagate_satellites(sim_data, prop_time)
        positions_over_time.append(sim_data['satellites']['position'][0])

    positions_array = np.array(positions_over_time)

    # --- 3D Plotting of the Trajectory ---
    print("\n--- Generating 3D plot for Demo 4 ---")
    earth_radius = 6378137.0
    fig = go.Figure()

    # Add the trajectory line
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='lines', line=dict(color='purple', width=4), name='Trajectory'
    ))

    # Add markers for each time step
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='markers', marker=dict(size=5, color='orange'),
        text=[f'T={t} hr' for t in time_steps], hoverinfo='text', name='Time Steps'
    ))

    # Add Earth and reference markers
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))

    fig.update_layout(
        title=f"Single GEO Satellite Trajectory over 23 Hours",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Trace'
    )
    fig.show()

def demo_fixedpoints():
    """
    Demonstrates the fixedpoints data structure by plotting it in 3D.
    """
    print("\n--- Starting Demo Fixedpoints ---")
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)

    # Initialize the data structures. We don't need any satellites for this demo.
    sim_data = initializeStructures(
        num_satellites=0,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )

    fixed_positions = sim_data['fixedpoints']['position']

    print(f"Plotting {len(fixed_positions)} fixed points.")

    plot_3d_scatter(
        positions=fixed_positions,
        title="Fixed Points Distribution",
        plot_time=sim_start_time,
        labels=[f"Point {i}" for i in range(len(fixed_positions))],
        marker_size=1,
        trace_name='Fixed Points'
    )

def demo_exclusion_table():
    """
    Demonstrates the creation and visualization of the exclusion table.

    This function sets up a scenario with 20 LEO satellites and calculates
    their exclusion status against the first 200 fixed points. It then
    displays this table as a red and green heatmap.
    """
    print("\n--- Starting Demo: Exclusion Table Visualization ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    # --- Propagate Satellites to the simulation start time ---
    # This step is crucial. Without it, satellite positions are [0,0,0].
    print("Propagating satellites to the simulation start time...")
    sim_data = propagate_satellites(sim_data, sim_start_time)

    # Set fixed exclusion angles for all satellites (in radians)
    # Approx 30 degrees for Sun/Moon, 10 degrees for Earth limb
    sim_data['satellites']['detector'][:, DETECTOR_SOLAR_EXCL_IDX] = np.deg2rad(30)
    sim_data['satellites']['detector'][:, DETECTOR_LUNAR_EXCL_IDX] = np.deg2rad(30)
    sim_data['satellites']['detector'][:, DETECTOR_EARTH_EXCL_IDX] = np.deg2rad(10)

    # --- Update Celestial Positions ---
    print("Calculating celestial body positions...")
    sim_data = celestial_update(sim_data, sim_start_time)

    # --- Create the Exclusion Table ---
    # To make the demo run faster and the plot readable, we'll only check
    # against the first 200 fixed points.
    print("Generating exclusion table (this may take a moment)...")
    original_fixed_points = sim_data['fixedpoints']['position']
    sim_data['fixedpoints']['position'] = original_fixed_points[:200]

    exclusion_matrix = create_exclusion_table(sim_data)

    # Restore original fixed points if needed elsewhere
    sim_data['fixedpoints']['position'] = original_fixed_points

    print("Exclusion table generated.")

    # --- Visualize the Table as a Heatmap ---
    print("Displaying results as a heatmap...")
    fig = go.Figure(data=go.Heatmap(
        z=exclusion_matrix,
        colorscale=[[0, 'green'], [1, 'red']], # 0 is green (clear), 1 is red (excluded)
        showscale=False # Hide the color bar
    ))

    fig.update_layout(
        title='Satellite vs. Fixed Point Exclusion Status (0=Clear, 1=Excluded)',
        xaxis_title='Fixed Point Index',
        yaxis_title='Satellite Index',
        yaxis=dict(autorange='reversed') # Puts satellite 0 at the top
    )
    fig.show()

def demo_pointing_plot():
    """
    Demonstrates the plot_pointing_vectors function with the standard satellite set.
    """
    print("\n--- Starting Demo: Pointing Vector Plot ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)
    num_sats = sim_data['counts']['satellites']

    # Propagate satellites to the start time
    sim_data = propagate_satellites(sim_data, sim_start_time)

    # Assign random pointing vectors to each satellite
    # This ensures the pointing vectors are initialized for the plot.
    random_vectors = np.random.rand(num_sats, 3) - 0.5
    norms = np.linalg.norm(random_vectors, axis=1)[:, np.newaxis]
    # Avoid division by zero if a random vector is [0,0,0]
    norms[norms == 0] = 1.0
    sim_data['satellites']['pointing'] = random_vectors / norms

    # Call the plotting function
    plot_pointing_vectors(
        data_struct=sim_data,
        title="Satellite Positions with Random Pointing Vectors",
        plot_time=sim_start_time
    )

# --- Main Execution Block ---
if __name__ == '__main__':

    # Each demo can be run independently.
    # To avoid generating multiple plots, comment out the ones you don't need.

    # demo1()
    # demo2()
    # demo3()
    # demo4()
    demo_fixedpoints()
    # demo_exclusion_table()
    # demo_pointing_plot()

