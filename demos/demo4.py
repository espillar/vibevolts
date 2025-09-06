import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from propagation import readtle, propagate_satellites
from simulation import initializeStructures

def demo4() -> go.Figure:
    """
    Runs a demonstration plotting a single GEO satellite trajectory.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)

    tle_data = '''GEO-TRAJECTORY
1 90301U 25006A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90301   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
'''
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

    positions_over_time = []
    time_steps = np.arange(0, 24, 1)

    for hours in time_steps:
        prop_time = sim_start_time + timedelta(hours=int(hours))
        sim_data = propagate_satellites(sim_data, prop_time)
        positions_over_time.append(sim_data['satellites']['position'][0])

    positions_array = np.array(positions_over_time)

    print("\n--- Generating 3D plot for Demo 4 ---")
    earth_radius = 6378137.0
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='lines', line=dict(color='purple', width=4), name='Trajectory'
    ))

    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='markers', marker=dict(size=5, color='orange'),
        text=[f'T={t} hr' for t in time_steps], hoverinfo='text', name='Time Steps'
    ))

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

    return fig
