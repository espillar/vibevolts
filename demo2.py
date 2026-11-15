import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from demo_common import initialize_standard_simulation
from constants import EARTH_RADIUS
from celestialbodies import celestial_update
from propagation import propagate_satellites
from plotting_3d import (
    add_sphere_trace,
    add_vector_traces,
    apply_3d_scene_layout,
    tag_demo_figure,
)

def demo2() -> go.Figure:
    """
    Runs a demonstration plotting satellite and celestial positions.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    # --- Satellite and Celestial Propagation ---
    positions_t0 = sim_data.satellites.position.copy()
    sim_data = celestial_update(sim_data, sim_start_time)
    celestial_pos_t0 = sim_data.celestial.position.copy()

    time_t1 = sim_start_time + timedelta(seconds=300)
    sim_data = propagate_satellites(sim_data, time_t1)
    positions_t1 = sim_data.satellites.position.copy()
    sim_data = celestial_update(sim_data, time_t1)
    celestial_pos_t1 = sim_data.celestial.position.copy()

    # --- 3D Plotting of All Time Steps ---
    print("\n--- Generating 3D plot for Demo 2 ---")
    vector_scale = 10 * EARTH_RADIUS
    fig = go.Figure()

    snapshots = [
        ('0s', positions_t0, celestial_pos_t0, 'blue', 'orange', 'gray'),
        ('300s', positions_t1, celestial_pos_t1, 'red', 'yellow', 'lightgray'),
    ]
    for label, sat_pos, cel_pos, sat_color, sun_color, moon_color in snapshots:
        fig.add_trace(go.Scatter3d(
            x=sat_pos[:, 0], y=sat_pos[:, 1], z=sat_pos[:, 2],
            mode='markers', marker=dict(size=5, color=sat_color), name=f'Sats (T={label})'
        ))
        add_vector_traces(fig, [0, 0, 0], cel_pos[0], scale=vector_scale,
                          color=sun_color, width=4, name=f'Sun (T={label})')
        add_vector_traces(fig, [0, 0, 0], cel_pos[1], scale=vector_scale,
                          color=moon_color, width=4, name=f'Moon (T={label})')

    add_sphere_trace(fig, radius=EARTH_RADIUS, name='Earth')
    apply_3d_scene_layout(fig, "Satellite and Celestial Positions at 0 and 300 seconds")
    return tag_demo_figure(fig, "demo2")