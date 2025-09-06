import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from demos.common import initialize_standard_simulation
from propagation import celestial_update, propagate_satellites

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
    positions_t0 = sim_data['satellites']['position'].copy()
    sim_data = celestial_update(sim_data, sim_start_time)
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

    fig.add_trace(go.Scatter3d(
        x=positions_t0[:, 0], y=positions_t0[:, 1], z=positions_t0[:, 2],
        mode='markers', marker=dict(size=5, color='blue'), name='Sats (T=0s)'
    ))
    fig.add_trace(go.Scatter3d(
        x=positions_t1[:, 0], y=positions_t1[:, 1], z=positions_t1[:, 2],
        mode='markers', marker=dict(size=5, color='red'), name='Sats (T=300s)'
    ))

    sun_vec_t0 = celestial_pos_t0[0] / np.linalg.norm(celestial_pos_t0[0]) * vector_scale
    moon_vec_t0 = celestial_pos_t0[1] / np.linalg.norm(celestial_pos_t0[1]) * vector_scale
    fig.add_trace(go.Scatter3d(x=[0, sun_vec_t0[0]], y=[0, sun_vec_t0[1]], z=[0, sun_vec_t0[2]], mode='lines', line=dict(color='orange', width=4), name='Sun (T=0s)'))
    fig.add_trace(go.Scatter3d(x=[0, moon_vec_t0[0]], y=[0, moon_vec_t0[1]], z=[0, moon_vec_t0[2]], mode='lines', line=dict(color='gray', width=4), name='Moon (T=0s)'))

    sun_vec_t1 = celestial_pos_t1[0] / np.linalg.norm(celestial_pos_t1[0]) * vector_scale
    moon_vec_t1 = celestial_pos_t1[1] / np.linalg.norm(celestial_pos_t1[1]) * vector_scale
    fig.add_trace(go.Scatter3d(x=[0, sun_vec_t1[0]], y=[0, sun_vec_t1[1]], z=[0, sun_vec_t1[2]], mode='lines', line=dict(color='yellow', width=4), name='Sun (T=300s)'))
    fig.add_trace(go.Scatter3d(x=[0, moon_vec_t1[0]], y=[0, moon_vec_t1[1]], z=[0, moon_vec_t1[2]], mode='lines', line=dict(color='lightgray', width=4), name='Moon (T=300s)'))

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

    return fig
