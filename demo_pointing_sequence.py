import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from constants import POINTING_COUNT_IDX
from minimalsimulation import create_empty_simulation
from celestialbodies import add_celestial_bodies, celestial_update
from propagation import add_satellites_from_tle
from demo_common import get_sample_geo_tles
from pointing import generate_pointing_sphere, update_detector_pointing
from plotting_3d import add_sphere_trace, apply_3d_scene_layout, tag_demo_figure

def demo_pointing_sequence() -> go.Figure:
    """
    Demonstrates the satellite pointing sequence functionality.

    This demo initializes a simulation with three satellites and visualizes the
    progression of their pointing vectors over five time steps (T=0 to T=4).

    - Satellite 1 (Red): Assigned a pointing sequence with 10 steps. Its
      pointing vector is expected to update at each time step.
    - Satellite 2 (Green): Assigned a pointing sequence with 20 steps. Its
      pointing vector is also expected to update at each time step.
    - Satellite 3 (Blue): Assigned a pointing sequence with 0 steps. Its
      pointing vector should remain fixed in its initial, randomly assigned
      direction.

    The plot displays the history of these pointing vectors on a unit sphere.
    Each satellite's path is shown with a distinct line color (Red, Green, Blue).
    The progression of time is indicated by the increasing size and changing
    color of the markers—from dark blue (T=0) to yellow (T=4)—as shown by the
    colorbar. A descriptive caption is included below the plot to explain the
    visualization.
    """
    print("\n--- Starting Demo: Pointing Sequence ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Initialize a simulation with 3 satellites
    sim_data = create_empty_simulation(sim_start_time)
    add_celestial_bodies(sim_data)
    
    add_satellites_from_tle(sim_data, get_sample_geo_tles(3), 'satellites')

    # Generate pointing spheres
    generate_pointing_sphere(sim_data, 100)
    generate_pointing_sphere(sim_data, 20)

    # Assign pointing counts to satellites using (row, col) indexing
    pointing_state = sim_data.satellites.detector.pointing_state
    pointing_state[0, POINTING_COUNT_IDX] = 100
    pointing_state[1, POINTING_COUNT_IDX] = 20
    pointing_state[2, POINTING_COUNT_IDX] = 0

    # Collect pointing trajectories for 30 steps (T=0 to T=29)
    n_steps = 30
    trajectories = np.zeros((n_steps, 3, 3))
    print("Initial pointing vectors:")
    for t in range(n_steps):
        if t > 0:
            print(f"\n--- Time Step {t} ---")
            current_time = sim_start_time + timedelta(seconds=t * sim_data.delta_time)
            sim_data = celestial_update(sim_data, current_time)
        update_detector_pointing(sim_data)
        trajectories[t] = sim_data.satellites.detector.pointing.copy()

    # --- Plotting ---
    fig = go.Figure(layout=go.Layout(width=1000, height=800))
    colors = ['red', 'green', 'blue']
    sat_names = [
        'Satellite 1 (100 steps)',
        'Satellite 2 (20 steps)',
        'Satellite 3 (0 steps - Fixed)'
    ]
    time_steps = list(range(n_steps))

    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=trajectories[:, i, 0], y=trajectories[:, i, 1], z=trajectories[:, i, 2],
            mode='lines+markers',
            marker=dict(
                size=[(j + 2) * 2 for j in time_steps],
                color=time_steps,
                colorscale='Viridis',
                showscale=False,
                opacity=0.8
            ),
            line=dict(color=colors[i], width=2),
            name=sat_names[i]
        ))

    # Add a dummy trace to create a single, shared colorbar for the time steps
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            colorscale='Viridis',
            showscale=True,
            cmin=0,
            cmax=29,
            colorbar=dict(
                title='Time Step',
                tickvals=time_steps,
                ticktext=[f'T={t}' for t in time_steps]
            )
        ),
        hoverinfo='none',
        name='',
        showlegend=False
    ))

    # Add a unit sphere for reference
    add_sphere_trace(fig, radius=1.0, name='Unit Sphere', opacity=0.1)

    # Add a caption to the plot
    caption = """
    <b>Satellite Pointing Sequence:</b><br>
    - <b>Satellite 1 (Red)</b>: 100-step pointing sequence. Updates frequently.<br>
    - <b>Satellite 2 (Green)</b>: 20-step pointing sequence. Updates frequently.<br>
    - <b>Satellite 3 (Blue)</b>: 0-step pointing sequence. Remains fixed.
    """

    apply_3d_scene_layout(
        fig,
        "Satellite Pointing Sequence Over 30 Time Steps",
        axis_unit='',
        annotations=[dict(text=caption, showarrow=False, xref="paper", yref="paper", x=0.5, y=-0.1)],
    )
    return tag_demo_figure(fig, "demo_pointing_sequence")

if __name__ == '__main__':
    fig = demo_pointing_sequence()
    fig.show()
