import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from simulation import create_empty_simulation
from celestialbodies import add_celestial_bodies
from observatories import add_observatories
from propagation import add_satellites_from_tle
from celestialbodies import celestial_update
from constants import POINTING_COUNT_IDX, POINTING_PLACE_IDX
from pointing import generate_pointing_sphere, update_satellite_pointing
from plotting_vectors import plot_pointing_vectors

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
    
    # Create a dummy TLE file for 1 satellite
    tle_data = """SAT-1
1 90401U 25007A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90401   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
"""
    dummy_tle_path = "dummy_tle_pointing.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    add_satellites_from_tle(sim_data, dummy_tle_path, 'satellites')

    # Generate pointing spheres
    generate_pointing_sphere(sim_data, 100)

    # Assign pointing counts to satellites
    pointing_state = sim_data['satellites']['pointing_state']
    pointing_state[0, POINTING_COUNT_IDX] = 100

    print("Initial pointing vectors:")
    update_satellite_pointing(sim_data)
    print(sim_data['satellites']['pointing'])

    # --- Create a figure to animate ---
    fig = go.Figure(
        layout=go.Layout(
            width=1000,  # Set width to 1000 pixels
            height=800  # Set height to 800 pixels
        )
    )
    
    # Store trajectory of each satellite
    trajectories = [[]]
    
    # Initial plot (T=0)
    vectors = sim_data['satellites']['pointing']
    trajectories[0].append(vectors[0].copy())

    # Simulation loop for 30 steps (T=0 to T=29)
    for t in range(1, 30):
        print(f"\n--- Time Step {t} ---")
        current_time = sim_start_time + timedelta(seconds=t * sim_data['delta_time'])
        sim_data = celestial_update(sim_data, current_time)
        update_satellite_pointing(sim_data)
        vectors = sim_data['satellites']['pointing']
        trajectories[0].append(vectors[0].copy())

    # --- Plotting ---
    colors = ['red']
    sat_names = ['Satellite 1 (30 steps)']
    time_steps = list(range(30))

    x_coords = [p[0] for p in trajectories[0]]
    y_coords = [p[1] for p in trajectories[0]]
    z_coords = [p[2] for p in trajectories[0]]
    
    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='lines+markers',
        marker=dict(
            size=[(j + 2) * 2 for j in time_steps],
            color=time_steps,
            colorscale='Viridis',
            showscale=False,
            opacity=0.8
        ),
        line=dict(
            color=colors[0],
            width=2
        ),
        name=sat_names[0]
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
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', showscale=False, opacity=0.1, name='Unit Sphere'))

    # Add a caption to the plot
    caption = """
    <b>Satellite Pointing Sequence:</b><br>
    - <b>Satellite 1 (Red)</b>: 30-step pointing sequence. Changes position at each time step.
    """

    fig.update_layout(
        title="Satellite Pointing Sequence Over 30 Time Steps",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        annotations=[
            dict(
                text=caption,
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1
            )
        ]
    )

    return fig

if __name__ == '__main__':
    fig = demo_pointing_sequence()
    fig.show()