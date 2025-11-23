import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from simulation import create_empty_simulation
from observatories import add_observatories
from propagation import add_satellites_from_tle
from constants import POINTING_COUNT_IDX, POINTING_PLACE_IDX
from pointing import generate_pointing_sphere, update_satellite_pointing, pointing_place_update
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
    
    # Create a dummy TLE file for 3 satellites
    tle_data = """SAT-1
1 90401U 25007A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90401   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
SAT-2
1 90402U 25007B   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90402   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
SAT-3
1 90403U 25007C   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90403   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
"""
    dummy_tle_path = "dummy_tle_pointing.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    add_satellites_from_tle(sim_data, dummy_tle_path, 'satellites')

    # Generate pointing spheres
    generate_pointing_sphere(sim_data, 10)
    generate_pointing_sphere(sim_data, 20)

    # Assign pointing counts to satellites
    pointing_state = sim_data['satellites']['pointing_state']
    pointing_state[0, POINTING_COUNT_IDX] = 10
    pointing_state[1, POINTING_COUNT_IDX] = 20
    # Satellite 2 will have a pointing_count of 0 and should not move

    print("Initial pointing vectors:")
    update_satellite_pointing(sim_data)
    print(sim_data['satellites']['pointing'])

    # --- Create a figure to animate ---
    fig = go.Figure()
    
    # Store trajectory of each satellite
    trajectories = [[] for _ in range(3)]
    
    # Initial plot (T=0)
    vectors = sim_data['satellites']['pointing']
    for i in range(3):
        trajectories[i].append(vectors[i].copy())

    # Simulation loop
    for t in range(1, 5):
        print(f"\n--- Time Step {t} ---")
        pointing_place_update(sim_data)
        update_satellite_pointing(sim_data)
        vectors = sim_data['satellites']['pointing']
        for i in range(3):
            trajectories[i].append(vectors[i].copy())

    # --- Plotting ---
    colors = ['red', 'green', 'blue']
    sat_names = ['Satellite 1 (10 steps)', 'Satellite 2 (20 steps)', 'Satellite 3 (0 steps)']
    time_steps = list(range(5))

    for i in range(3):
        x_coords = [p[0] for p in trajectories[i]]
        y_coords = [p[1] for p in trajectories[i]]
        z_coords = [p[2] for p in trajectories[i]]
        
        # For Satellite 3 (no steps), plot only the first point as it doesn't move
        if i == 2:
            x_coords, y_coords, z_coords = [x_coords[0]], [y_coords[0]], [z_coords[0]]

        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines+markers',
            marker=dict(
                size=[(j + 2) * 2 for j in time_steps] if i < 2 else 6,  # Keep marker size constant for non-moving satellite
                color=time_steps if i < 2 else 'blue', # Use time-based color for moving, static for non-moving
                colorscale='Viridis',
                showscale=False,  # A single colorbar will be added later
                opacity=0.8
            ),
            line=dict(
                color=colors[i],
                width=2
            ),
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
            cmax=4,
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
    - <b>Satellite 1 (Red)</b>: 10-step pointing sequence. Changes position at each time step.<br>
    - <b>Satellite 2 (Green)</b>: 20-step pointing sequence. Changes position at each time step.<br>
    - <b>Satellite 3 (Blue)</b>: 0-step pointing sequence. Remains fixed.
    """

    fig.update_layout(
        title="Satellite Pointing Sequence Over 5 Time Steps",
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