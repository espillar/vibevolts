import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from simulation import initializeStructures, POINTING_COUNT_IDX, POINTING_PLACE_IDX
from pointing import generate_pointing_sphere, update_satellite_pointing, pointing_place_update
from plotting_vectors import plot_pointing_vectors

def demo_pointing_sequence() -> go.Figure:
    """
    Demonstrates the satellite pointing sequence functionality.
    """
    print("\n--- Starting Demo: Pointing Sequence ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Initialize a simulation with 3 satellites
    sim_data = initializeStructures(
        num_satellites=3,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )

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

    # Initial plot
    vectors = sim_data['satellites']['pointing']
    fig.add_trace(go.Scatter3d(
        x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
        mode='markers', marker=dict(size=10, color=['red', 'green', 'blue']),
        name='T=0'
    ))

    # Simulation loop
    for t in range(1, 5):
        print(f"\n--- Time Step {t} ---")
        pointing_place_update(sim_data)
        update_satellite_pointing(sim_data)
        print("Pointing vectors:")
        print(sim_data['satellites']['pointing'])

        vectors = sim_data['satellites']['pointing']
        fig.add_trace(go.Scatter3d(
            x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
            mode='markers', marker=dict(size=10, color=['red', 'green', 'blue'], opacity=0.5),
            name=f'T={t}'
        ))

    # Add a unit sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', showscale=False, opacity=0.1))

    fig.update_layout(
        title="Satellite Pointing Sequence",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        )
    )

    return fig

if __name__ == '__main__':
    fig = demo_pointing_sequence()
    fig.show()
