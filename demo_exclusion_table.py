import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go

from demo_common import initialize_standard_simulation
from celestialbodies import celestial_update
from exclusion import update_exclusion_table

def demo_exclusion_table() -> go.Figure:
    """
    Demonstrates the creation and visualization of the exclusion table.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
    print("\n--- Starting Demo: Exclusion Table Visualization ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    sim_data['detector'].solarEx[:] = np.deg2rad(30)
    sim_data['detector'].lunarex[:] = np.deg2rad(30)
    sim_data['detector'].earthEx[:] = np.deg2rad(10)

    print("Calculating celestial body positions...")
    sim_data = celestial_update(sim_data, sim_start_time)

    print("Generating exclusion table (this may take a moment)...")
    original_fixed_points = sim_data['fixedpoints']['position']
    sim_data['fixedpoints']['position'] = original_fixed_points[:200]

    update_exclusion_table(sim_data)
    exclusion_matrix = sim_data['fixedpoints']['exclusion']

    sim_data['fixedpoints']['position'] = original_fixed_points

    print("Exclusion table generated.")

    print("Displaying results as a heatmap...")
    fig = go.Figure(data=go.Heatmap(
        z=exclusion_matrix.T,
        colorscale=[[0, 'green'], [1, 'red']],
        showscale=False
    ))

    fig.update_layout(
        title='Satellite vs. Fixed Point Exclusion Status (0=Clear, 1=Excluded)',
        xaxis_title='Fixed Point Index',
        yaxis_title='Satellite Index',
        yaxis=dict(autorange='reversed')
    )

    return fig
