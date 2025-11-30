from datetime import datetime, timezone
import plotly.graph_objects as go

from simulation import create_empty_simulation
from targets import add_fixed_points
from plotting_3d import plot_3d_scatter

def demo_fixedpoints() -> go.Figure:
    """
    Demonstrates the fixedpoints data structure by plotting it in 3D.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
    print("\n--- Starting Demo Fixedpoints ---")
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)

    sim_data = create_empty_simulation(sim_start_time)
    add_fixed_points(sim_data, 100)

    fixed_positions = sim_data['fixedpoints']['position']

    print(f"Plotting {len(fixed_positions)} fixed points.")

    fig = plot_3d_scatter(
        positions=fixed_positions,
        title="Fixed Points Distribution",
        plot_time=sim_start_time,
        labels=[f"Point {i}" for i in range(len(fixed_positions))],
        marker_size=3,
        trace_name='Fixed Points'
    )
    return fig



