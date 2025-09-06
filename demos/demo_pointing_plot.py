from datetime import datetime, timezone
import plotly.graph_objects as go

from demos.common import initialize_standard_simulation
from plotting_vectors import plot_pointing_vectors

def demo_pointing_plot() -> go.Figure:
    """
    Demonstrates the plot_pointing_vectors function.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object.
    """
    print("\n--- Starting Demo: Pointing Vector Plot ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    fig = plot_pointing_vectors(
        data_struct=sim_data,
        title="Satellite Positions with Radially Outward Pointing Vectors",
        plot_time=sim_start_time
    )
    return fig
