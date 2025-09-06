from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from demos.common import initialize_standard_simulation
from propagation import celestial_update, propagate_satellites
from plotting_3d import plot_3d_scatter

def demo1() -> go.Figure:
    """
    Runs a full demonstration of the simulation tools.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object for the satellite positions plot.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    # --- Celestial Body Updates ---
    sim_data = celestial_update(sim_data, sim_start_time)
    print("\n--- Celestial Positions at Start Time ---")
    print(f"Time: {sim_start_time.isoformat()}")
    print(sim_data['celestial']['position'])

    # --- Satellite Propagation and Plotting ---
    time_t1 = sim_start_time + timedelta(hours=1, minutes=30)
    print(f"\n--- Propagating satellites to T1: {time_t1.isoformat()} ---")
    sim_data = propagate_satellites(sim_data, time_t1)

    fig = plot_3d_scatter(
        positions=sim_data['satellites']['position'],
        title=f"Satellite Positions at {time_t1.isoformat()}",
        plot_time=time_t1,
        marker_size=2,
        trace_name='Satellites'
    )
    return fig
