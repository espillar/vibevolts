
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from common import initialize_standard_simulation
from propagation import propagate_satellites
from plotting_3d import plot_3d_scatter
from constellation import geos

def demo_constellation() -> go.Figure:
    """
    Runs a demonstration of the constellation creation tools.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object for the satellite positions plot.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    # Create a constellation of 10 GEO satellites
    geos(sim_data, 10, 'satellites')

    # Propagate the satellites for 1 hour
    time_t1 = sim_start_time + timedelta(hours=1)
    sim_data = propagate_satellites(sim_data, time_t1)
    
    fig = plot_3d_scatter(
        positions=sim_data['satellites']['position'],
        title=f"GEO Constellation at {time_t1.isoformat()}",
        plot_time=time_t1,
        marker_size=5,
        trace_name='GEO Satellites'
    )
#    fig.show()
#    print(sim_data)
    return fig

if __name__ == '__main__':
    fig = demo_constellation()
    fig.show()
