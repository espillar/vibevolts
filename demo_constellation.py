from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from astropy.coordinates import solar_system_ephemeris

from simulation import create_empty_simulation, add_celestial_bodies, add_fixed_points
from observatories import add_observatories
from propagation import propagate_satellites_new
from plotting_3d import plot_3d_scatter
from constellation import geos
from sim_check import sim_check

def demo_constellation() -> go.Figure:
    """
    Runs a demonstration of the constellation creation tools.

    This function generates and returns a Plotly figure object but does not
    display it. The caller is responsible for rendering the plot.

    Returns:
        The Plotly figure object for the satellite positions plot.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)

    add_observatories(sim_data, 0)
    add_celestial_bodies(sim_data)
    add_fixed_points(sim_data, 100)
    sim_check(sim_data)
    solar_system_ephemeris.set('jpl')

    sim_data = propagate_satellites_new(sim_data, sim_start_time)

    num_sats_before = sim_data['counts'].get('satellites', 0)
    geos(sim_data, 10, 0.1)
    
    # Propagate the satellites for 1 hour
    time_t1 = sim_start_time + timedelta(hours=1)
    sim_data = propagate_satellites_new(sim_data, time_t1)

    sim_check(sim_data)
    fig = plot_3d_scatter(
        positions=sim_data['satellites']['position'][num_sats_before:],
        title=f"GEO Constellation at {time_t1.isoformat()} from demo_constellation",
        plot_time=time_t1,
        marker_size=5,
        trace_name='GEO Satellites'
    )
    return fig

if __name__ == '__main__':
    fig = demo_constellation()
    fig.show()
