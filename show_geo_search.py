import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go

from simulation import create_empty_simulation, add_celestial_bodies
from constellation import geos
from propagation import propagate_satellites_new
from pointing import pointing_place_update, update_satellite_pointing
from plotting_vectors import plot_pointing_vectors

def show_geo_search():
    """
    This demo initializes a simulation, adds a GEO constellation, and then
    generates several plots to visualize the satellite pointing updates and
    the RA/Dec history of one satellite.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_celestial_bodies(sim_data)

    geos(sim_data, 12, 0.4)

    propagate_satellites_new(sim_data, sim_start_time)
    
    update_satellite_pointing(sim_data)
    fig1 = plot_pointing_vectors(sim_data, 'Initial Pointing Vectors', sim_start_time)

    ra_history = []
    dec_history = []

    def record_ra_dec():
        p = sim_data['satellites']['pointing'][0]
        p_norm = p / np.linalg.norm(p)
        ra = np.arctan2(p_norm[1], p_norm[0])
        dec = np.arcsin(p_norm[2])
        ra_history.append(np.rad2deg(ra))
        dec_history.append(np.rad2deg(dec))

    record_ra_dec()

    for _ in range(5):
        pointing_place_update(sim_data)
        update_satellite_pointing(sim_data)
        record_ra_dec()
    
    fig2 = plot_pointing_vectors(sim_data, 'After 5 Updates', sim_start_time)

    for _ in range(10):
        pointing_place_update(sim_data)
        update_satellite_pointing(sim_data)
        record_ra_dec()

    fig3 = plot_pointing_vectors(sim_data, 'After 15 Updates', sim_start_time)

    for _ in range(85):
        pointing_place_update(sim_data)
        update_satellite_pointing(sim_data)
        record_ra_dec()

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=ra_history, mode='lines', name='RA (deg)'))
    fig4.add_trace(go.Scatter(y=dec_history, mode='lines', name='Dec (deg)'))
    fig4.update_layout(
        title='RA and Dec History of a GEO Satellite',
        xaxis_title='Update Number',
        yaxis_title='Angle (degrees)'
    )

    return (fig1, fig2, fig3, fig4)
