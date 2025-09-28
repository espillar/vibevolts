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
    This demo initializes a simulation structure with the sun and the moon and the earth and time, 
    then adds a constellation of 12 GEOs with 0.4 radian fields of view. 
    After propogating to the starting position it does a plot of the satellites with the pointing vectors.
    It then updates all of the pointing vectors 5 times using pointing_place_update and redisplays the constellation.
    Update 10 times and redisplay again.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_celestial_bodies(sim_data)

    geos(sim_data, 12, 0.4)

    propagate_satellites_new(sim_data, sim_start_time)
    
    update_satellite_pointing(sim_data)
    fig1 = plot_pointing_vectors(sim_data, 'Initial Pointing Vectors', sim_start_time)
    fig1.show()

    for _ in range(5):
        pointing_place_update(sim_data)
    
    update_satellite_pointing(sim_data)
    fig2 = plot_pointing_vectors(sim_data, 'After 5 Updates', sim_start_time)
    fig2.show()

    for _ in range(10):
        pointing_place_update(sim_data)

    update_satellite_pointing(sim_data)
    fig3 = plot_pointing_vectors(sim_data, 'After 15 Updates', sim_start_time)
    fig3.show()

if __name__ == '__main__':
    show_geo_search()