import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go

from minimalsimulation import create_empty_simulation
from propagation import add_satellites_from_tle
from targets import add_fixed_points
from celestialbodies import add_celestial_bodies, celestial_update
from cadenceController import initCadence, nextIntegration
from pointing import detectorPointingInitialize, update_detector_pointing
from dataHandling import DataHandler
from constants import *

def demo_gap_time_histogram() -> go.Figure:
    """
    Demonstrates the calculation of target interobservation gap times
    and plots them on a Plotly histogram.

    It runs a short cadence simulation with 4 LEO/GEO satellites and 20 targets
    over 50 iterations, accumulates detection data, and generates a
    gorgeous pooled histogram of the resulting gap times.

    Returns:
        plotly.graph_objects.Figure: The gap times histogram figure.
    """
    print("\n--- Starting Demo: Gap Time Histogram ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    
    # Create a dummy TLE file for 4 satellites
    tle_data = """SAT-1
1 90401U 25007A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90401   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
SAT-2
1 90402U 25007B   25210.50000000  .00000000  00000-0  00000-0 0  9992
2 90402   0.0500  45.0000 0001000  90.0000  60.0000  1.00270000    12
SAT-3
1 90403U 25007C   25210.50000000  .00000000  00000-0  00000-0 0  9993
2 90403   0.0500  45.0000 0001000  90.0000 100.0000  1.00270000    13
SAT-4
1 90404U 25007D   25210.50000000  .00000000  00000-0  00000-0 0  9994
2 90404   0.0500  45.0000 0001000  90.0000 140.0000  1.00270000    14
"""
    dummy_tle_path = "dummy_tle_gap.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    add_satellites_from_tle(sim_data, dummy_tle_path, 'satellites')
    num_sats = sim_data['counts']['satellites']

    # Set detector properties
    sim_data['detector'].integrationTime[:] = 10.0
    sim_data['detector'].filt = ["V"] * num_sats
    sim_data['detector'].apertureArea[:] = 1.0
    sim_data['detector'].qe[:] = 0.8
    sim_data['detector'].fov[:] = np.radians(45)  # Large FOV to ensure detections
    sim_data['detector'].pixelOmega[:] = (3 * ARCSEC)**2

    add_fixed_points(sim_data, num_points=20)
    add_celestial_bodies(sim_data)
    celestial_update(sim_data)

    detectorPointingInitialize(sim_data, 100)
    initCadence(sim_data)

    handler = DataHandler()

    print("Running 50 cadence steps...")
    for _ in range(50):
        update_detector_pointing(sim_data)
        results = nextIntegration(sim_data, print_output=0)
        handler.add_results(results)

    # Generate the histogram plot
    fig = handler.plot_gap_times_histogram(show_plot=False)
    return fig

if __name__ == '__main__':
    fig = demo_gap_time_histogram()
    fig.show()
