import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go

from minimalsimulation import create_empty_simulation
from propagation import add_satellites_from_tle
from demo_common import get_sample_geo_tles
from detector import makeDetector
from targets import add_fixed_points
from celestialbodies import add_celestial_bodies, celestial_update
from cadenceController import initCadence, nextIntegration
from pointing import detectorPointingInitialize, update_detector_pointing
from dataHandling import DataHandler
from constants import ARCSEC
from plotting_3d import tag_demo_figure

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

    add_satellites_from_tle(sim_data, get_sample_geo_tles(4), 'satellites')
    num_sats = sim_data.counts.satellites

    sim_data.satellites.detector = makeDetector(
        num_sats, band="V", fov=np.radians(45),
        ifov=3 * ARCSEC, aper=np.sqrt(4.0 / np.pi), qe=0.8,
    )
    sim_data.satellites.detector.integrationTime[:] = 10.0

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
    return tag_demo_figure(fig, "demo_gap_time_histogram")

if __name__ == '__main__':
    fig = demo_gap_time_histogram()
    fig.show()
