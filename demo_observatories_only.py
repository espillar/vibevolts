import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go

from minimalsimulation import create_empty_simulation
from observatories import add_observatories, propagate_observatories
from plotting_3d import plot_3d_scatter, add_vector_traces, tag_demo_figure
from detector import makeDetector
from constants import DEGREE, ARCSEC, EARTH_RADIUS
from celestialbodies import add_celestial_bodies, celestial_update
from pointing import detectorPointingInitialize, update_detector_pointing
from propagation import add_satellites_from_tle, propagate_satellites
from demo_common import STANDARD_TLE_DATA
from sim_report import print_simulation_state

def demo_observatories_only() -> tuple:
    """
    Runs a demonstration placing 5 observatories on Earth with no satellites.
    Propagates through 5 time steps, showing pointing vectors.

    Returns:
        A tuple of Plotly figure objects.
    """
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_celestial_bodies(sim_data)

    # 5 Observatory locations: Mauna Kea, Paranal, La Palma, Siding Spring, Kitt Peak
    latitudes = np.array([19.8206, -24.6272, 28.7636, -31.2755, 31.9583])
    longitudes = np.array([-155.4681, -70.4042, -17.8947, 149.0673, -111.5967])
    altitudes = np.array([4205.0, 2635.0, 2396.0, 1165.0, 2096.0])
    labels = ["Mauna Kea", "Paranal", "La Palma", "Siding Spring", "Kitt Peak"]

    print("\n--- Generating 3D plots for Demo Observatories Only ---")
    
    add_observatories(
        sim_data=sim_data,
        num_observatories=5,
        latitudes=latitudes,
        longitudes=longitudes,
        altitudes=altitudes
    )

    # Add satellites
    add_satellites_from_tle(sim_data, STANDARD_TLE_DATA, 'satellites')

    # Associate each observatory and satellite with a proper sensor/detector
    num_obs = sim_data.counts.observatories
    sim_data.observatories.detector = makeDetector(
        n=num_obs, band="V", fov=10 * DEGREE, ifov=3 * ARCSEC, aper=1.0
    )

    num_sats = sim_data.counts.satellites
    sim_data.satellites.detector = makeDetector(
        n=num_sats, band="V", fov=10 * DEGREE, ifov=3 * ARCSEC, aper=1.0
    )

    detectorPointingInitialize(sim_data, grid_points=400)

    figs = []
    current_time = sim_start_time

    for step in range(5):
        # Propagate observatories and satellites to current_time (Earth rotation)
        propagate_observatories(sim_data, current_time)
        propagate_satellites(sim_data, current_time)
        celestial_update(sim_data, current_time)
        
        # Update pointing using the pointing algorithm for both
        update_detector_pointing(sim_data, sat_category='observatories')
        update_detector_pointing(sim_data, sat_category='satellites')

        # Plot satellites and observatories in unified detector order
        all_positions = sim_data.get_detector_positions()
        all_labels = [f"Sat {i}" for i in range(num_sats)] + labels

        fig = plot_3d_scatter(
            positions=all_positions,
            title=f"Step {step+1}: Observatories and Satellites at {current_time.isoformat()}",
            plot_time=current_time,
            labels=all_labels,
            marker_size=4,
            trace_name='Assets',
            marker_color='blue'
        )
        
        # Plot red pointing vectors originating at each asset
        add_vector_traces(
            fig, all_positions, sim_data.detector.pointing,
            scale=0.5 * EARTH_RADIUS, color='red', width=4,
        )

        tag_demo_figure(fig, "demo_observatories_only")
        figs.append(fig)

        if step == 0:
            print_simulation_state(sim_data)
        
        # Propagate forward in time by 2 hours
        current_time += timedelta(hours=2)
        
    return tuple(figs)
