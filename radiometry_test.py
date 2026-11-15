
import numpy as np
from typing import Dict, Any

import astropy.units as u

from constants import ARCSEC, DEGREE
from detector import makeDetector
from celestialbodies import add_celestial_bodies
import plotly.graph_objects as go
from minimalsimulation import (
    create_empty_simulation,
    SatellitesState,
    FixedPointsState,
)
from datetime import datetime, timezone
from scandetectors import scandetectors
from plotting_3d import add_vector_traces, tag_demo_figure


def fixedSat(sim_data: Dict[str, Any], x: float, y: float, z: float, fov= 10 * DEGREE):
    """
    Creates a single satellite fixed at the given x, y, z coordinates in meters.

    This function is responsible for adding a fixed satellite  data to the simulation,
    mostly for testing purposes: no dynamic here.
    If no satellite data exists in `sim_data`, it initializes the necessary
    data structures for a single satellite, including its position, velocity,
    acceleration, orbital elements, and associated detector properties.
    The parameters used to create this initial detector are stored in
    `sim_data.initial_detector_params` for future use.

    If satellite data already exists, it appends the new satellite's
    information to the existing arrays. To maintain consistent detector
    properties, a new single detector is created using `makeDetector` with
    the parameters stored from the first detector. The attributes of this
    newly created detector are then appended to the existing detector object
    (`sim_data.detector`). It also calculates an initial pointing vector
    for the detector based on the satellite's position (pointing away from Earth).



    
    Args:
        sim_data (Dict[str, Any]): The main simulation data dictionary.
        x (float): The x-coordinate of the satellite's position in meters.
        y (float): The y-coordinate of the satellite's position in meters.
        z (float): The z-coordinate of the satellite's position in meters.
    """
    new_pos = np.array([[x, y, z]], dtype=float)
    new_vel = np.zeros((1, 3), dtype=float)
    new_orbital_elements = np.zeros((1, 6), dtype=float)

    norm_pos = np.linalg.norm(new_pos)
    new_pointing_vector = (
        new_pos / norm_pos if norm_pos > 0 else np.array([[1.0, 0.0, 0.0]])
    )

    if not sim_data.satellites or 'initial_detector_params' not in sim_data:
        sim_data.initial_detector_params = {
            'n': 1, 'band': 'V', 'fov': fov, 'ifov': 1 * ARCSEC,
            'intTime': 1.0, 'aper': 1, 'qe': 0.5, 'photfrac': 0.7,
            'solarex': 20.0 * DEGREE, 'lunarex': 10.0 * DEGREE, 'earthex': 15.0 * DEGREE,
        }

    det_params = sim_data.initial_detector_params.copy()
    det_params['n'] = 1
    detector = makeDetector(**det_params)
    detector.pointing[0, :] = new_pointing_vector[0]

    new_sat = SatellitesState(
        position=new_pos,
        velocity=new_vel,
        orbital_elements=new_orbital_elements,
        epochs=[sim_data.time],
        detector=detector,
    )
    if not sim_data.satellites:
        sim_data.satellites = new_sat
    else:
        sim_data.satellites.append(new_sat)

def fixedTarget(sim_data: Dict[str, Any], size: float, x: float, y: float, z: float):
    """
    Places a fixed target at the given x, y, z coordinates in meters.

    This function adds a stationary lambertian sphere target to the simulation environment.
    If no fixed target data exists in `sim_data`, it initializes the required
    data structures, including arrays for position, exclusion flags, size,
    and albedo. If fixed target data already exists, it appends the new
    target's information to the existing arrays. A default albedo of 0.2
    is assigned to new targets.

    Args:
        sim_data (Dict[str, Any]): The main simulation data dictionary.
        size (float): The diameter of the target in meters, assumed to be a lambertian sphere.
        x (float): The x-coordinate of the target's position in meters.
        y (float): The y-coordinate of the target's position in meters.
        z (float): The z-coordinate of the target's position in meters.
    """
    if not sim_data.fixedpoints:
        sim_data.fixedpoints = FixedPointsState()

    sim_data.fixedpoints.add_target(position=np.array([x, y, z]), size=size, albedo=0.2)

def fixSun(sim_data: Dict[str, Any]) -> None:
    """
    Fixes the sun's position in the simulation.

    This function ensures that celestial body data is initialized in `sim_data`
    if it doesn't already exist. It then sets the Sun's position to a fixed
    point on the negative x-axis at a distance of 1 Astronomical Unit (AU)
    from the origin (Earth-centered). This provides a static illumination
    source for simulations.

    Args:
        sim_data (Dict[str, Any]): The main simulation data dictionary.
    """
    if not sim_data.celestial:
        add_celestial_bodies(sim_data)

    sun_pos_m = -1 * u.au.to(u.m)
    
    # Update the celestial position for the sun in meters
    sim_data.celestial.position[0] = np.array([sun_pos_m, 0, 0])

def demoFixed():
    """
    Demonstrates the use of the fixedSat, fixedTarget, and fixSun functions
    to set up a basic simulation scenario with a fixed satellite and targets.
    Prints out radiometry results.

    The general flow of this demonstration function is as follows:
    1.  **Simulation Setup**: Initializes an empty simulation data structure
        with a specified start time using `create_empty_simulation`.
    2.  **Object Creation**:
        -   The Sun's position is fixed using `fixSun`.
        -   A single satellite is placed at a static `(x, y, z)` coordinate
            using `fixedSat`.
        -   Multiple fixed targets are placed at specific coordinates and
            assigned a size using `fixedTarget`.
    3.  **Log-Scale Transformation**: A helper function `log_scale_pos` is defined
        and applied to all celestial body, satellite, and target positions.
        This transforms the actual physical distances into a logarithmic scale
        for better visualization in a 3D plot, allowing widely separated objects
        to be viewed within a single plot.
    4.  **3D Plotting**: A Plotly 3D scatter plot is generated to visualize
        the Earth (origin), the log-scaled Sun, the log-scaled satellite,
        and the log-scaled fixed targets. A viewing vector originating from
        the satellite, representing its detector's pointing direction, is also
        added to the plot.
    5.  **Scandetectors (Commented)**: The function includes a commented-out
        call to `scandetectors`, which would process the created simulation
        data to determine target visibility and SNR. This part is
        intentionally commented out in the demonstration but shows how
        `scandetectors` would integrate into the workflow.

    Returns:
        tuple: A tuple containing:
            -   fig (plotly.graph_objects.Figure): The Plotly figure object
                displaying the 3D visualization.
            -   sim_data (Dict[str, Any]): The simulation data dictionary
                after all objects have been added.
    """
    # --- Setup ---
    start_time = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)

    # --- Run functions to create objects ---
    fixSun(sim_data)
    # Place a satellite near the center of the earth
    fixedSat(sim_data, x=100, y=0, z=0, fov=180 * DEGREE)
    
    # Place a target 1e8 m away and another 1e9m away
    targRng = 100_000_000
    print("targets are 1e8 m diameter 1 along line of sight")
    print("targets are 1e8 m diameter 2")
    print("targets are 1e9 m diameter 1")
    print("targets are 1e8 m diameter 1 perp to line of site")
    fixedTarget(sim_data, size=1.0, x=targRng, y=0, z=0)
    fixedTarget(sim_data, size=2.0, x=targRng, y=0, z=0)
    fixedTarget(sim_data, size=1.0, x=targRng * 10, y=0, z=0) # New target at 10 times the range
    fixedTarget(sim_data, size=1.0, x=0, y=targRng, z=0) # Right angles
    
    # --- Log-scale positions helper function ---
    def log_scale_pos(pos):
        r = np.linalg.norm(pos)
        if r == 0:
            return pos
        log_r = np.log(r)
        return pos * (log_r / r)

    # --- Extract data for plotting ---
    sun_pos = sim_data.celestial.position[0]
    sat_pos = sim_data.satellites.position[0]
    
    # Get all target positions and log-scale them
    all_target_positions = sim_data.fixedpoints.position
    all_target_positions_log = np.array([log_scale_pos(pos) for pos in all_target_positions])

    viewing_vector = sim_data.satellites.detector.pointing[0] # Assuming first satellite's detector

    sun_pos_log = log_scale_pos(sun_pos)
    sat_pos_log = log_scale_pos(sat_pos)
    # target_pos_log is now handled by all_target_positions_log

    # --- Plotting ---
    fig = go.Figure()

    # Earth
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color='blue'), name='Earth'))

    # Sun
    fig.add_trace(go.Scatter3d(x=[sun_pos_log[0]], y=[sun_pos_log[1]], z=[sun_pos_log[2]], mode='markers', marker=dict(size=8, color='yellow'), name='Sun'))

    # Satellite
    fig.add_trace(go.Scatter3d(x=[sat_pos_log[0]], y=[sat_pos_log[1]], z=[sat_pos_log[2]], mode='markers', marker=dict(size=5, color='red'), name='Satellite'))
    
    # Targets (both original and new)
    fig.add_trace(go.Scatter3d(
        x=all_target_positions_log[:, 0],
        y=all_target_positions_log[:, 1],
        z=all_target_positions_log[:, 2],
        mode='markers',
        marker=dict(size=3, color='green'),
        name='Targets'
    ))

    # Viewing vector
    vec_len_log = np.linalg.norm(sat_pos_log) * 0.3
    add_vector_traces(
        fig, sat_pos_log, viewing_vector, scale=vec_len_log,
        color='red', width=2, name='Viewing Vector', normalize=False,
    )

    fig.update_layout(          # 
        title_text='Demonstration of Fixed Objects (Log Scale Distance)',
        scene=dict(
            xaxis_title='X (log-scaled)',
            yaxis_title='Y (log-scaled)',
            zaxis_title='Z (log-scaled)',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    print("--- Running scandetector ---")
    scan_output = scandetectors(sim_data, print_output=1)
    print(f"Output of scandetector: {scan_output}") 

    return tag_demo_figure(fig, "demoFixed"), sim_data

if __name__ == '__main__':
    fig, simdata = demoFixed()
    fig.show()
