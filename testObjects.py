import numpy as np
from typing import Dict, Any
import math

import astropy.units as u

from constants import EARTH_RADIUS, ARCSEC, DEGREE
from detector import makeBlankDetector, makeDetector
from celestialbodies import add_celestial_bodies
import plotly.graph_objects as go
from simulation import create_empty_simulation
from datetime import datetime, timezone
from scandetectors import scandetectors


def fixedSat(sim_data: Dict[str, Any], x: float, y: float, z: float):
    """
    Creates a single satellite fixed at the given x, y, z coordinates in meters.

    This function is responsible for adding satellite data to the simulation.
    If no satellite data exists in `sim_data`, it initializes the necessary
    data structures for a single satellite, including its position, velocity,
    acceleration, orbital elements, and associated detector properties.
    If satellite data already exists, it appends the new satellite's
    information to the existing arrays, ensuring that detector parameters
    are consistently added for the new satellite based on the last existing
    detector's properties. It also calculates an initial pointing vector
    for the detector based on the satellite's position (pointing away from Earth).

    Args:
        sim_data (Dict[str, Any]): The main simulation data dictionary.
        x (float): The x-coordinate of the satellite's position in meters.
        y (float): The y-coordinate of the satellite's position in meters.
        z (float): The z-coordinate of the satellite's position in meters.
    """
    new_pos = np.array([x, y, z]).reshape(1, 3)
    new_vel = np.array([0, 0, 0]).reshape(1, 3)
    new_accel = np.zeros((1, 3), dtype=float)
    new_orbital_elements = np.zeros((1, 6), dtype=float)

    # Calculate pointing vector for the new satellite
    norm_pos = np.linalg.norm(new_pos)
    if norm_pos > 0:
        new_pointing_vector = new_pos / norm_pos
    else:
        new_pointing_vector = np.array([1, 0, 0]).reshape(1, 3) # Default if at origin

    if 'satellites' not in sim_data or not sim_data.get('satellites'):
        # Initialize for the first satellite
        sim_data['counts']['satellites'] = 1
        sim_data['satellites'] = {
            'position': new_pos,
            'velocity': new_vel,
            'acceleration': new_accel,
            'orbital_elements': new_orbital_elements,
            'epochs': [sim_data['time']],
        }
        # Initialize detector for the first satellite
        detector = makeDetector(n=1, band='V', fov=90 * DEGREE, ifov=1 * ARCSEC, aper=1)
        detector.pointing[0, :] = new_pointing_vector[0]
        sim_data['detector'] = detector
    else:
        # Append for subsequent satellites
        sim_data['counts']['satellites'] += 1

        sim_data['satellites']['position'] = np.vstack([sim_data['satellites']['position'], new_pos])
        sim_data['satellites']['velocity'] = np.vstack([sim_data['satellites']['velocity'], new_vel])
        sim_data['satellites']['acceleration'] = np.vstack([sim_data['satellites']['acceleration'], new_accel])
        sim_data['satellites']['orbital_elements'] = np.vstack([sim_data['satellites']['orbital_elements'], new_orbital_elements])
        sim_data['satellites']['epochs'].append(sim_data['time']) # epochs is a list, not numpy array

        # Append detector attributes for the new satellite
        current_detector = sim_data['detector']

        # Append to 1D arrays - copy last element's value (assuming uniform detector properties from makeDetector)
        current_detector.apertureArea = np.append(current_detector.apertureArea, current_detector.apertureArea[-1])
        current_detector.pixelArea = np.append(current_detector.pixelArea, current_detector.pixelArea[-1])
        current_detector.qe = np.append(current_detector.qe, current_detector.qe[-1])
        current_detector.photoEff = np.append(current_detector.photoEff, current_detector.photoEff[-1])
        current_detector.pixCount = np.append(current_detector.pixCount, current_detector.pixCount[-1])
        current_detector.solarEx = np.append(current_detector.solarEx, current_detector.solarEx[-1])
        current_detector.lunarex = np.append(current_detector.lunarex, current_detector.lunarex[-1])
        current_detector.earthEx = np.append(current_detector.earthEx, current_detector.earthEx[-1])
        current_detector.skyBack = np.append(current_detector.skyBack, current_detector.skyBack[-1])
        current_detector.zpCal = np.append(current_detector.zpCal, current_detector.zpCal[-1])
        current_detector.integrationTime = np.append(current_detector.integrationTime, current_detector.integrationTime[-1])
        current_detector.fov = np.append(current_detector.fov, current_detector.fov[-1])
        current_detector.ifov = np.append(current_detector.ifov, current_detector.ifov[-1])
        
        # Append to list attribute
        current_detector.filt.append(current_detector.filt[-1])

        # Append to 2D array: pointing
        current_detector.pointing = np.vstack([current_detector.pointing, new_pointing_vector])
        
        # For pointing_state, append a new zero column for the new satellite
        current_detector.pointing_state = np.hstack([current_detector.pointing_state, np.zeros((2,1), dtype=int)])

def fixedTarget(sim_data: Dict[str, Any], size: float, x: float, y: float, z: float):
    """
    Places a fixed target at the given x, y, z coordinates in meters.

    This function adds a stationary target to the simulation environment.
    If no fixed target data exists in `sim_data`, it initializes the required
    data structures, including arrays for position, exclusion flags, size,
    and albedo. If fixed target data already exists, it appends the new
    target's information to the existing arrays. A default albedo of 0.2
    is assigned to new targets.

    Args:
        sim_data (Dict[str, Any]): The main simulation data dictionary.
        size (float): The physical size (e.g., diameter or characteristic length) of the target.
        x (float): The x-coordinate of the target's position in meters.
        y (float): The y-coordinate of the target's position in meters.
        z (float): The z-coordinate of the target's position in meters.
    """
    if 'fixedpoints' not in sim_data or not sim_data.get('fixedpoints'):
        sim_data['counts']['fixedpoints'] = 0
        sim_data['fixedpoints'] = {
            'position': np.empty((0, 3), dtype=float),
            'exclusion': np.empty(0, dtype=int),
            'size': np.empty(0, dtype=float),
            'albedo': np.empty(0, dtype=float),
        }

    pos = np.array([x, y, z]).reshape(1, 3)

    # --- Add target to sim_data ---
    sim_data['counts']['fixedpoints'] += 1
    sim_data['fixedpoints']['position'] = np.vstack([sim_data['fixedpoints']['position'], pos])
    sim_data['fixedpoints']['exclusion'] = np.append(sim_data['fixedpoints']['exclusion'], 0)
    sim_data['fixedpoints']['size'] = np.append(sim_data['fixedpoints']['size'], size)
    sim_data['fixedpoints']['albedo'] = np.append(sim_data['fixedpoints']['albedo'], 0.2) # Default albedo

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
    if 'celestial' not in sim_data:
        add_celestial_bodies(sim_data)

    sun_pos_m = -1 * u.au.to(u.m)
    
    # Update the celestial position for the sun in meters
    sim_data['celestial']['position'][0] = np.array([sun_pos_m, 0, 0])

def demoFixed():
    """
    Demonstrates the use of the fixedSat, fixedTarget, and fixSun functions
    to set up a basic simulation scenario with a fixed satellite and targets.

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
    fixedSat(sim_data, x=100, y=0, z=0)
    # Place a target somewhere, using EARTH_RADIUS which is imported in the file
    original_target_x = 100_000_000
    fixedTarget(sim_data, size=1.0, x=original_target_x, y=0, z=0)
    fixedTarget(sim_data, size=1.0, x=original_target_x * 10, y=0, z=0) # New target at 10 times the range

    # --- Log-scale positions helper function ---
    def log_scale_pos(pos):
        r = np.linalg.norm(pos)
        if r == 0:
            return pos
        log_r = np.log(r)
        return pos * (log_r / r)

    # --- Extract data for plotting ---
    sun_pos = sim_data['celestial']['position'][0]
    sat_pos = sim_data['satellites']['position'][0]
    
    # Get all target positions and log-scale them
    all_target_positions = sim_data['fixedpoints']['position']
    all_target_positions_log = np.array([log_scale_pos(pos) for pos in all_target_positions])

    viewing_vector = sim_data['detector'].pointing[0] # Assuming first satellite's detector

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
    vec_end = sat_pos_log + viewing_vector * vec_len_log
    
    fig.add_trace(go.Scatter3d(
        x=[sat_pos_log[0], vec_end[0]],
        y=[sat_pos_log[1], vec_end[1]],
        z=[sat_pos_log[2], vec_end[2]],
        mode='lines',
        line=dict(color='red', width=2),
        name='Viewing Vector'
    ))

    fig.update_layout(          # 
        title_text='Demonstration of Fixed Objects (Log Scale Distance)',
        scene=dict(
            xaxis_title='X (log-scaled)',
            yaxis_title='Y (log-scaled)',
            zaxis_title='Z (log-scaled)',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    # print("--- Running scandetector ---")
    # scan_output = scandetectors(sim_data)
    # print(f"Output of scandetector: {scan_output}") 

    return fig, sim_data

if __name__ == '__main__':
    fig, simdata = demoFixed()
    fig.show()
