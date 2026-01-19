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
    Creates a single satellite fixed at the given x, y, z coordinates.
    This function initializes or replaces the satellite data in sim_data.
    """
    sim_data['counts']['satellites'] = 1
    
    pos = np.array([x, y, z])
    vel = np.array([0, 0, 0])
    breakpoint()
    sim_data['satellites'] = {
        'position': pos.reshape(1, 3),
        'velocity': vel.reshape(1, 3),
        'acceleration': np.zeros((1, 3), dtype=float),
        'orbital_elements': np.zeros((1, 6), dtype=float), # Dummy
        'epochs': [sim_data['time']],
    }

    # --- Detector ---
    detector = makeDetector(n=1, band='V', fov=90 * DEGREE, ifov=1 * ARCSEC, aper=1)
    
    # Pointing "up" is radially outward from Earth's center
    # Handle the case where the position is at the origin
    norm_pos = np.linalg.norm(pos)
    if norm_pos > 0:
        pointing_vector = pos / norm_pos
    else:
        pointing_vector = np.array([1, 0, 0]) # Default pointing vector if at origin

    detector.pointing[0, :] = pointing_vector
    
    sim_data['detector'] = detector

def fixedTarget(sim_data: Dict[str, Any], size: float, x: float, y: float, z: float):
    """
    Places a fixed target at the given x, y, z coordinates.
    Appends to existing fixedpoints if they exist.
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
    Fixes the sun's position on the negative x-axis at 1 AU.
    """
    if 'celestial' not in sim_data:
        add_celestial_bodies(sim_data)

    sun_pos_m = -1 * u.au.to(u.m)
    
    # Update the celestial position for the sun in meters
    sim_data['celestial']['position'][0] = np.array([sun_pos_m, 0, 0])

def demoFixed():
    """
    Demonstrates the use of the fixedSat and fixedTarget functions.
    Displays the positions of the sun, satellite, and target in a 3D plot
    with a logarithmic scale for distance.
    """
    # --- Setup ---
    start_time = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)

    # --- Run functions to create objects ---
    fixSun(sim_data)
    fixedSat(sim_data, x=100, y=0, z=0)
    # Place a target somewhere, using EARTH_RADIUS which is imported in the file
    fixedTarget(sim_data, size=1.0, x=100_000_000, y=0, z=0)

    # --- Extract data for plotting ---
    sun_pos = sim_data['celestial']['position'][0]
    sat_pos = sim_data['satellites']['position'][0]
    target_pos = sim_data['fixedpoints']['position'][0]
    viewing_vector = sim_data['detector'].pointing[0]

    # --- Log-scale positions ---
    def log_scale_pos(pos):
        r = np.linalg.norm(pos)
        if r == 0:
            return pos
        log_r = np.log(r)
        return pos * (log_r / r)

    sun_pos_log = log_scale_pos(sun_pos)
    sat_pos_log = log_scale_pos(sat_pos)
    target_pos_log = log_scale_pos(target_pos)

    # --- Plotting ---
    fig = go.Figure()

    # Earth
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color='blue'), name='Earth'))

    # Sun
    fig.add_trace(go.Scatter3d(x=[sun_pos_log[0]], y=[sun_pos_log[1]], z=[sun_pos_log[2]], mode='markers', marker=dict(size=8, color='yellow'), name='Sun'))

    # Satellite
    fig.add_trace(go.Scatter3d(x=[sat_pos_log[0]], y=[sat_pos_log[1]], z=[sat_pos_log[2]], mode='markers', marker=dict(size=5, color='red'), name='Satellite'))
    
    # Target
    fig.add_trace(go.Scatter3d(x=[target_pos_log[0]], y=[target_pos_log[1]], z=[target_pos_log[2]], mode='markers', marker=dict(size=3, color='green'), name='Target'))

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

    fig.update_layout(
        title_text='Demonstration of Fixed Objects (Log Scale Distance)',
        scene=dict(
            xaxis_title='X (log-scaled)',
            yaxis_title='Y (log-scaled)',
            zaxis_title='Z (log-scaled)',
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    
    print("--- Running scandetector ---")
    scan_output = scandetectors(sim_data)
    print(f"Output of scandetector: {scan_output}")

    return fig

if __name__ == '__main__':
    fig = demoFixed()
    fig.show()
