import numpy as np
from typing import Dict, Any

from constants import POINTING_COUNT_IDX, POINTING_PLACE_IDX
from fibonacciSearch import pointing_vectors, resort_vectors_by_proximity
from exclusion import exclusion

def generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int, debug: bool = False) -> None:
    """
    Generates a pointing sphere with n_points and stores it in the data_struct.
    These positions will be used by the update_satellte_pointing to
    point sensors incrementaly.
    The index is ['pointing_spheres'][n] 
    If a sphere with the same number of points already exists, this function does nothing.
    """
    if n_points not in data_struct['pointing_spheres']:
        print(f"Generating pointing sphere with {n_points} points...")
        data_struct['pointing_spheres'][n_points] = \
           resort_vectors_by_proximity(pointing_vectors(n_points))

    if debug:
        print(f"\n--- Debugging Pointing Sphere (n_points={n_points}) ---")
        generated_vectors = data_struct['pointing_spheres'][n_points]
        print(f"Total number of pointing vectors generated: {len(generated_vectors)}")

        num_to_show = min(5, len(generated_vectors))
        print(f"Showing first {num_to_show} pointing vectors and their norms:")
        print(f"{'Index':<7} {'Vector (x, y, z)':<35} {'Norm':<10}")
        print(f"{'-------':<7} {'-----------------------------------':<35} {'----------':<10}")
        for i in range(num_to_show):
            vec = generated_vectors[i]
            norm = np.linalg.norm(vec)
            print(f"{i:<7} {str(vec):<35} {norm:<10.6f}")
        print("-------------------------------------------\n")


def update_satellite_pointing(data_struct: Dict[str, Any]) -> None:
    """
    Updates the pointing vector for each satellite, skipping excluded pointing directions.
    """
    num_sats = data_struct['counts']['satellites']
    if num_sats == 0:
        return

    pointing_state = data_struct['satellites']['pointing_state']
    pointing_vectors_all = data_struct['satellites']['pointing']

    for i in range(num_sats):
        count = int(pointing_state[i, POINTING_COUNT_IDX])
        if count <= 0:
            continue

        if count not in data_struct['pointing_spheres']:
            raise ValueError(f"Pointing sphere for {count} points not generated.")

        grid = data_struct['pointing_spheres'][count]
        
        place = int(pointing_state[i, POINTING_PLACE_IDX])
        start_place = place

        while True:
            place += 1
            if place >= count:
                place = 0

            pointing_vectors_all[i] = grid[place]
            
            if exclusion(data_struct, i) == 0:
                pointing_state[i, POINTING_PLACE_IDX] = place
                break
            
            if place == start_place:
                print(f"Warning: Satellite {i} has all pointing vectors excluded.")
                pointing_state[i, POINTING_PLACE_IDX] = place
                break


def jerk(data_struct: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]:
    """
    Moves the pointing vector of specific satellites by 0.3 radians in a
    random direction.

    This function applies a random rotation to the satellites' pointing vectors.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_indices: The indices of the satellites to modify.

    Returns:
        The modified data_struct with the updated pointing vectors.
    """
    if satellite_indices.size == 0:
        return data_struct

    p = data_struct['satellites']['pointing'][satellite_indices]
    p_norm = p / np.linalg.norm(p, axis=1)[:, np.newaxis]

    # Generate a random vector not parallel to p_norm
    r = np.random.randn(*p.shape)
    r -= np.sum(r * p_norm, axis=1)[:, np.newaxis] * p_norm
    k_hat = r / np.linalg.norm(r, axis=1)[:, np.newaxis]

    theta = 0.3
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rodrigues' rotation formula
    p_new = p_norm * cos_theta + np.cross(k_hat, p_norm) * sin_theta

    data_struct['satellites']['pointing'][satellite_indices] = p_new

    return data_struct





import math
from datetime import datetime, timezone
import plotly.graph_objects as go
import os

from simulation import create_empty_simulation, add_celestial_bodies
from propagation import add_satellites_from_tle, propagate_satellites_new, celestial_update
from plotting_3d import plot_3d_scatter
from constants import DETECTOR_FOV_IDX, SOLAR_EXCLUSION_ANGLE_IDX, SAT_DETECTOR_IDX

def demo_exclusion_pointing():
    """
    Demonstrates satellite pointing with a solar exclusion angle and a detector
    field of view, plotting the pointing history on a sphere.
    """
    start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)
    add_celestial_bodies(sim_data)

    # Create a temporary TLE file with a single satellite for this demo
    single_tle_content = """ISS (ZARYA)
1 25544U 98067A   25209.52203988  .00012111  00000+0  22159-3 0  9991
2 25544  51.6412 254.9961 0006733  98.4322 261.6813 15.49493393462383"""
    
    temp_tle_path = "temp_single_sat.tle"
    with open(temp_tle_path, "w") as f:
        f.write(single_tle_content)

    add_satellites_from_tle(sim_data, temp_tle_path, 'satellites')
    
    # Clean up the temporary TLE file
    os.remove(temp_tle_path)

    # Set solar exclusion angle and detector FOV for the single satellite
    sim_data['satellites']['detector'][0, SOLAR_EXCLUSION_ANGLE_IDX] = math.pi
    sim_data['satellites']['detector'][0, DETECTOR_FOV_IDX] = math.pi / 2

    # Generate 100 pointing points using the module's generate_pointing_sphere
    n_points_sphere = 100
    generate_pointing_sphere(sim_data, n_points_sphere)

    # Initialize pointing_state for the single satellite
    # Assuming the first satellite (index 0)
    sim_data['satellites']['pointing_state'][0, POINTING_COUNT_IDX] = n_points_sphere
    sim_data['satellites']['pointing_state'][0, POINTING_PLACE_IDX] = 0 # Start at the first point

    pointed_directions_history = []
    
    # Propagate the satellite once to get an initial position
    sim_data = celestial_update(sim_data, start_time)
    sim_data = propagate_satellites_new(sim_data, start_time, 'satellites')
    
    initial_sat_pos = sim_data['satellites']['position'][0]
    
    for i in range(200):
        update_satellite_pointing(sim_data)
        current_pointed_direction = sim_data['satellites']['pointing'][0]
        pointed_directions_history.append(current_pointed_direction)

    pointed_directions_history = np.array(pointed_directions_history)

    fig = plot_3d_scatter(
        positions=np.array([initial_sat_pos]), # Plot initial satellite position
        title="Satellite Pointing with Exclusion",
        plot_time=start_time,
        labels=["Satellite"],
        marker_size=5,
        trace_name="Satellite"
    )

    # Add the pointing history as markers
    colors = np.arange(len(pointed_directions_history))
    fig.add_trace(go.Scatter3d(
        x=pointed_directions_history[:, 0],
        y=pointed_directions_history[:, 1],
        z=pointed_directions_history[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors,
            colorscale='Plasma', # Using a gradient color
            opacity=0.7
        ),
        name='Pointing History'
    ))

    # Connect the pointing points with lines
    fig.add_trace(go.Scatter3d(
        x=pointed_directions_history[:, 0],
        y=pointed_directions_history[:, 1],
        z=pointed_directions_history[:, 2],
        mode='lines',
        line=dict(color='grey', width=1),
        name='Pointing Path'
    ))
    
    # Add a sphere at the origin to represent the pointing sphere itself for context
    # Scaling it to be visible but transparent
    sphere_radius = np.max(np.linalg.norm(pointed_directions_history, axis=1))
    if sphere_radius == 0: sphere_radius = 1.0 # Avoid division by zero if no pointing history
    
    u_sphere = np.linspace(0, 2 * np.pi, 50)
    v_sphere = np.linspace(0, np.pi, 50)
    x_sphere = sphere_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = sphere_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale='Viridis', showscale=False, opacity=0.1, name='Pointing Sphere'
    ))

    fig.show() 
    return fig

