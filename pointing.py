import numpy as np
from typing import Dict, Any
import math
from datetime import datetime, timezone
import plotly.graph_objects as go
import os

from simulation import create_empty_simulation
from celestialbodies import add_celestial_bodies, celestial_update
from propagation import add_satellites_from_tle, propagate_satellites
from plotting_3d import plot_3d_scatter
from constants import POINTING_COUNT_IDX, POINTING_PLACE_IDX
from fibonacciSearch import pointing_vectors, resort_vectors_by_proximity
from exclusion import exclusion


def generate_pointing_sphere(sim_data: Dict[str, Any], n_points: int, debug: bool = False) -> None:
    """
    Generates a pointing sphere with n_points and stores it in the sim_data['pointing_sphers'][n_points]
    A pointing sphere is a 3 by n_points numpy array with the 3 representing unit vectors to be
    pointed to
    These positions will be used by the update_satellte_pointing to
    point sensors incrementaly.
    The index is ['pointing_spheres'][n] 
    If a sphere with the same number of points already exists, this function does nothing.
    The current version of the code resorts the vector by proximity to make the sky search
    more efficient, although this is not deeply optimized yet.
    """
    if n_points not in sim_data['pointing_spheres']:
        print(f"Generating pointing sphere with {n_points} points...")
        sim_data['pointing_spheres'][n_points] = \
           resort_vectors_by_proximity(pointing_vectors(n_points))

    if debug:
        print(f"\n--- Debugging Pointing Sphere (n_points={n_points}) ---")
        generated_vectors = sim_data['pointing_spheres'][n_points]
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


def update_detector_pointing(sim_data: Dict[str, Any], debug: bool = False) -> None:
    """
    Updates the pointing vector for each detector, skipping excluded pointing directions.
    """
    num_detectors = len(sim_data['detector'].filt)  
    if num_detectors == 0:
        return

# Bring in the approprieate pieces of the data structure for easier reference
    pointing_state = sim_data['detector'].pointing_state 
#    print('pointing_state in pointing.py ', pointing_state)
    pointing_vectors_all = sim_data['detector'].pointing

# Iterate over satellites
    for i in range(num_detectors):
# Place a grid of vectors to use in grid
        count = int(pointing_state[POINTING_COUNT_IDX, i])
        if count == 0:
            continue
        grid = sim_data['pointing_spheres'][count]
        
        place = int(pointing_state[POINTING_PLACE_IDX, i])
        start_place = place

        
        while True:
# MOve to the next place, wrap around if at end
            place += 1
            if place >= count:
                place = 0
            pointing_vectors_all[i] = grid[place]
            
            excluded = exclusion(sim_data, i)
            if debug:
                print(f"Detector {i}: Pointing location {place}, Excluded: {excluded != 0}")
                print(grid[place])

            if excluded == 0:
                pointing_state[POINTING_PLACE_IDX, i] = place
                break
            
            if place == start_place:
                print(f"Warning: Satellite {i} has all pointing vectors excluded.")
                pointing_state[POINTING_PLACE_IDX, i] = place
                break


def demo_exclusion_pointing():
    """
    Demonstrates satellite pointing with a solar exclusion angle and a detector
    field of view, plotting the pointing history on a sphere.
    """
    start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)
    add_celestial_bodies(sim_data)

    # Create a temporary TLE file with a single satellite for this demo
    single_tle_content = """GEO-SAT
1 99999U 99999A   25001.00000000  .00000000  00000-0  00000-0 0  9999
2 99999   0.0001 254.9961 0000001  98.4322 261.6813  1.00271900    10"""
    
    temp_tle_path = "temp_single_sat.tle"
    with open(temp_tle_path, "w") as f:
        f.write(single_tle_content)

    add_satellites_from_tle(sim_data, temp_tle_path, 'satellites')
    
    # Clean up the temporary TLE file
    os.remove(temp_tle_path)

    # Set solar exclusion angle and detector FOV for the single satellite
    sim_data['detector'].solarEx[0] = math.pi/2
    sim_data['detector'].fov[0] = math.pi/5

    # Generate 400 pointing points using the module's generate_pointing_sphere
    n_points_sphere = 400
    generate_pointing_sphere(sim_data, n_points_sphere)

    # Initialize pointing_state for the single detector
    # Assuming the first detector (index 0)
    sim_data['detector'].pointing_state[POINTING_COUNT_IDX, 0] = n_points_sphere
    sim_data['detector'].pointing_state[POINTING_PLACE_IDX, 0] = 0 # Start at the first point

    pointed_directions_history = []
    
    # Propagate the satellite once to get an initial position
    sim_data = celestial_update(sim_data, start_time)
    sim_data = propagate_satellites(sim_data, start_time, 'satellites')
    
    initial_sat_pos = sim_data['satellites']['position'][0]
    
    # --- Plot initialization with unit sphere ---
    fig = go.Figure()
    # Add a sphere at the origin to represent the pointing sphere itself for context
    sphere_radius = 1.0 # Unit sphere
    u_sphere = np.linspace(0, 2 * np.pi, 50)
    v_sphere = np.linspace(0, np.pi, 50)
    x_sphere = sphere_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = sphere_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale='Blues', showscale=False, opacity=0.1, name='Unit Sphere'
    ))
    
    # Add initial satellite position as a marker
    # The plot_3d_scatter function would normally plot initial_sat_pos scaled,
    # but here we are just adding it as a marker on the unit sphere for context
    # as the vectors are also on the unit sphere.
    initial_sat_direction = initial_sat_pos / np.linalg.norm(initial_sat_pos)
    fig.add_trace(go.Scatter3d(
        x=[initial_sat_direction[0]], y=[initial_sat_direction[1]], z=[initial_sat_direction[2]],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Initial Satellite Pointing'
    ))

    for i in range(400):
        update_detector_pointing(sim_data, debug=False) # Turn off debug
        current_pointed_direction = sim_data['detector'].pointing[0]
        snapshot = current_pointed_direction.copy()
#        print(f"Current pointed direction: {snapshot}") # Print current direction
        pointed_directions_history.append(snapshot)

    pointed_directions_history = np.array(pointed_directions_history)
#   print("Array pointedDirectionsHistory", pointed_directions_history)

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
    
    fig.update_layout(
        title="Detector Pointing with Exclusion",
        scene=dict(
            xaxis_title='X (Unit Vector)', yaxis_title='Y (Unit Vector)', zaxis_title='Z (Unit Vector)',
            aspectmode='data'
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )

#    print(pointed_directions_history[:100])
    fig.show() 
    return fig

def jerk(sim_data: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]:
    """
    Moves the pointing vector of specific satellites by 0.3 radians in a
    random direction.

    This function applies a random rotation to the satellites' pointing vectors.

    Args:
        sim_data: The main simulation data dictionary.
        satellite_indices: The indices of the satellites to modify.

    Returns:
        The modified sim_data with the updated pointing vectors.
    """
    if satellite_indices.size == 0:
        return sim_data

    p = sim_data['detector'].pointing[satellite_indices]
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

    sim_data['detector'].pointing[satellite_indices] = p_new

    return sim_data
