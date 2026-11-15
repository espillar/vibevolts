import numpy as np
from typing import Any
import math
from datetime import datetime, timezone
import plotly.graph_objects as go

from minimalsimulation import create_empty_simulation
from celestialbodies import add_celestial_bodies, celestial_update
from propagation import add_satellites_from_tle, propagate_satellites
from plotting_3d import add_sphere_trace, apply_3d_scene_layout, tag_demo_figure
from constants import POINTING_COUNT_IDX, POINTING_PLACE_IDX
from fibonacciSearch import pointing_vectors, resort_vectors_by_proximity
from exclusion import exclusion


def generate_pointing_sphere(sim_data: Any, n_points: int, debug: bool = False) -> None:
    """
    Generates a pointing sphere with n_points and stores it in the sim_data.pointing_spheres[n_points]
    A pointing sphere is a 3 by n_points numpy array with the 3 representing unit vectors to be
    pointed to
    These positions will be used by the update_detector_pointing to
    point sensors incrementaly.
    The index is sim_data.pointing_spheres[n] 
    If a sphere with the same number of points already exists, this function does nothing.
    The current version of the code resorts the vector by proximity to make the sky search
    more efficient, although this is not deeply optimized yet.
    """
    if n_points not in sim_data.pointing_spheres:
        print(f"Generating pointing sphere with {n_points} points...")
        sim_data.pointing_spheres[n_points] = \
           resort_vectors_by_proximity(pointing_vectors(n_points))

    if debug:
        print(f"\n--- Debugging Pointing Sphere (n_points={n_points}) ---")
        generated_vectors = sim_data.pointing_spheres[n_points]
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


def _get_asset_detectors(sim_data: Any, sat_category: str = None):
    """
    Helper function to yield (component_detector, global_offset_index) pairs
    for active asset categories (satellites, observatories).
    """
    offset = 0
    for name in ('satellites', 'observatories'):
        comp = getattr(sim_data, name, None)
        if comp is not None and getattr(comp, 'detector', None) is not None:
            det = comp.detector
            if len(det.filt) > 0:
                if sat_category is None or sat_category == name:
                    yield det, offset
                offset += len(det.filt)


def update_detector_pointing(
    sim_data: Any,
    sat_category: str = None,
    debug: bool = False,
) -> None:
    """
    Updates the pointing vector for each detector across satellites and observatories,
    skipping excluded pointing directions.
    """
    for det, global_offset in _get_asset_detectors(sim_data, sat_category=sat_category):
        num_detectors = len(det.filt)
        pointing_state = det.pointing_state
        pointing_vectors_all = det.pointing

        for i_local in range(num_detectors):
            i_global = global_offset + i_local
            count = int(pointing_state[i_local, POINTING_COUNT_IDX])
            if count == 0:
                continue
            grid = sim_data.pointing_spheres[count]

            place = int(pointing_state[i_local, POINTING_PLACE_IDX])
            start_place = place

            while True:
                place += 1
                if place >= count:
                    place = 0
                pointing_vectors_all[i_local] = grid[place]

                excluded = exclusion(sim_data, i_global)
                if debug:
                    print(f"Detector {i_global} (local {i_local}): Pointing location {place}, Excluded: {excluded != 0}")
                    print(grid[place])

                if excluded == 0:
                    pointing_state[i_local, POINTING_PLACE_IDX] = place
                    break

                if place == start_place:
                    print(f"Warning: Detector {i_global} has all pointing vectors excluded.")
                    pointing_state[i_local, POINTING_PLACE_IDX] = place
                    break


def detectorPointingInitialize(sim_data: Any, grid_points: int) -> None:
    """
    Initializes pointing and pointing_state inside component detectors
    (satellites and observatories), and adds a pointing sphere to sim_data.
    """
    generate_pointing_sphere(sim_data, grid_points)

    for det, _ in _get_asset_detectors(sim_data):
        sensor_count = len(det.filt)
        det.pointing_state = np.zeros((sensor_count, 2), dtype=int)
        det.pointing_state[:, POINTING_COUNT_IDX] = grid_points
        det.pointing_state[:, POINTING_PLACE_IDX] = np.random.randint(
            0, grid_points - 1, size=sensor_count
        )

    update_detector_pointing(sim_data)


def demo_exclusion_pointing():
    """
    Demonstrates satellite pointing with a solar exclusion angle and a detector
    field of view, plotting the pointing history on a sphere.
    """
    start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)
    add_celestial_bodies(sim_data)

    # Use a single satellite TLE string for this demo
    single_tle_content = """GEO-SAT
1 99999U 99999A   25001.00000000  .00000000  00000-0  00000-0 0  9999
2 99999   0.0001 254.9961 0000001  98.4322 261.6813  1.00271900    10"""

    add_satellites_from_tle(sim_data, single_tle_content, 'satellites')

    # Set solar exclusion angle and detector FOV for the single satellite
    sim_data.satellites.detector.solarEx[0] = math.pi/2
    sim_data.satellites.detector.fov[0] = math.pi/5

    # Generate 400 pointing points using the module's generate_pointing_sphere
    n_points_sphere = 400
    generate_pointing_sphere(sim_data, n_points_sphere)

    # Initialize pointing_state for the single detector
    # Assuming the first detector (index 0)
    sim_data.satellites.detector.pointing_state[0, POINTING_COUNT_IDX] = n_points_sphere
    sim_data.satellites.detector.pointing_state[0, POINTING_PLACE_IDX] = 0 # Start at the first point

    pointed_directions_history = []
    
    # Propagate the satellite once to get an initial position
    sim_data = celestial_update(sim_data, start_time)
    sim_data = propagate_satellites(sim_data, start_time, 'satellites')
    
    initial_sat_pos = sim_data.satellites.position[0]
    
    # --- Plot initialization with unit sphere ---
    fig = go.Figure()
    # Add a sphere at the origin to represent the pointing sphere itself for context
    add_sphere_trace(
        fig, radius=1.0, name='Unit Sphere', opacity=0.1, resolution=50
    )
    
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
        update_detector_pointing(sim_data, debug=False)
        current_pointed_direction = sim_data.detector.pointing[0]
        snapshot = current_pointed_direction.copy()
        pointed_directions_history.append(snapshot)

    pointed_directions_history = np.array(pointed_directions_history)

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
    
    apply_3d_scene_layout(fig, "Detector Pointing with Exclusion", axis_unit=' (Unit Vector)')

#    print(pointed_directions_history[:100])
    return tag_demo_figure(fig, "demo_exclusion_pointing")

