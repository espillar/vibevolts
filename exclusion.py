import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple
import vibevolts as vv

# The astropy library is required for accurate astronomical calculations.
# You can install it with: pip install astropy jplephem
from astropy.time import Time
from astropy.coordinates import get_body, GCRS, solar_system_ephemeris

# The plotly library is required for 3D plotting.
# You can install it with: pip install plotly
import plotly.graph_objects as go

# --- Global Constants ---

# -- Radii in Meters --
earth_radius = 6378137.0
moon_radius = 1737400.0

# -- Detector Array Indices (from the original file) --
DETECTOR_SOLAR_EXCL_IDX = 4
DETECTOR_LUNAR_EXCL_IDX = 5
DETECTOR_EARTH_EXCL_IDX = 6

# --- New Exclusion Function ---

def exclusion(data_struct: Dict[str, Any], satellite_index: int) -> bool:
    """
    Determines if a satellite's pointing vector is excluded by the Sun, Moon, or Earth.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_index: The index of the satellite to check.

    Returns:
        True if the satellite's view is excluded, False otherwise.
    """
    # --- 1. Extract Data ---
    sat_pos = data_struct['satellites']['position'][satellite_index]
    sat_pointing = data_struct['satellites']['pointing'][satellite_index]
    sun_pos = data_struct['celestial']['position'][0]
    moon_pos = data_struct['celestial']['position'][1]
    
    # Get exclusion angles for the specific satellite
    detector_props = data_struct['satellites']['detector'][satellite_index]
    solar_excl_angle = detector_props[DETECTOR_SOLAR_EXCL_IDX]
    lunar_excl_angle = detector_props[DETECTOR_LUNAR_EXCL_IDX]
    earth_excl_angle = detector_props[DETECTOR_EARTH_EXCL_IDX]

    # --- 2. Compute Vectors and Normalize ---
    # Vector from satellite to celestial bodies
    vec_to_sun = sun_pos - sat_pos
    vec_to_moon = moon_pos - sat_pos
    vec_to_earth = -sat_pos  # Vector from satellite to Earth's center

    # Distances
    dist_to_sun = np.linalg.norm(vec_to_sun)
    dist_to_moon = np.linalg.norm(vec_to_moon)
    dist_to_earth = np.linalg.norm(vec_to_earth)
    
    # Normalize all relevant vectors to unit vectors
    # Handle potential zero-length vectors to avoid division by zero errors
    u_vec_to_sun = vec_to_sun / dist_to_sun if dist_to_sun > 0 else np.array([0., 0., 0.])
    u_vec_to_moon = vec_to_moon / dist_to_moon if dist_to_moon > 0 else np.array([0., 0., 0.])
    u_vec_to_earth = vec_to_earth / dist_to_earth if dist_to_earth > 0 else np.array([0., 0., 0.])
    
    norm_pointing = np.linalg.norm(sat_pointing)
    u_sat_pointing = sat_pointing / norm_pointing if norm_pointing > 0 else np.array([0., 0., 0.])

    # --- 3. Calculate Angles and Check for Exclusion ---
    sun_flag, moon_flag, earth_flag = False, False, False

    # -- Sun Exclusion --
    # Angle between pointing vector and sun vector
    cos_angle_sun = np.clip(np.dot(u_sat_pointing, u_vec_to_sun), -1.0, 1.0)
    angle_sun = np.arccos(cos_angle_sun)
    if angle_sun < solar_excl_angle:
        sun_flag = True

    # -- Moon Exclusion --
    # Angle between pointing vector and moon vector
    cos_angle_moon = np.clip(np.dot(u_sat_pointing, u_vec_to_moon), -1.0, 1.0)
    angle_moon = np.arccos(cos_angle_moon)
    # Apparent angular radius of the Moon from the satellite's perspective
    apparent_radius_moon = np.arctan(moon_radius / dist_to_moon) if dist_to_moon > 0 else 0
    if (angle_moon - apparent_radius_moon) < lunar_excl_angle:
        moon_flag = True

    # -- Earth Exclusion --
    # Angle between pointing vector and Earth vector
    cos_angle_earth = np.clip(np.dot(u_sat_pointing, u_vec_to_earth), -1.0, 1.0)
    angle_earth = np.arccos(cos_angle_earth)
    # Apparent angular radius of the Earth from the satellite's perspective
    apparent_radius_earth = np.arctan(earth_radius / dist_to_earth) if dist_to_earth > 0 else 0
    if (angle_earth - apparent_radius_earth) < earth_excl_angle:
        earth_flag = True

    # --- 4. Set Global Exclusion Flag and Return ---
    is_excluded = sun_flag or moon_flag or earth_flag
    return is_excluded


def check_all_exclusions(data_struct: Dict[str, Any]) -> np.ndarray:
    """
    Checks exclusion status for all satellites in the 'satellites' category.

    Args:
        data_struct: The main simulation data dictionary.

    Returns:
        A NumPy array of integers (0 or 1) where 1 indicates an excluded view
        and 0 indicates a clear view for each satellite.
    """
    num_satellites = data_struct['counts']['satellites']
    exclusion_vector = np.zeros(num_satellites, dtype=int)

    for i in range(num_satellites):
        if exclusion(data_struct, i):
            exclusion_vector[i] = 1

    return exclusion_vector


# --- New Testing Function ---

def test_exclusion_plot():
    """
    Tests the exclusion function by creating a scenario with 100 satellites
    in random orbits with random pointing vectors and plots the results for 15 of them.
    The last 5 plotted satellites are restricted to LEO.
    """
    print("--- Starting Exclusion Test and Plotting Demo ---")
    sim_start_time = datetime(2025, 8, 16, 10, 22, 0, tzinfo=timezone.utc)
    num_sats = 100

    # --- 1. Initialize Simulation ---
    # Ensure planetary ephemeris data is available
    solar_system_ephemeris.set('jpl')
    
    sim_data = vv.initializeStructures(num_sats, 0, 0, sim_start_time)
    sim_data = vv.celestial_update(sim_data, sim_start_time)
    
    # --- 2. Create Random Satellites ---
    leo_radius = earth_radius + 500e3  # 500 km altitude
    leo_max_radius = earth_radius + 2000e3 # 2000 km altitude
    geo_radius = 42164e3 # Geostationary radius

    # Generate radii: 10 LEO-GEO, 5 LEO-only, and the rest LEO-GEO
    radii_part1 = np.random.uniform(leo_radius, geo_radius, 10)
    radii_part2_leo = np.random.uniform(leo_radius, leo_max_radius, 5)
    radii_part3 = np.random.uniform(leo_radius, geo_radius, num_sats - 15)
    radii = np.concatenate((radii_part1, radii_part2_leo, radii_part3))

    # Random positions on a sphere for each radius
    phi = np.random.uniform(0, 2 * np.pi, num_sats)
    costheta = np.random.uniform(-1, 1, num_sats)
    theta = np.arccos(costheta)
    
    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    sim_data['satellites']['position'] = np.vstack((x, y, z)).T

    # Random pointing vectors (normalized)
    rand_vecs = np.random.randn(num_sats, 3)
    norms = np.linalg.norm(rand_vecs, axis=1)[:, np.newaxis]
    sim_data['satellites']['pointing'] = rand_vecs / norms
    
    # Set fixed exclusion angles (in radians)
    exclusion_angle = 1.0
    print(f"Setting all exclusion angles (Solar, Lunar, Earth) to: {exclusion_angle} radians")
    sim_data['satellites']['detector'][:, DETECTOR_SOLAR_EXCL_IDX] = exclusion_angle
    sim_data['satellites']['detector'][:, DETECTOR_LUNAR_EXCL_IDX] = exclusion_angle
    sim_data['satellites']['detector'][:, DETECTOR_EARTH_EXCL_IDX] = exclusion_angle

    # --- 3. Loop, Call Exclusion, and Plot for 15 examples ---
    print(f"Running exclusion check for {num_sats} satellites and plotting 15 examples...")
    
    sun_pos = sim_data['celestial']['position'][0]
    moon_pos = sim_data['celestial']['position'][1]
    
    for i in range(15): # Plot the first 15 to avoid too many windows
        is_excluded = exclusion(sim_data, i)
        status = "EXCLUDED" if is_excluded else "CLEAR"
        
        sat_pos = sim_data['satellites']['position'][i]
        
        fig = go.Figure()

        # Add Earth sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_e = earth_radius * np.outer(np.cos(u), np.sin(v))
        y_e = earth_radius * np.outer(np.sin(u), np.sin(v))
        z_e = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        fig.add_trace(go.Surface(x=x_e, y=y_e, z=z_e, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))

        # Add Satellite marker
        fig.add_trace(go.Scatter3d(
            x=[sat_pos[0]], y=[sat_pos[1]], z=[sat_pos[2]],
            mode='markers', marker=dict(size=8, color='red'), name=f'Satellite {i}'
        ))

        # Add vectors from satellite
        vector_scale = 0.5 * np.linalg.norm(sat_pos)
        
        # Pointing vector
        p_vec = sim_data['satellites']['pointing'][i] * vector_scale
        fig.add_trace(go.Scatter3d(
            x=[sat_pos[0], sat_pos[0] + p_vec[0]], y=[sat_pos[1], sat_pos[1] + p_vec[1]], z=[sat_pos[2], sat_pos[2] + p_vec[2]],
            mode='lines', line=dict(color='red', width=5), name='Pointing Vector'
        ))
        
        # Vector to Sun
        s_vec = (sun_pos - sat_pos)
        s_vec = s_vec / np.linalg.norm(s_vec) * vector_scale * 1.5
        fig.add_trace(go.Scatter3d(
            x=[sat_pos[0], sat_pos[0] + s_vec[0]], y=[sat_pos[1], sat_pos[1] + s_vec[1]], z=[sat_pos[2], sat_pos[2] + s_vec[2]],
            mode='lines', line=dict(color='yellow', width=5), name='To Sun'
        ))

        # Vector to Moon
        m_vec = (moon_pos - sat_pos)
        m_vec = m_vec / np.linalg.norm(m_vec) * vector_scale
        fig.add_trace(go.Scatter3d(
            x=[sat_pos[0], sat_pos[0] + m_vec[0]], y=[sat_pos[1], sat_pos[1] + m_vec[1]], z=[sat_pos[2], sat_pos[2] + m_vec[2]],
            mode='lines', line=dict(color='gray', width=5), name='To Moon'
        ))

        # Vector to Earth
        e_vec = (-sat_pos)
        e_vec = e_vec / np.linalg.norm(e_vec) * vector_scale
        fig.add_trace(go.Scatter3d(
            x=[sat_pos[0], sat_pos[0] + e_vec[0]], y=[sat_pos[1], sat_pos[1] + e_vec[1]], z=[sat_pos[2], sat_pos[2] + e_vec[2]],
            mode='lines', line=dict(color='blue', width=5), name='To Earth'
        ))

        fig.update_layout(
            title=f"Satellite {i} Viewpoint Check - Status: {status}",
            scene=dict(
                xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                aspectmode='data'
            ),
            margin=dict(r=10, b=10, l=10, t=40)
        )
        fig.show()

    # --- 4. Test the new check_all_exclusions function ---
    print("\n--- Testing check_all_exclusions function ---")
    exclusion_results = check_all_exclusions(sim_data)
    print(f"Exclusion vector for all {num_sats} satellites:")
    print(exclusion_results)
    print(f"Total excluded satellites: {np.sum(exclusion_results)}")


# --- Main Execution Block ---
if __name__ == '__main__':
    test_exclusion_plot()
