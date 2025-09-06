import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go

from simulation import initializeStructures, DETECTOR_SOLAR_EXCL_IDX, DETECTOR_LUNAR_EXCL_IDX, DETECTOR_EARTH_EXCL_IDX
from propagation import readtle, propagate_satellites, celestial_update
from visibility import exclusion

def demo_sky_scan() -> go.Figure:
    """
    Performs a sky scan from a GEO satellite to map celestial exclusion zones.
    """
    print("\n--- Starting Demo: Sky Scan from GEO ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

    # a. Initialize a simulation with a single GEO satellite
    tle_data = """GEO-SCAN
1 90401U 25007A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90401   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
"""
    dummy_tle_path = "dummy_tle_skyscan.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    orbital_elements, epochs = readtle(dummy_tle_path)
    sim_data = initializeStructures(
        num_satellites=1,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )
    sim_data['satellites']['orbital_elements'] = orbital_elements
    sim_data['satellites']['epochs'] = epochs

    # Set some reasonable exclusion angles
    sim_data['satellites']['detector'][:, DETECTOR_SOLAR_EXCL_IDX] = np.deg2rad(30)
    sim_data['satellites']['detector'][:, DETECTOR_LUNAR_EXCL_IDX] = np.deg2rad(15)
    sim_data['satellites']['detector'][:, DETECTOR_EARTH_EXCL_IDX] = np.deg2rad(10)

    # b. Update celestial body positions
    sim_data = propagate_satellites(sim_data, sim_start_time)
    sim_data = celestial_update(sim_data, sim_start_time)
    print("Initialized GEO satellite and celestial bodies.")

    # c. Create a 2D grid of pointing vectors
    declinations = np.linspace(-np.pi / 2, np.pi / 2, 50)
    right_ascensions = np.linspace(0, 2 * np.pi, 100)

    # d. & e. Iterate and store results
    sky_map = np.zeros((len(declinations), len(right_ascensions)))

    print("Scanning the sky for exclusion zones...")
    for i, dec in enumerate(declinations):
        for j, ra in enumerate(right_ascensions):
            # Convert spherical to Cartesian coordinates for the pointing vector
            x = np.cos(dec) * np.cos(ra)
            y = np.cos(dec) * np.sin(ra)
            z = np.sin(dec)
            pointing_vector = np.array([x, y, z])

            # Update the satellite's pointing vector
            sim_data['satellites']['pointing'][0] = pointing_vector

            # Call the exclusion function
            is_clear = exclusion(sim_data, satellite_index=0)
            sky_map[i, j] = is_clear

    print("Sky scan complete.")

    # f. Use Plotly to create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sky_map,
        x=np.rad2deg(right_ascensions),
        y=np.rad2deg(declinations),
        colorscale=[[0, 'red'], [1, 'lightgreen']],
        showscale=False,
        colorbar=dict(
            title='Visibility',
            tickvals=[0, 1],
            ticktext=['Excluded', 'Clear']
        )
    ))

    fig.update_layout(
        title='Sky Exclusion Map from a GEO Satellite',
        xaxis_title='Right Ascension (degrees)',
        yaxis_title='Declination (degrees)',
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig

if __name__ == '__main__':
    fig = demo_sky_scan()
    fig.show()
