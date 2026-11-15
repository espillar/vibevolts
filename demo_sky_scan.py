
import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go

from minimalsimulation import create_empty_simulation
from celestialbodies import add_celestial_bodies, celestial_update
from propagation import add_satellites_from_tle, propagate_satellites
from demo_common import get_sample_geo_tles, set_exclusion_deg
from exclusion import exclusion
from plotting_3d import tag_demo_figure

def demo_sky_scan() -> go.Figure:
    """
    Performs a sky scan from a GEO satellite to map celestial exclusion zones.
    """
    print("\n--- Starting Demo: Sky Scan from GEO ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

    sim_data = create_empty_simulation(sim_start_time)
    add_satellites_from_tle(sim_data, get_sample_geo_tles(1), 'satellites')
    add_celestial_bodies(sim_data)

    # Set some reasonable exclusion angles
    set_exclusion_deg(sim_data, 30, 15, 10)

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
            sim_data.detector.pointing[0] = pointing_vector

            # Call the exclusion function
            is_excluded = exclusion(sim_data, satellite_index=0)
            sky_map[i, j] = is_excluded

    print("Sky scan complete.")

    # f. Use Plotly to create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=sky_map,
        x=np.rad2deg(right_ascensions),
        y=np.rad2deg(declinations),
        colorscale=[[0, 'lightgreen'], [1, 'red']],
        showscale=False,
        colorbar=dict(
            title='Exclusion',
            tickvals=[0, 1],
            ticktext=['Clear', 'Excluded']
        )
    ))

    fig.update_layout(
        title='Sky Exclusion Map from a GEO Satellite',
        xaxis_title='Right Ascension (degrees)',
        yaxis_title='Declination (degrees)',
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return tag_demo_figure(fig, "demo_sky_scan")

if __name__ == '__main__':
    fig = demo_sky_scan()
    fig.show()
