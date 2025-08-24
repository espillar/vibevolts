import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any

import plotly.graph_objects as go
from astropy.time import Time
from astropy.coordinates import GCRS, EarthLocation
import astropy.units as u

# Constant for Earth's radius in meters
EARTH_RADIUS = 6378137.0


def plot_3d_scatter(
    positions: np.ndarray,
    title: str,
    plot_time: datetime,
    labels: Optional[List[str]] = None,
    marker_size: int = 1,
    trace_name: str = 'Points'
):
    """
    Displays a 3D interactive plot of object positions with Earth references.

    Args:
        positions: An (n x 3) NumPy array of (x, y, z) positions in meters.
        title: The title for the plot.
        plot_time: The UTC datetime for which the plot is generated. This is
                   used to correctly orient the Earth.
        labels: An optional list of names for each point to display on hover.
        marker_size: The size of the markers in the plot. Defaults to 1.
        trace_name: The name for the trace to appear in the legend.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions array must have shape (n, 3).")

    fig = go.Figure()

    # Add the main scatter plot trace for the positions
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=np.arange(len(positions)),  # Color by index
            colorscale='Viridis',
            opacity=0.8
        ),
        text=labels,
        hoverinfo='text' if labels else 'none',
        name=trace_name
    ))

    # --- Add Earth and reference markers ---
    # Add a sphere to represent the Earth
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = EARTH_RADIUS * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = EARTH_RADIUS * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = EARTH_RADIUS * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Blues', showscale=False, opacity=0.5, name='Earth'
    ))

    # Add Equator line
    theta = np.linspace(0, 2 * np.pi, 100)
    x_eq = EARTH_RADIUS * np.cos(theta)
    y_eq = EARTH_RADIUS * np.sin(theta)
    z_eq = np.zeros_like(theta)
    fig.add_trace(go.Scatter3d(
        x=x_eq, y=y_eq, z=z_eq, mode='lines',
        line=dict(color='green', width=3), name='Equator'
    ))

    # Add North Pole marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[EARTH_RADIUS * 1.1], mode='text',
        text=['N'], textfont=dict(size=15, color='red'), name='North Pole'
    ))

    # Add El Segundo marker (approximated location)
    lat_es, lon_es = 33.92 * u.deg, -118.42 * u.deg
    el_segundo_loc = EarthLocation.from_geodetic(lon=lon_es, lat=lat_es)
    itrs_coords = el_segundo_loc.get_itrs(obstime=Time(plot_time))
    gcrs_coords = itrs_coords.transform_to(GCRS(obstime=Time(plot_time)))
    es_pos = gcrs_coords.cartesian.xyz.to(u.m).value * 1.05
    fig.add_trace(go.Scatter3d(
        x=[es_pos[0]], y=[es_pos[1]], z=[es_pos[2]], mode='text',
        text=['ES'], textfont=dict(size=15, color='yellow'), name='El Segundo'
    ))

    # --- Final Layout Configuration ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'  # Ensures a 1:1:1 aspect ratio
        ),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Objects'
    )
    fig.show()


def plot_pointing_vectors(data_struct: Dict[str, Any], title: str, plot_time: datetime):
    """
    Displays a 3D plot of satellites with their pointing vectors.

    Args:
        data_struct: The main simulation data dictionary.
        title: The title for the plot.
        plot_time: The UTC datetime for the plot, used for Earth orientation.
    """
    sat_positions = data_struct['satellites']['position']
    sat_pointing = data_struct['satellites']['pointing']
    num_sats = data_struct['counts']['satellites']

    if num_sats == 0:
        print("No satellites to plot.")
        return

    fig = go.Figure()

    # Add satellite markers
    fig.add_trace(go.Scatter3d(
        x=sat_positions[:, 0],
        y=sat_positions[:, 1],
        z=sat_positions[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        name='Satellites'
    ))

    # Add pointing vectors for each satellite
    vector_scale = 0.5 * EARTH_RADIUS
    for i in range(num_sats):
        start_point = sat_positions[i]

        pointing_vec = sat_pointing[i]
        norm = np.linalg.norm(pointing_vec)
        unit_vec = pointing_vec / norm if norm > 0 else np.array([0, 0, 0])

        end_point = start_point + unit_vec * vector_scale

        fig.add_trace(go.Scatter3d(
            x=[start_point[0], end_point[0]],
            y=[start_point[1], end_point[1]],
            z=[start_point[2], end_point[2]],
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False
        ))

    # Add a sphere to represent the Earth
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = EARTH_RADIUS * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = EARTH_RADIUS * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = EARTH_RADIUS * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Blues', showscale=False, opacity=0.5, name='Earth'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )
    fig.show()
