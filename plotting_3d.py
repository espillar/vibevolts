import numpy as np
from datetime import datetime
from typing import List, Optional

import plotly.graph_objects as go
from astropy.time import Time
from astropy.coordinates import GCRS, EarthLocation
import astropy.units as u

EARTH_RADIUS = 6378137.0

def plot_3d_scatter(
    positions: np.ndarray,
    title: str,
    plot_time: datetime,
    labels: Optional[List[str]] = None,
    marker_size: int = 1,
    trace_name: str = 'Points'
) -> go.Figure:
    """
    Creates a 3D plot of object positions.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions array must have shape (n, 3).")

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=np.arange(len(positions)),
            colorscale='Viridis',
            opacity=0.8
        ),
        text=labels,
        hoverinfo='text' if labels else 'none',
        name=trace_name
    ))

    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = EARTH_RADIUS * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = EARTH_RADIUS * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = EARTH_RADIUS * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Blues', showscale=False, opacity=0.5, name='Earth'
    ))

    theta = np.linspace(0, 2 * np.pi, 100)
    x_eq = EARTH_RADIUS * np.cos(theta)
    y_eq = EARTH_RADIUS * np.sin(theta)
    z_eq = np.zeros_like(theta)
    fig.add_trace(go.Scatter3d(
        x=x_eq, y=y_eq, z=z_eq, mode='lines',
        line=dict(color='green', width=3), name='Equator'
    ))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[EARTH_RADIUS * 1.1], mode='text',
        text=['N'], textfont=dict(size=15, color='red'), name='North Pole'
    ))

    lat_es, lon_es = 33.92 * u.deg, -118.42 * u.deg
    el_segundo_loc = EarthLocation.from_geodetic(lon=lon_es, lat=lat_es)
    itrs_coords = el_segundo_loc.get_itrs(obstime=Time(plot_time))
    gcrs_coords = itrs_coords.transform_to(GCRS(obstime=Time(plot_time)))
    es_pos = gcrs_coords.cartesian.xyz.to(u.m).value * 1.05
    fig.add_trace(go.Scatter3d(
        x=[es_pos[0]], y=[es_pos[1]], z=[es_pos[2]], mode='text',
        text=['ES'], textfont=dict(size=15, color='yellow'), name='El Segundo'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Objects'
    )

    return fig
