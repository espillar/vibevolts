import numpy as np
from datetime import datetime
from typing import Dict, Any

import plotly.graph_objects as go

EARTH_RADIUS = 6378137.0

def plot_pointing_vectors(
    data_struct: Dict[str, Any],
    title: str,
    plot_time: datetime
) -> go.Figure:
    """
    Creates a 3D plot of satellites with pointing vectors.
    """
    sat_positions = data_struct['satellites']['position']
    sat_pointing = data_struct['satellites']['pointing']
    num_sats = data_struct['counts']['satellites']

    if num_sats == 0:
        print("No satellites to plot.")
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=sat_positions[:, 0], y=sat_positions[:, 1], z=sat_positions[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        name='Satellites'
    ))

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
            xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(r=20, b=10, l=10, t=40)
    )

    return fig
