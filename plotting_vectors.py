import numpy as np
from datetime import datetime
from typing import Dict, Any

import plotly.graph_objects as go

from constants import EARTH_RADIUS
from plotting_3d import add_sphere_trace, add_vector_traces, apply_3d_scene_layout

def plot_pointing_vectors(
    data_struct: Dict[str, Any],
    title: str,
    plot_time: datetime
) -> go.Figure:
    """
    Creates a 3D plot of satellites with pointing vectors.
    """
    num_sats = data_struct.counts.get('satellites', 0) if hasattr(data_struct, 'counts') and data_struct.counts else 0

    if num_sats == 0 or not hasattr(data_struct, 'satellites') or data_struct.satellites is None:
        print("No satellites to plot.")
        return go.Figure()

    sat_positions = data_struct.satellites.position.copy()
    sat_det = getattr(data_struct.satellites, 'detector', None)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=sat_positions[:, 0], y=sat_positions[:, 1], z=sat_positions[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        name='Satellites'
    ))

    if sat_det is not None and len(sat_det.pointing) > 0:
        add_vector_traces(
            fig, sat_positions, sat_det.pointing[:num_sats],
            scale=0.5 * EARTH_RADIUS, color='red', width=3,
        )

    add_sphere_trace(fig, radius=EARTH_RADIUS, name='Earth')

    apply_3d_scene_layout(fig, title)

    return fig
