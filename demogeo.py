import numpy as np
import plotly.graph_objects as go

from demo_common import get_sample_geo_tles, run_trajectory_demo


def demogeo() -> go.Figure:
    """
    Runs a demonstration plotting a single GEO satellite trajectory.

    Returns:
        The Plotly figure object.
    """
    print("\n--- Starting Demo 4 ---")
    return run_trajectory_demo(
        tle_data=get_sample_geo_tles(1),
        time_steps=np.arange(0, 24, 1),
        step_unit='hours',
        unit_label='hr',
        orbit_label='GEO',
        title='Single GEO Satellite Trajectory over 23 Hours',
        demo_name='demogeo',
        line_color='purple',
        marker_color='orange',
    )
