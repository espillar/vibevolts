import numpy as np
import plotly.graph_objects as go

from demo_common import run_trajectory_demo


def demo3() -> go.Figure:
    """
    Runs a demonstration plotting a single LEO satellite trajectory.

    Returns:
        The Plotly figure object.
    """
    tle_data = '''LEO-TRAJECTORY
1 90201U 25005A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90201  51.6000  10.0000 0010000  90.0000  20.0000 15.50000000    11
'''
    print("\n--- Starting Demo 3 ---")
    return run_trajectory_demo(
        tle_data=tle_data,
        time_steps=np.arange(0, 91, 10),
        step_unit='minutes',
        unit_label='min',
        orbit_label='LEO',
        title='Single LEO Satellite Trajectory over 90 Minutes',
        demo_name='demo3',
        line_color='red',
        marker_color='blue',
    )
