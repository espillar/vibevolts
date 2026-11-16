from datetime import datetime, timezone, timedelta
from typing import Any, Callable, List, Optional, Union
import math
import numpy as np
import plotly.graph_objects as go
from astropy.coordinates import solar_system_ephemeris

from cadenceController import initCadence, nextIntegration
from cislunar import demo_cislunar
from celestialbodies import add_celestial_bodies, celestial_update
from constants import ARCSEC, DEGREE, EARTH_RADIUS, POINTING_COUNT_IDX, POINTING_PLACE_IDX
from constellation import geos
from dataHandling import DataHandler
import detector
from detector import makeDetector, requiredIntegrationTime
from exclusion import exclusion, update_exclusion_table
from lambertian import includedAngle, lambertiansphere
from minimalsimulation import create_empty_simulation
from observatories import add_observatories, propagate_observatories
from plotting_3d import (
    add_sphere_trace,
    add_vector_traces,
    apply_3d_scene_layout,
    plot_3d_scatter,
    plot_pointing_vectors,
    tag_demo_figure,
)
from pointing import (
    detectorPointingInitialize,
    generate_pointing_sphere,
    test_vector_resorting,
    update_detector_pointing,
)
from propagation import add_satellites_from_tle, propagate_satellites
from scandetectors import scandetectors
from sim_report import print_simulation_state, sim_check
from targets import add_fixed_points, fixedSat, fixedTarget, fixSun


STANDARD_TLE_DATA = """ISS (ZARYA)
1 25544U 98067A   25209.52203988  .00012111  00000+0  22159-3 0  9991
2 25544  51.6412 254.9961 0006733  98.4322 261.6813 15.49493393462383
NOAA 19
1 33591U 09005A   25209.38959223  .00000100  00000+0  97987-4 0  9993
2 33591  99.1533 244.3362 0013327 101.3725 258.7562 14.12510122810029
HST
1 20580U 90037B   25208.83160218  .00000113  00000+0  35999-4 0  9990
2 20580  28.4695 177.8391 0001259 138.5273 221.5822 15.09326468 23453
GEO-01
1 90001U 25001A   25209.50000000  .00000000  00000-0  00000-0 0  9991
2 90001   0.0500   0.0000 0001000   0.0000   0.0000  1.00270000    12
GEO-02
1 90002U 25001B   25209.50000000  .00000000  00000-0  00000-0 0  9992
2 90002   0.0500  36.0000 0001000   0.0000   0.0000  1.00270000    13
GEO-03
1 90003U 25001C   25209.50000000  .00000000  00000-0  00000-0 0  9993
2 90003   0.0500  72.0000 0001000   0.0000  45.0000  1.00270000    14
GEO-04
1 90004U 25001D   25209.50000000  .00000000  00000-0  00000-0 0  9994
2 90004   0.0500 108.0000 0001000   0.0000  90.0000  1.00270000    15
GEO-05
1 90005U 25001E   25209.50000000  .00000000  00000-0  00000-0 0  9995
2 90005   0.0500 144.0000 0001000   0.0000 135.0000  1.00270000    16
GEO-06
1 90006U 25001F   25209.50000000  .00000000  00000-0  00000-0 0  9996
2 90006   0.0500 180.0000 0001000   0.0000 180.0000  1.00270000    17
GEO-07
1 90007U 25001G   25209.50000000  .00000000  00000-0  00000-0 0  9997
2 90007   0.0500 216.0000 0001000   0.0000 225.0000  1.00270000    18
GEO-08
1 90008U 25001H   25209.50000000  .00000000  00000-0  00000-0 0  9998
2 90008   0.0500 252.0000 0001000   0.0000 270.0000  1.00270000    19
GEO-09
1 90009U 25001I   25209.50000000  .00000000  00000-0  00000-0 0  9999
2 90009   0.0500 288.0000 0001000   0.0000 315.0000  1.00270000    10
GEO-10
1 90010U 25001J   25209.50000000  .00000000  00000-0  00000-0 0  9990
2 90010   0.0500 324.0000 0001000   0.0000 360.0000  1.00270000    11
HEO-01 (MOLNIYA)
1 90011U 25002A   25209.50000000  .00000000  00000-0  00000-0 0  9995
2 90011  63.4000  50.0000 7500000  270.0000  45.0000  2.00560000    13
HEO-02 (MOLNIYA)
1 90012U 25002B   25209.50000000  .00000000  00000-0  00000-0 0  9996
2 90012  63.4000 110.0000 7500000  270.0000  90.0000  2.00560000    14
HEO-03 (MOLNIYA)
1 90013U 25002C   25209.50000000  .00000000  00000-0  00000-0 0  9997
2 90013  63.4000 170.0000 7500000  270.0000 135.0000  2.00560000    15
HEO-04 (MOLNIYA)
1 90014U 25002D   25209.50000000  .00000000  00000-0  00000-0 0  9998
2 90014  63.4000 230.0000 7500000  270.0000 180.0000  2.00560000    16
HEO-05 (MOLNIYA)
1 90015U 25002E   25209.50000000  .00000000  00000-0  00000-0 0  9999
2 90015  63.4000 290.0000 7500000  270.0000 225.0000  2.00560000    17
LEO-01 (POLAR)
1 90016U 25003A   25209.50000000  .00000000  00000-0  00000-0 0  9990
2 90016  98.0000 120.0000 0010000  90.0000  20.0000 14.50000000    18
LEO-02 (POLAR)
1 90017U 25003B   25209.50000000  .00000000  00000-0  00000-0 0  9991
2 90017  98.0000 240.0000 0010000  90.0000  40.0000 14.50000000    19
LEO-03
1 90018U 25003C   25209.50000000  .00000000  00000-0  00000-0 0  9992
2 90018  45.0000  80.0000 0010000  45.0000  60.0000 15.00000000    10
LEO-04
1 90019U 25003D   25209.50000000  .00000000  00000-0  00000-0 0  9993
2 90019  45.0000 200.0000 0010000  45.0000  80.0000 15.00000000    11
LEO-05
1 90020U 25003E   25209.50000000  .00000000  00000-0  00000-0 0  9994
2 90020  28.5000  10.0000 0010000  10.0000 100.0000 15.50000000    12
"""

FOUR_GEO_TLE_DATA = """SAT-1
1 90401U 25007A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90401   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
SAT-2
1 90402U 25007B   25210.50000000  .00000000  00000-0  00000-0 0  9992
2 90402   0.0500  45.0000 0001000  90.0000  60.0000  1.00270000    12
SAT-3
1 90403U 25007C   25210.50000000  .00000000  00000-0  00000-0 0  9993
2 90403   0.0500  45.0000 0001000  90.0000 100.0000  1.00270000    13
SAT-4
1 90404U 25007D   25210.50000000  .00000000  00000-0  00000-0 0  9994
2 90404   0.0500  45.0000 0001000  90.0000 140.0000  1.00270000    14
"""

LEO_TRAJECTORY_TLE = """LEO-TRAJECTORY
1 90201U 25005A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90201  51.6000  10.0000 0010000  90.0000  20.0000 15.50000000    11
"""


def get_sample_geo_tles(n: int = 4) -> str:
    """Returns the first n satellites (3 lines each) from FOUR_GEO_TLE_DATA."""
    lines = [line.strip() for line in FOUR_GEO_TLE_DATA.strip().splitlines() if line.strip()]
    return "\n".join(lines[: 3 * n])


def set_exclusion_deg(
    sim_data: Any,
    solar: float = 30.0,
    lunar: float = 15.0,
    earth: float = 10.0,
) -> None:
    """Sets solar, lunar, and Earth exclusion angles (in degrees) on sim_data.detector."""
    sim_data.detector.solarEx[:] = np.deg2rad(solar)
    sim_data.detector.lunarEx[:] = np.deg2rad(lunar)
    sim_data.detector.earthEx[:] = np.deg2rad(earth)


def initialize_standard_simulation(start_time: Optional[datetime] = None) -> Any:
    """Initializes and propagates the standard 23-satellite simulation with bodies and targets."""
    if start_time is None:
        start_time = datetime.now(tz=timezone.utc)
    sim_data = create_empty_simulation(start_time)
    add_satellites_from_tle(sim_data, STANDARD_TLE_DATA, 'satellites')
    add_observatories(sim_data, 0)
    add_celestial_bodies(sim_data)
    add_fixed_points(sim_data, 100)
    num_sats = sim_data.counts.satellites
    sim_data.satellites.detector = detector.makeDetector(
        num_sats, "V", 10 * DEGREE, 3 * ARCSEC, 1
    )
    print(f"Initializing standard simulation with {num_sats} satellites.")
    solar_system_ephemeris.set('jpl')
    print(f"Propagating satellites to start time: {start_time.isoformat()} to set initial state.")
    return propagate_satellites(sim_data, start_time)


def run_trajectory_demo(
    tle_data: str,
    time_steps: np.ndarray,
    step_unit: str,
    unit_label: str,
    orbit_label: str,
    title: str,
    demo_name: str,
    line_color: str,
    marker_color: str,
) -> go.Figure:
    """Propagates a single satellite across time_steps and plots its 3D trajectory."""
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_satellites_from_tle(sim_data, tle_data, 'satellites')
    print(f"Initializing structures for {sim_data.counts.satellites} {orbit_label} satellite.")

    positions = []
    for step in time_steps:
        prop_time = sim_start_time + timedelta(**{step_unit: int(step)})
        sim_data = propagate_satellites(sim_data, prop_time)
        sim_data.time = prop_time
        positions.append(sim_data.satellites.position[0].copy())
    positions_array = np.array(positions)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='lines', line=dict(color=line_color, width=4), name='Trajectory'
    ))
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='markers', marker=dict(size=5, color=marker_color),
        text=[f'T={t} {unit_label}' for t in time_steps], hoverinfo='text', name='Time Steps'
    ))
    add_sphere_trace(fig, radius=EARTH_RADIUS, name='Earth')
    apply_3d_scene_layout(fig, title, legend_title='Trace')
    return tag_demo_figure(fig, demo_name)


# --- Orbits & Constellations Demos ---

def demo2() -> go.Figure:
    """Plots 23 satellites and Sun/Moon vectors at T=0s and T=300s."""
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    positions_t0 = sim_data.satellites.position.copy()
    sim_data = celestial_update(sim_data, sim_start_time)
    celestial_pos_t0 = sim_data.celestial.position.copy()

    time_t1 = sim_start_time + timedelta(seconds=300)
    sim_data = propagate_satellites(sim_data, time_t1)
    positions_t1 = sim_data.satellites.position.copy()
    sim_data = celestial_update(sim_data, time_t1)
    celestial_pos_t1 = sim_data.celestial.position.copy()

    print("\n--- Generating 3D plot for Demo 2 ---")
    vector_scale = 10 * EARTH_RADIUS
    fig = go.Figure()
    snapshots = [
        ('0s', positions_t0, celestial_pos_t0, 'blue', 'orange', 'gray'),
        ('300s', positions_t1, celestial_pos_t1, 'red', 'yellow', 'lightgray'),
    ]
    for label, sat_pos, cel_pos, sat_color, sun_color, moon_color in snapshots:
        fig.add_trace(go.Scatter3d(
            x=sat_pos[:, 0], y=sat_pos[:, 1], z=sat_pos[:, 2],
            mode='markers', marker=dict(size=5, color=sat_color), name=f'Sats (T={label})'
        ))
        add_vector_traces(fig, [0, 0, 0], cel_pos[0], scale=vector_scale,
                          color=sun_color, width=4, name=f'Sun (T={label})')
        add_vector_traces(fig, [0, 0, 0], cel_pos[1], scale=vector_scale,
                          color=moon_color, width=4, name=f'Moon (T={label})')

    add_sphere_trace(fig, radius=EARTH_RADIUS, name='Earth')
    apply_3d_scene_layout(fig, "Satellite and Celestial Positions at 0 and 300 seconds")
    return tag_demo_figure(fig, "demo2")


def demo_trajectories() -> tuple:
    """Demonstrates single-satellite LEO (90 min) and GEO (23 hr) trajectories."""
    print("\n--- Starting Demo: LEO & GEO Trajectories ---")
    fig_leo = run_trajectory_demo(
        tle_data=LEO_TRAJECTORY_TLE,
        time_steps=np.arange(0, 91, 10),
        step_unit='minutes',
        unit_label='min',
        orbit_label='LEO',
        title='Single LEO Satellite Trajectory over 90 Minutes',
        demo_name='demo_trajectories (LEO)',
        line_color='red',
        marker_color='blue',
    )
    fig_geo = run_trajectory_demo(
        tle_data=get_sample_geo_tles(1),
        time_steps=np.arange(0, 24, 1),
        step_unit='hours',
        unit_label='hr',
        orbit_label='GEO',
        title='Single GEO Satellite Trajectory over 23 Hours',
        demo_name='demo_trajectories (GEO)',
        line_color='purple',
        marker_color='orange',
    )
    return fig_leo, fig_geo


def demo_constellation() -> go.Figure:
    """Demonstrates GEO constellation creation and propagation."""
    sim_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_time)
    add_observatories(sim_data, 0)
    add_celestial_bodies(sim_data)
    add_fixed_points(sim_data, 100)
    sim_check(sim_data)
    solar_system_ephemeris.set('jpl')

    sim_data = propagate_satellites(sim_data, sim_time)
    num_sats_before = sim_data.counts.satellites
    geos(sim_data, 10, 0.1)

    time_t1 = sim_time + timedelta(hours=1)
    sim_data.time = time_t1
    sim_data = propagate_satellites(sim_data, time_t1)
    sim_check(sim_data)

    fig = plot_3d_scatter(
        positions=sim_data.satellites.position[num_sats_before:],
        title=f"GEO Constellation at {time_t1.isoformat()} from demo_constellation",
        plot_time=time_t1,
        marker_size=5,
        trace_name='GEO Satellites',
    )
    return tag_demo_figure(fig, "demo_constellation")


def demo_observatories_only() -> tuple:
    """Demonstrates 5 ground observatories with 23 satellites across 5 time steps."""
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_celestial_bodies(sim_data)

    latitudes = np.array([19.8206, -24.6272, 28.7636, -31.2755, 31.9583])
    longitudes = np.array([-155.4681, -70.4042, -17.8947, 149.0673, -111.5967])
    altitudes = np.array([4205.0, 2635.0, 2396.0, 1165.0, 2096.0])
    labels = ["Mauna Kea", "Paranal", "La Palma", "Siding Spring", "Kitt Peak"]

    print("\n--- Generating 3D plots for Demo Observatories Only ---")
    add_observatories(
        sim_data=sim_data, num_observatories=5,
        latitudes=latitudes, longitudes=longitudes, altitudes=altitudes,
    )
    add_satellites_from_tle(sim_data, STANDARD_TLE_DATA, 'satellites')

    num_obs = sim_data.counts.observatories
    num_sats = sim_data.counts.satellites
    sim_data.observatories.detector = makeDetector(num_obs, band="V", fov=10 * DEGREE, ifov=3 * ARCSEC, aper=1.0)
    sim_data.satellites.detector = makeDetector(num_sats, band="V", fov=10 * DEGREE, ifov=3 * ARCSEC, aper=1.0)
    detectorPointingInitialize(sim_data, grid_points=400)

    figs = []
    current_time = sim_start_time
    for step in range(5):
        propagate_observatories(sim_data, current_time)
        propagate_satellites(sim_data, current_time)
        celestial_update(sim_data, current_time)
        update_detector_pointing(sim_data, sat_category='observatories')
        update_detector_pointing(sim_data, sat_category='satellites')

        all_positions = sim_data.get_detector_positions()
        all_labels = [f"Sat {i}" for i in range(num_sats)] + labels
        fig = plot_3d_scatter(
            positions=all_positions,
            title=f"Step {step+1}: Observatories and Satellites at {current_time.isoformat()}",
            plot_time=current_time,
            labels=all_labels,
            marker_size=4,
            trace_name='Assets',
            marker_color='blue',
        )
        add_vector_traces(fig, all_positions, sim_data.detector.pointing,
                          scale=0.5 * EARTH_RADIUS, color='red', width=4)
        tag_demo_figure(fig, "demo_observatories_only")
        figs.append(fig)

        if step == 0:
            print_simulation_state(sim_data)
        current_time += timedelta(hours=2)

    return tuple(figs)


def run_observatory_animation() -> go.Figure:
    """Generates a 24-hour 3D animation of a ground observatory tracking 3 targets."""
    print("=== Generating Ground Observatory 3D Animation ===")
    start_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time, delta_time=3600.0)
    add_observatories(
        sim_data, num_observatories=1,
        latitudes=np.array([34.38]), longitudes=np.array([-117.68]), altitudes=np.array([2286.0]),
    )
    add_celestial_bodies(sim_data)

    d = makeDetector(1, band='V', fov=np.radians(15), ifov=np.radians(0.15), aper=1.0)
    d.earthEx = np.array([np.radians(10.0)])
    sim_data.observatories.detector = d

    propagate_observatories(sim_data, start_time)
    obs_pos_init = sim_data.observatories.position[0]
    target_geo = (obs_pos_init / np.linalg.norm(obs_pos_init)) * 42164000.0
    target_rising = np.array([42164000.0, 0.0, 0.0])
    targets = np.vstack([target_geo, target_rising, -target_geo])
    target_names = ["GEO (Zenith Lock)", "Inertial Star (Rises/Sets)", "Hidden Target (Behind Earth)"]
    scale = 1e6

    frames = []
    for step in range(24):
        current_time = start_time + timedelta(hours=step)
        celestial_update(sim_data, current_time)
        propagate_observatories(sim_data, current_time)
        obs_pos = sim_data.observatories.position[0]

        lines_x, lines_y, lines_z, colors = [], [], [], []
        for targ in targets:
            pointing_vec = targ - obs_pos
            sim_data.observatories.detector.pointing[0] = pointing_vec / np.linalg.norm(pointing_vec)
            colors.append("red" if exclusion(sim_data, 0) else "green")
            lines_x.extend([obs_pos[0] / scale, targ[0] / scale, None])
            lines_y.extend([obs_pos[1] / scale, targ[1] / scale, None])
            lines_z.extend([obs_pos[2] / scale, targ[2] / scale, None])

        frames.append({
            'obs_pos': obs_pos / scale,
            'lines_x': lines_x,
            'lines_y': lines_y,
            'lines_z': lines_z,
            'colors': colors,
            'time': current_time.strftime("%Y-%m-%d %H:%M UTC"),
        })

    fig = go.Figure()
    add_sphere_trace(fig, radius=EARTH_RADIUS / scale, name='Earth', opacity=0.3, resolution=50)
    fig.add_trace(go.Scatter3d(
        x=targets[:, 0] / scale, y=targets[:, 1] / scale, z=targets[:, 2] / scale,
        mode='markers+text', marker=dict(size=8, color='gold', symbol='diamond'),
        text=target_names, textposition="top center", name='Targets',
    ))
    fig.add_trace(go.Scatter3d(
        x=[frames[0]['obs_pos'][0]], y=[frames[0]['obs_pos'][1]], z=[frames[0]['obs_pos'][2]],
        mode='markers+text', marker=dict(size=10, color='darkblue', symbol='circle'),
        text=['Table Mountain Observatory'], textposition="bottom center", name='Observatory',
    ))
    for i in range(3):
        c = frames[0]['colors'][i]
        fig.add_trace(go.Scatter3d(
            x=frames[0]['lines_x'][i*3:i*3+2],
            y=frames[0]['lines_y'][i*3:i*3+2],
            z=frames[0]['lines_z'][i*3:i*3+2],
            mode='lines',
            line=dict(color=c, width=3, dash='dash' if c == 'red' else 'solid'),
            name=f'LOS to {target_names[i]}',
        ))

    plotly_frames = []
    for f in frames:
        frame_data = [
            fig.data[0],
            fig.data[1],
            go.Scatter3d(
                x=[f['obs_pos'][0]], y=[f['obs_pos'][1]], z=[f['obs_pos'][2]],
                mode='markers+text', marker=dict(size=10, color='darkblue'),
                text=['Table Mountain Observatory'], textposition="bottom center",
            ),
        ]
        for i in range(3):
            c = f['colors'][i]
            frame_data.append(go.Scatter3d(
                x=f['lines_x'][i*3:i*3+2],
                y=f['lines_y'][i*3:i*3+2],
                z=f['lines_z'][i*3:i*3+2],
                mode='lines',
                line=dict(color=c, width=3, dash='dash' if c == 'red' else 'solid'),
            ))
        plotly_frames.append(go.Frame(data=frame_data, name=f['time']))

    fig.frames = plotly_frames
    fig.update_layout(
        title="24-Hour Ground Observatory Tracking Movie (GCRS Frame)",
        scene=dict(
            xaxis=dict(title='X (1000s km)', range=[-50, 50]),
            yaxis=dict(title='Y (1000s km)', range=[-50, 50]),
            zaxis=dict(title='Z (1000s km)', range=[-50, 50]),
            aspectmode='cube',
        ),
        width=1000, height=800,
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
            direction="left", pad={"r": 10, "t": 87}, showactive=False,
            x=0.1, xanchor="right", y=0, yanchor="top",
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[f['time']], dict(mode="immediate", frame=dict(duration=300, redraw=True),
                                        transition=dict(duration=0))],
                label=f['time'],
            ) for f in frames],
            transition=dict(duration=0), x=0.1, y=0, len=0.9,
            currentvalue=dict(font=dict(size=12), prefix="Epoch: ", visible=True, xanchor="right"),
        )],
    )
    return tag_demo_figure(fig, "run_observatory_animation")


# --- Pointing & Scanning Demos ---

def demo_pointing_plot() -> go.Figure:
    """Demonstrates plotting satellites with radially outward pointing vectors."""
    print("\n--- Starting Demo: Pointing Vector Plot ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)
    fig = plot_pointing_vectors(
        data_struct=sim_data,
        title="Satellite Positions with Radially Outward Pointing Vectors",
        plot_time=sim_start_time,
    )
    return tag_demo_figure(fig, "demo_pointing_plot")


def demo_sky_scan() -> go.Figure:
    """Performs a sky scan from a GEO satellite to map celestial exclusion zones."""
    print("\n--- Starting Demo: Sky Scan from GEO ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_satellites_from_tle(sim_data, get_sample_geo_tles(1), 'satellites')
    add_celestial_bodies(sim_data)
    set_exclusion_deg(sim_data, 30, 15, 10)
    sim_data = propagate_satellites(sim_data, sim_start_time)
    sim_data = celestial_update(sim_data, sim_start_time)

    declinations = np.linspace(-np.pi / 2, np.pi / 2, 50)
    right_ascensions = np.linspace(0, 2 * np.pi, 100)
    sky_map = np.zeros((len(declinations), len(right_ascensions)))

    for i, dec in enumerate(declinations):
        for j, ra in enumerate(right_ascensions):
            sim_data.detector.pointing[0] = np.array([
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec),
            ])
            sky_map[i, j] = exclusion(sim_data, satellite_index=0)

    fig = go.Figure(data=go.Heatmap(
        z=sky_map, x=np.rad2deg(right_ascensions), y=np.rad2deg(declinations),
        colorscale=[[0, 'lightgreen'], [1, 'red']], showscale=False,
    ))
    fig.update_layout(
        title='Sky Exclusion Map from a GEO Satellite',
        xaxis_title='Right Ascension (degrees)',
        yaxis_title='Declination (degrees)',
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return tag_demo_figure(fig, "demo_sky_scan")


def demo_pointing_sequence() -> go.Figure:
    """Demonstrates satellite pointing sequence progression over 30 time steps."""
    print("\n--- Starting Demo: Pointing Sequence ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_celestial_bodies(sim_data)
    add_satellites_from_tle(sim_data, get_sample_geo_tles(3), 'satellites')

    generate_pointing_sphere(sim_data, 100)
    generate_pointing_sphere(sim_data, 20)
    sim_data.satellites.detector.pointing_state[:, POINTING_COUNT_IDX] = [100, 20, 0]

    n_steps = 30
    trajectories = np.zeros((n_steps, 3, 3))
    for t in range(n_steps):
        if t > 0:
            current_time = sim_start_time + timedelta(seconds=t * sim_data.delta_time)
            sim_data = celestial_update(sim_data, current_time)
        update_detector_pointing(sim_data)
        trajectories[t] = sim_data.satellites.detector.pointing.copy()

    fig = go.Figure(layout=go.Layout(width=1000, height=800))
    colors = ['red', 'green', 'blue']
    sat_names = [
        'Satellite 1 (100 steps)',
        'Satellite 2 (20 steps)',
        'Satellite 3 (0 steps - Fixed)',
    ]
    time_steps = list(range(n_steps))
    for i in range(3):
        fig.add_trace(go.Scatter3d(
            x=trajectories[:, i, 0], y=trajectories[:, i, 1], z=trajectories[:, i, 2],
            mode='lines+markers',
            marker=dict(size=[(j + 2) * 2 for j in time_steps], color=time_steps,
                        colorscale='Viridis', showscale=False, opacity=0.8),
            line=dict(color=colors[i], width=2),
            name=sat_names[i],
        ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode='markers',
        marker=dict(colorscale='Viridis', showscale=True, cmin=0, cmax=29,
                    colorbar=dict(title='Time Step', tickvals=time_steps,
                                  ticktext=[f'T={t}' for t in time_steps])),
        hoverinfo='none', name='', showlegend=False,
    ))
    add_sphere_trace(fig, radius=1.0, name='Unit Sphere', opacity=0.1)
    apply_3d_scene_layout(fig, "Satellite Pointing Sequence Over 30 Time Steps", axis_unit='')
    return tag_demo_figure(fig, "demo_pointing_sequence")


def demo_show_geo_search() -> tuple:
    """Visualizes pointing updates and RA/Dec history across a GEO constellation."""
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_celestial_bodies(sim_data)
    geos(sim_data, 12, 0.4)
    sim_data = propagate_satellites(sim_data, sim_start_time)

    update_detector_pointing(sim_data)
    fig1 = plot_pointing_vectors(sim_data, 'Initial Pointing Vectors', sim_start_time)

    ra_history, dec_history = [], []

    def record_ra_dec():
        p = sim_data.detector.pointing[0]
        p_norm = p / np.linalg.norm(p)
        ra_history.append(np.rad2deg(np.arctan2(p_norm[1], p_norm[0])))
        dec_history.append(np.rad2deg(np.arcsin(p_norm[2])))

    record_ra_dec()
    snapshots = {}
    for step in range(1, 101):
        update_detector_pointing(sim_data)
        record_ra_dec()
        if step in (5, 15):
            snapshots[step] = plot_pointing_vectors(sim_data, f'After {step} Updates', sim_start_time)

    fig2, fig3 = snapshots[5], snapshots[15]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=ra_history, mode='lines', name='RA (deg)'))
    fig4.add_trace(go.Scatter(y=dec_history, mode='lines', name='Dec (deg)'))
    fig4.update_layout(
        title='RA and Dec History of a GEO Satellite',
        xaxis_title='Update Number', yaxis_title='Angle (degrees)',
    )
    for f in (fig1, fig2, fig3, fig4):
        tag_demo_figure(f, "demo_show_geo_search")
    return fig1, fig2, fig3, fig4


def demo_exclusion_pointing() -> go.Figure:
    """Demonstrates satellite pointing with a solar exclusion angle on a unit sphere."""
    start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)
    add_celestial_bodies(sim_data)

    single_tle_content = """GEO-SAT
1 99999U 99999A   25001.00000000  .00000000  00000-0  00000-0 0  9999
2 99999   0.0001 254.9961 0000001  98.4322 261.6813  1.00271900    10"""
    add_satellites_from_tle(sim_data, single_tle_content, 'satellites')

    sim_data.satellites.detector.solarEx[0] = math.pi / 2
    sim_data.satellites.detector.fov[0] = math.pi / 5

    n_points_sphere = 400
    generate_pointing_sphere(sim_data, n_points_sphere)
    sim_data.satellites.detector.pointing_state[0, POINTING_COUNT_IDX] = n_points_sphere
    sim_data.satellites.detector.pointing_state[0, POINTING_PLACE_IDX] = 0

    sim_data = celestial_update(sim_data, start_time)
    sim_data = propagate_satellites(sim_data, start_time, 'satellites')
    initial_sat_pos = sim_data.satellites.position[0]
    initial_sat_direction = initial_sat_pos / np.linalg.norm(initial_sat_pos)

    pointed_directions_history = []
    for _ in range(400):
        update_detector_pointing(sim_data, debug=False)
        pointed_directions_history.append(sim_data.detector.pointing[0].copy())
    pointed_directions_history = np.array(pointed_directions_history)

    fig = go.Figure()
    add_sphere_trace(fig, radius=1.0, name='Unit Sphere', opacity=0.1, resolution=50)
    fig.add_trace(go.Scatter3d(
        x=[initial_sat_direction[0]], y=[initial_sat_direction[1]], z=[initial_sat_direction[2]],
        mode='markers', marker=dict(size=5, color='blue'), name='Initial Satellite Pointing',
    ))
    fig.add_trace(go.Scatter3d(
        x=pointed_directions_history[:, 0], y=pointed_directions_history[:, 1],
        z=pointed_directions_history[:, 2], mode='markers',
        marker=dict(size=3, color=np.arange(len(pointed_directions_history)),
                    colorscale='Plasma', opacity=0.7),
        name='Pointing History',
    ))
    fig.add_trace(go.Scatter3d(
        x=pointed_directions_history[:, 0], y=pointed_directions_history[:, 1],
        z=pointed_directions_history[:, 2], mode='lines',
        line=dict(color='grey', width=1), name='Pointing Path',
    ))
    apply_3d_scene_layout(fig, "Detector Pointing with Exclusion", axis_unit=' (Unit Vector)')
    return tag_demo_figure(fig, "demo_exclusion_pointing")


# --- Radiometry & Target Exclusion Demos ---

def demo_fixedpoints() -> go.Figure:
    """Demonstrates the fixedpoints data structure by plotting it in 3D."""
    print("\n--- Starting Demo Fixedpoints ---")
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_fixed_points(sim_data, 100)
    fixed_positions = sim_data.fixedpoints.position
    fig = plot_3d_scatter(
        positions=fixed_positions,
        title="Fixed Points Distribution",
        plot_time=sim_start_time,
        labels=[f"Point {i}" for i in range(len(fixed_positions))],
        marker_size=3,
        trace_name='Fixed Points',
    )
    return tag_demo_figure(fig, "demo_fixedpoints")


def demo_exclusion_table() -> go.Figure:
    """Demonstrates generating and visualizing the satellite-target exclusion matrix."""
    print("\n--- Starting Demo: Exclusion Table Visualization ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)
    set_exclusion_deg(sim_data, 30, 30, 10)
    sim_data = celestial_update(sim_data, sim_start_time)

    original_fixed_points = sim_data.fixedpoints.position
    sim_data.fixedpoints.position = original_fixed_points[:200]
    update_exclusion_table(sim_data)
    exclusion_matrix = sim_data.fixedpoints.exclusion
    sim_data.fixedpoints.position = original_fixed_points

    fig = go.Figure(data=go.Heatmap(
        z=exclusion_matrix.T, colorscale=[[0, 'green'], [1, 'red']], showscale=False,
    ))
    fig.update_layout(
        title='Satellite vs. Fixed Point Exclusion Status (0=Clear, 1=Excluded)',
        xaxis_title='Fixed Point Index',
        yaxis_title='Satellite Index',
        yaxis=dict(autorange='reversed'),
    )
    return tag_demo_figure(fig, "demo_exclusion_table")


def demo_lambertian() -> go.Figure:
    """Demonstrates the lambertiansphere brightness model across phase angles."""
    SATELLITE_ALBEDO = 0.2
    SATELLITE_RADIUS = 1.5
    BASE_BRIGHTNESS = 1361
    OBSERVER_DISTANCE = 1000e3

    vec_sun = np.array([1, 0, 0])
    examples = [
        ("Example 1: Full Illumination", [1, 0, 0]),
        ("Example 2: Half Illumination", [0, 1, 0]),
        ("Example 3: No Illumination", [-1, 0, 0]),
    ]
    for title, direction in examples:
        print(f"--- {title} ---")
        vec_obs = np.array(direction) * OBSERVER_DISTANCE
        angle_rad = includedAngle(np.array([vec_sun]), np.array([vec_obs]))
        norm_obs = np.linalg.norm(np.array([vec_obs]), axis=1)
        emitted = lambertiansphere(
            angle_rad, np.array([SATELLITE_ALBEDO]),
            np.array([SATELLITE_RADIUS]), np.array([BASE_BRIGHTNESS]),
        )
        brightness = emitted / (np.pi * norm_obs ** 2)
        phase_deg = np.rad2deg(np.arccos(np.dot(vec_sun, vec_obs / np.linalg.norm(vec_obs))))
        print(f"Phase Angle: {phase_deg:.2f} degrees | Brightness: {brightness[0]:.4e} W/m^2")

    angles_deg = np.linspace(0, 180, 200)
    angles_rad = np.deg2rad(angles_deg)
    num_points = len(angles_rad)
    plot_vec_obs = np.zeros((num_points, 3))
    plot_vec_obs[:, 0] = np.cos(angles_rad) * OBSERVER_DISTANCE
    plot_vec_obs[:, 1] = np.sin(angles_rad) * OBSERVER_DISTANCE

    angle_plot = includedAngle(np.tile(vec_sun, (num_points, 1)), plot_vec_obs)
    emitted_values = lambertiansphere(
        angle_plot,
        np.full(num_points, SATELLITE_ALBEDO),
        np.full(num_points, SATELLITE_RADIUS),
        np.full(num_points, BASE_BRIGHTNESS),
    )
    brightness_values = emitted_values / (np.pi * np.linalg.norm(plot_vec_obs, axis=1) ** 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=angles_deg, y=brightness_values, mode='lines', name='Apparent Brightness'))
    fig.update_layout(
        title_text=(
            f'Lambertian Sphere Apparent Brightness vs. Phase Angle<br>'
            f'<sup>Albedo={SATELLITE_ALBEDO}, Radius={SATELLITE_RADIUS}m, '
            f'Distance={OBSERVER_DISTANCE / 1e3}km</sup>'
        ),
        xaxis_title="Physical Phase Angle (degrees)",
        yaxis_title="Apparent Brightness (W/m^2)",
        template="plotly_white",
        xaxis=dict(range=[0, 180]),
        yaxis_type="log",
        yaxis=dict(autorange=True),
    )
    return tag_demo_figure(fig, "demo_lambertian")


def demo_requiredIntegrationTime() -> go.Figure:
    """Demonstrates requiredIntegrationTime across a range of limiting magnitudes."""
    det = makeDetector(1, 'V', 30 * DEGREE, 1 * ARCSEC, 1, qe=1.0, photfrac=1.0)
    limiting_mags = np.arange(15, 26)
    integration_times = [requiredIntegrationTime(mag, 10, det) for mag in limiting_mags]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=limiting_mags, y=np.array(integration_times).flatten(), mode='lines+markers'))
    fig.update_layout(
        title="Required Integration Time vs. Limiting Magnitude",
        xaxis_title="Limiting Magnitude",
        yaxis_title="Integration Time (s)",
    )
    fig.update_yaxes(type="log")
    return tag_demo_figure(fig, "demo_requiredIntegrationTime")


def demo_vector_resorting_plot() -> go.Figure:
    """Runs test_vector_resorting and returns its figure."""
    print("\n--- Starting Demo: Vector Resorting ---")
    return test_vector_resorting()


def demoFixed() -> tuple:
    """Demonstrates fixedSat, fixedTarget, fixSun, and scandetectors radiometry."""
    start_time = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)

    fixSun(sim_data)
    fixedSat(sim_data, x=100, y=0, z=0, fov=180 * DEGREE)
    targ_rng = 100_000_000
    fixedTarget(sim_data, size=1.0, x=targ_rng, y=0, z=0)
    fixedTarget(sim_data, size=2.0, x=targ_rng, y=0, z=0)
    fixedTarget(sim_data, size=1.0, x=targ_rng * 10, y=0, z=0)
    fixedTarget(sim_data, size=1.0, x=0, y=targ_rng, z=0)

    def log_scale_pos(pos):
        r = np.linalg.norm(pos)
        return pos * (np.log(r) / r) if r > 0 else pos

    sun_log = log_scale_pos(sim_data.celestial.position[0])
    sat_log = log_scale_pos(sim_data.satellites.position[0])
    targets_log = np.array([log_scale_pos(p) for p in sim_data.fixedpoints.position])
    viewing_vec = sim_data.satellites.detector.pointing[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color='blue'), name='Earth'))
    fig.add_trace(go.Scatter3d(x=[sun_log[0]], y=[sun_log[1]], z=[sun_log[2]], mode='markers', marker=dict(size=8, color='yellow'), name='Sun'))
    fig.add_trace(go.Scatter3d(x=[sat_log[0]], y=[sat_log[1]], z=[sat_log[2]], mode='markers', marker=dict(size=5, color='red'), name='Satellite'))
    fig.add_trace(go.Scatter3d(
        x=targets_log[:, 0], y=targets_log[:, 1], z=targets_log[:, 2],
        mode='markers', marker=dict(size=3, color='green'), name='Targets',
    ))
    add_vector_traces(
        fig, sat_log, viewing_vec, scale=np.linalg.norm(sat_log) * 0.3,
        color='red', width=2, name='Viewing Vector', normalize=False,
    )
    apply_3d_scene_layout(fig, 'Demonstration of Fixed Objects (Log Scale Distance)', axis_unit=' (log-scaled)')

    scan_output = scandetectors(sim_data, print_output=1)
    print(f"Output of scandetector: {scan_output}")
    return tag_demo_figure(fig, "demoFixed"), sim_data


def demo_gap_time_histogram() -> go.Figure:
    """Runs a 50-step cadence simulation and plots pooled target interobservation gap times."""
    print("\n--- Starting Demo: Gap Time Histogram ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_satellites_from_tle(sim_data, get_sample_geo_tles(4), 'satellites')
    num_sats = sim_data.counts.satellites

    sim_data.satellites.detector = makeDetector(
        num_sats, band="V", fov=np.radians(45),
        ifov=3 * ARCSEC, aper=np.sqrt(4.0 / np.pi), qe=0.8,
    )
    sim_data.satellites.detector.integrationTime[:] = 10.0

    add_fixed_points(sim_data, num_points=20)
    add_celestial_bodies(sim_data)
    celestial_update(sim_data)
    detectorPointingInitialize(sim_data, 100)
    initCadence(sim_data)

    handler = DataHandler()
    for _ in range(50):
        update_detector_pointing(sim_data)
        handler.add_results(nextIntegration(sim_data, print_output=0))

    fig = handler.plot_gap_times_histogram(show_plot=False)
    return tag_demo_figure(fig, "demo_gap_time_histogram")


DEMO_GROUPS = {
    'orbits': [
        ('demo2', demo2),
        ('demo_trajectories', demo_trajectories),
        ('demo_constellation', demo_constellation),
        ('demo_observatories_only', demo_observatories_only),
        ('run_observatory_animation', run_observatory_animation),
        ('demo_cislunar', demo_cislunar),
    ],
    'pointing': [
        ('demo_pointing_plot', demo_pointing_plot),
        ('demo_sky_scan', demo_sky_scan),
        ('demo_pointing_sequence', demo_pointing_sequence),
        ('demo_show_geo_search', demo_show_geo_search),
        ('demo_exclusion_pointing', demo_exclusion_pointing),
    ],
    'radiometry': [
        ('demo_fixedpoints', demo_fixedpoints),
        ('demo_exclusion_table', demo_exclusion_table),
        ('demo_lambertian', demo_lambertian),
        ('demo_requiredIntegrationTime', demo_requiredIntegrationTime),
        ('demo_vector_resorting_plot', demo_vector_resorting_plot),
        ('demoFixed', demoFixed),
        ('demo_gap_time_histogram', demo_gap_time_histogram),
    ],
}


def list_demo_groups() -> None:
    """Lists available demo groups and their included demo functions."""
    print("=== VibeVolts Demo Groups ===")
    for group_name, demo_list in DEMO_GROUPS.items():
        print(f"\nGroup: '{group_name}' ({len(demo_list)} demos)")
        for name, _ in demo_list:
            print(f"  - {name}")
    print("\nUse run_all_demos(group='<name>') to run a specific group inline.")


def run_all_demos(
    group: str = 'all',
    demos: Optional[List[Union[str, Callable]]] = None,
    save_html: bool = False,
) -> None:
    """Runs demo functions by suite group, specific list, or all demos."""
    funcs_to_run = []

    if demos is not None:
        all_name_map = {
            name: func
            for g_list in DEMO_GROUPS.values()
            for name, func in g_list
        }
        for item in demos:
            if callable(item):
                funcs_to_run.append((item.__name__, item))
            elif isinstance(item, str) and item in all_name_map:
                funcs_to_run.append((item, all_name_map[item]))
            else:
                print(f"Warning: Demo '{item}' not recognized. Skipping.")
    elif group in DEMO_GROUPS:
        funcs_to_run = DEMO_GROUPS[group]
    elif group == 'all':
        for g_list in DEMO_GROUPS.values():
            funcs_to_run.extend(g_list)
        if not save_html:
            print("Notice: Displaying all demos inline can exceed browser WebGL limits (16 max).")
            print("Defaulting to save_html=True. Set group='orbits', 'pointing', or 'radiometry' for inline display.")
            save_html = True
    else:
        print(f"Unknown group '{group}'. Available groups: {list(DEMO_GROUPS.keys())} or 'all'.")
        return

    figs = []
    print(f"--- Running Demos (Group: {group}, Count: {len(funcs_to_run)}) ---")
    for name, func in funcs_to_run:
        print(f"\n\033[91m--- Executing {name} ---\033[0m")
        res = func()
        if isinstance(res, go.Figure):
            figs.append(res)
        elif isinstance(res, tuple):
            figs.extend(item for item in res if isinstance(item, go.Figure))

    if save_html:
        filename = "all_demo_plots.html" if group == 'all' else f"demo_plots_{group}.html"
        with open(filename, "w") as f:
            f.write("<html><head><title>VibeVolts Demos</title></head><body>")
            f.write(f"<h1>VibeVolts Demo Plots ({group.capitalize()})</h1>")
            for i, fig in enumerate(figs):
                title = fig.layout.title.text if fig.layout.title else f"Plot {i+1}"
                f.write(f"<h2>{title}</h2>")
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("</body></html>")
        print(f"\n--- Demos complete. {len(figs)} plots saved to {filename} ---")
    else:
        print(f"\n--- Displaying {len(figs)} demo plots inline ---")
        for fig in figs:
            fig.show()


if __name__ == '__main__':
    run_all_demos()
