from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import numpy as np
import plotly.graph_objects as go
from astropy.coordinates import solar_system_ephemeris
import detector
from constants import *


# Adjusting import paths for the new project structure
from propagation import add_satellites_from_tle, propagate_satellites
from minimalsimulation import create_empty_simulation
from celestialbodies import add_celestial_bodies
from targets import add_fixed_points
from observatories import add_observatories
from plotting_3d import add_sphere_trace, apply_3d_scene_layout, tag_demo_figure


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


def initialize_standard_simulation(start_time=None) -> Dict[str, Any]:
    """
    Initializes a standard simulation with a predefined set of satellites.

    This function consolidates TLE data, initializes the main data structure,
    and propagates satellites to their initial positions at the specified start
    time. This ensures the returned simulation state is fully populated with
    positions and radially-outward pointing vectors.

    Args:
        start_time: The timezone-aware datetime object for the simulation start.

    Returns:
        The fully initialized and propagated simulation data dictionary.
    """
    if start_time is None:
        start_time = datetime.now(tz=timezone.utc)

    sim_data = create_empty_simulation(start_time)

    add_satellites_from_tle(sim_data, STANDARD_TLE_DATA, 'satellites')
    add_observatories(sim_data, 0)
    add_celestial_bodies(sim_data)
    add_fixed_points(sim_data, 100)
    num_sats = sim_data.counts.satellites
    sat_det = detector.makeDetector(
        num_sats, "V", 10 * DEGREE, 3 * ARCSEC, 1
    )
    if sim_data.satellites is not None:
        sim_data.satellites.detector = sat_det
    
    
    print(f"Initializing standard simulation with {sim_data.counts.satellites} satellites.")

    solar_system_ephemeris.set('jpl')

    print(f"Propagating satellites to start time: {start_time.isoformat()} to set initial state.")
    sim_data = propagate_satellites(sim_data, start_time)

    return sim_data
