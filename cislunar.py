import numpy as np
from datetime import datetime, timezone
from typing import Any, Optional, Tuple
from scipy.integrate import solve_ivp
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import plotly.graph_objects as go

from constants import EARTH_RADIUS, MOON_ORBIT_RADIUS, MOON_RADIUS
from plotting_3d import add_sphere_trace, apply_3d_scene_layout, tag_demo_figure


# Earth-Moon Circular Restricted Three-Body Problem (CR3BP) constants
MU_EARTH_MOON = 0.01215058560962404
L_STAR = MOON_ORBIT_RADIUS  # 384,400,000 m
GM_EARTH_MOON = 3.986004418e14 + 4.902800066e12  # m^3 / s^2
T_STAR = np.sqrt(L_STAR**3 / GM_EARTH_MOON)      # ~375,190 s (~4.34 days)

# Periodic Earth-Moon CR3BP orbit catalog [x, y, z, vx, vy, vz] (dimensionless)
SAMPLE_L1_HALO_STATE = np.array([
    0.823385182067467, 0.0, 0.022092729457745812,
    0.0, 0.13406042066407478, 0.0,
])
SAMPLE_L1_HALO_PERIOD_ND = 2.7462832889478705

CISLUNAR_ORBITS = {
    'L1 Halo': {
        'state': SAMPLE_L1_HALO_STATE,
        'period_nd': SAMPLE_L1_HALO_PERIOD_ND,
        'color': 'cyan',
    },
    'L2 Southern NRHO': {
        'state': np.array([
            1.0218813357840564, 0.0, -0.182,
            0.0, -0.10295079995193875, 0.0,
        ]),
        'period_nd': 1.509263542700784,
        'color': 'magenta',
    },
    'Distant Retrograde (DRO)': {
        'state': np.array([
            0.8, 0.0, 0.0,
            0.0, 0.5263593142960079, 0.0,
        ]),
        'period_nd': 3.317125713193684,
        'color': 'lime',
    },
    'L2 Lyapunov': {
        'state': np.array([
            1.12, 0.0, 0.0,
            0.0, 0.17787385503274158, 0.0,
        ]),
        'period_nd': 3.4164332828883124,
        'color': 'orange',
    },
}


def cr3bp_eom(_t: float, state: np.ndarray, mu: float = MU_EARTH_MOON) -> np.ndarray:
    """Dimensionless CR3BP equations of motion in the rotating Earth-Moon frame."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    r1_3, r2_3 = r1 ** 3, r2 ** 3

    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / r1_3 - mu * (x - 1.0 + mu) / r2_3
    ay = -2.0 * vx + y - (1.0 - mu) * y / r1_3 - mu * y / r2_3
    az = -(1.0 - mu) * z / r1_3 - mu * z / r2_3
    return np.array([vx, vy, vz, ax, ay, az], dtype=float)


def jacobi_constant(state: np.ndarray, mu: float = MU_EARTH_MOON) -> float:
    """Computes the CR3BP Jacobi constant C = 2*Omega - v^2 for a rotating state."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y ** 2 + z ** 2)
    omega = 0.5 * (x ** 2 + y ** 2) + (1.0 - mu) / r1 + mu / r2
    return float(2.0 * omega - (vx ** 2 + vy ** 2 + vz ** 2))


def propagate_cr3bp(
    state0: np.ndarray,
    dt_seconds: float,
    mu: float = MU_EARTH_MOON,
) -> np.ndarray:
    """Propagates a 6-element CR3BP rotating state by dt_seconds."""
    dt_nd = dt_seconds / T_STAR
    if abs(dt_nd) < 1e-15:
        return np.array(state0, dtype=float, copy=True)
    sol = solve_ivp(
        cr3bp_eom,
        (0.0, dt_nd),
        state0,
        args=(mu,),
        method='DOP853',
        rtol=1e-11,
        atol=1e-11,
    )
    return sol.y[:, -1]


def get_moon_gcrs_state(time_date: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the Moon's GCRS position (m) and velocity (m/s) at time_date."""
    astro_time = Time(time_date)
    moon_pos, moon_vel = get_body_barycentric_posvel('moon', astro_time)
    earth_pos, earth_vel = get_body_barycentric_posvel('earth', astro_time)
    r_gcrs = (moon_pos.xyz - earth_pos.xyz).to(u.m).value
    v_gcrs = (moon_vel.xyz - earth_vel.xyz).to(u.m / u.s).value
    return r_gcrs, v_gcrs


def emb_basis(
    moon_pos: np.ndarray,
    moon_vel: np.ndarray,
    mu: float = MU_EARTH_MOON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Computes the instantaneous Earth-Moon rotating frame in GCRS.

    Returns:
        (origin_emb_m, rot_matrix, omega_rad_s, moon_dist_m)
    """
    d = float(np.linalg.norm(moon_pos))
    x_hat = moon_pos / d
    h_vec = np.cross(moon_pos, moon_vel)
    z_hat = h_vec / np.linalg.norm(h_vec)
    y_hat = np.cross(z_hat, x_hat)
    rot_matrix = np.column_stack([x_hat, y_hat, z_hat])
    origin_emb = mu * moon_pos
    omega_vec = h_vec / (d ** 2)
    return origin_emb, rot_matrix, omega_vec, d


def emb_rotating_to_gcrs(
    pos_nd: np.ndarray,
    moon_pos: np.ndarray,
    moon_vel: np.ndarray,
    vel_nd: Optional[np.ndarray] = None,
    mu: float = MU_EARTH_MOON,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Converts dimensionless Earth-Moon rotating coordinates to GCRS (meters, m/s).
    Earth (-mu, 0, 0) maps to [0, 0, 0] and Moon (1-mu, 0, 0) maps to moon_pos.
    """
    origin_emb, rot_matrix, omega_vec, d = emb_basis(moon_pos, moon_vel, mu=mu)
    pos_arr = np.atleast_2d(np.asarray(pos_nd, dtype=float))
    rel_gcrs = (pos_arr * d) @ rot_matrix.T
    pos_gcrs = origin_emb + rel_gcrs
    if pos_nd.ndim == 1:
        pos_gcrs = pos_gcrs[0]

    if vel_nd is None:
        return pos_gcrs, None

    vel_arr = np.atleast_2d(np.asarray(vel_nd, dtype=float))
    v_scale = d / T_STAR
    vel_rot_gcrs = (vel_arr * v_scale) @ rot_matrix.T
    vel_gcrs = mu * moon_vel + vel_rot_gcrs + np.cross(omega_vec, rel_gcrs)
    if vel_nd.ndim == 1:
        vel_gcrs = vel_gcrs[0]
    return pos_gcrs, vel_gcrs


def gcrs_to_emb_rotating(
    pos_gcrs: np.ndarray,
    moon_pos: np.ndarray,
    moon_vel: np.ndarray,
    mu: float = MU_EARTH_MOON,
) -> np.ndarray:
    """Converts GCRS positions (meters) to dimensionless Earth-Moon rotating coordinates."""
    origin_emb, rot_matrix, _, d = emb_basis(moon_pos, moon_vel, mu=mu)
    pos_arr = np.atleast_2d(np.asarray(pos_gcrs, dtype=float))
    pos_nd = ((pos_arr - origin_emb) @ rot_matrix) / d
    return pos_nd[0] if np.asarray(pos_gcrs).ndim == 1 else pos_nd


def add_cislunar_satellites(
    sim_data: Any,
    states_nd: np.ndarray,
    epoch: Optional[datetime] = None,
    sat_category: str = 'cislunar',
) -> None:
    """
    Adds cislunar satellites defined by CR3BP rotating initial states (N, 6)
    and propagates them into GCRS on sim_data[sat_category].
    """
    from detector import makeBlankDetector
    from minimalsimulation import CislunarState

    states_arr = np.atleast_2d(np.asarray(states_nd, dtype=float))
    n_sats = states_arr.shape[0]
    ref_epoch = epoch or sim_data.time
    new_cislunar = CislunarState(
        position=np.zeros((n_sats, 3), dtype=float),
        velocity=np.zeros((n_sats, 3), dtype=float),
        cr3bp_states=states_arr.copy(),
        epochs=[ref_epoch] * n_sats,
        detector=makeBlankDetector(n_sats),
    )
    sim_data.add_component(sat_category, new_cislunar)
    propagate_cislunar(sim_data, sim_data.time, sat_category=sat_category)


def propagate_cislunar(
    sim_data: Any,
    time_date: Optional[datetime] = None,
    sat_category: str = 'cislunar',
) -> Any:
    """Propagates CR3BP satellites and stores their GCRS positions/velocities in sim_data."""
    comp = sim_data.get(sat_category)
    if comp is None or len(comp) == 0:
        return sim_data

    target_time = time_date or sim_data.time
    moon_pos, moon_vel = get_moon_gcrs_state(target_time)

    n_sats = len(comp.cr3bp_states)
    states_nd = np.zeros((n_sats, 6), dtype=float)
    for i in range(n_sats):
        dt_sec = (target_time - comp.epochs[i]).total_seconds()
        states_nd[i] = propagate_cr3bp(comp.cr3bp_states[i], dt_sec)

    pos_gcrs, vel_gcrs = emb_rotating_to_gcrs(
        states_nd[:, :3], moon_pos, moon_vel, vel_nd=states_nd[:, 3:]
    )
    comp.position = pos_gcrs
    comp.velocity = vel_gcrs
    return sim_data


def demo_cislunar() -> Tuple[go.Figure, go.Figure]:
    """
    Demonstrates four distinct Earth-Moon cislunar orbit families (L1 Halo,
    L2 Southern NRHO, Distant Retrograde Orbit, and L2 Lyapunov) propagated
    together in CR3BP and viewed in both EMB rotating and GCRS inertial frames.
    """
    from datetime import timedelta
    from minimalsimulation import create_empty_simulation

    orbit_names = list(CISLUNAR_ORBITS.keys())
    states0 = np.vstack([CISLUNAR_ORBITS[name]['state'] for name in orbit_names])
    n_orbits = len(orbit_names)

    start_time = datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)
    add_cislunar_satellites(sim_data, states0, epoch=start_time)

    n_steps = 4000
    max_period_sec = 5.0 * max(v['period_nd'] for v in CISLUNAR_ORBITS.values()) * T_STAR
    dt_grid = np.linspace(0.0, max_period_sec, n_steps)
    dt_nd = dt_grid / T_STAR

    rot_positions = np.zeros((n_steps, n_orbits, 3))
    gcrs_positions = np.zeros((n_steps, n_orbits, 3))

    times = Time([start_time + timedelta(seconds=float(s)) for s in dt_grid])
    moon_p, moon_v = get_body_barycentric_posvel('moon', times)
    earth_p, earth_v = get_body_barycentric_posvel('earth', times)
    moon_gcrs_positions = (moon_p.xyz - earth_p.xyz).to(u.m).value.T
    moon_gcrs_velocities = (moon_v.xyz - earth_v.xyz).to(u.m / u.s).value.T

    for j, s0 in enumerate(states0):
        sol = solve_ivp(
            cr3bp_eom,
            (0.0, float(dt_nd[-1])),
            s0,
            t_eval=dt_nd,
            args=(MU_EARTH_MOON,),
            method='DOP853',
            rtol=1e-11,
            atol=1e-11,
        )
        rot_positions[:, j, :] = sol.y[:3].T

    for i in range(n_steps):
        gcrs_positions[i], _ = emb_rotating_to_gcrs(
            rot_positions[i], moon_gcrs_positions[i], moon_gcrs_velocities[i]
        )

    # 1. Rotating Earth-Moon Barycentric (EMB) plot
    fig_rot = go.Figure()
    for j, name in enumerate(orbit_names):
        color = CISLUNAR_ORBITS[name]['color']
        fig_rot.add_trace(go.Scatter3d(
            x=rot_positions[:, j, 0],
            y=rot_positions[:, j, 1],
            z=rot_positions[:, j, 2],
            mode='lines',
            line=dict(color=color, width=4),
            name=name,
        ))
    fig_rot.add_trace(go.Scatter3d(
        x=[-MU_EARTH_MOON, 1.0 - MU_EARTH_MOON, 0.8369, 1.1557],
        y=[0.0, 0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0, 0.0],
        mode='markers+text',
        marker=dict(size=[10, 6, 5, 5], color=['blue', 'gray', 'yellow', 'red']),
        text=['Earth', 'Moon', 'L1', 'L2'],
        textposition='top center',
        name='Bodies & L1/L2',
    ))
    apply_3d_scene_layout(
        fig_rot, 'Cislunar Orbit Families in Earth-Moon Rotating Frame', axis_unit=' (L*)',
    )
    tag_demo_figure(fig_rot, 'demo_cislunar (EMB Rotating)')

    # 2. GCRS Inertial plot
    fig_gcrs = go.Figure()
    add_sphere_trace(fig_gcrs, radius=EARTH_RADIUS, name='Earth')
    for j, name in enumerate(orbit_names):
        color = CISLUNAR_ORBITS[name]['color']
        fig_gcrs.add_trace(go.Scatter3d(
            x=gcrs_positions[:, j, 0],
            y=gcrs_positions[:, j, 1],
            z=gcrs_positions[:, j, 2],
            mode='lines',
            line=dict(color=color, width=3),
            name=name,
        ))
    fig_gcrs.add_trace(go.Scatter3d(
        x=moon_gcrs_positions[:, 0], y=moon_gcrs_positions[:, 1], z=moon_gcrs_positions[:, 2],
        mode='lines', line=dict(color='gray', width=2, dash='dash'), name='Moon Orbit (GCRS)',
    ))
    apply_3d_scene_layout(fig_gcrs, 'Cislunar Orbit Families in GCRS Inertial Frame')
    tag_demo_figure(fig_gcrs, 'demo_cislunar (GCRS)')

    return fig_rot, fig_gcrs
