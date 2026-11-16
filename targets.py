from typing import Any
import numpy as np
import astropy.units as u

from constants import ARCSEC, DEGREE, GEO_RADIUS
from detector import makeDetector
from minimalsimulation import FixedPointsState, SatellitesState
from pointing import pointing_vectors


def generate_log_spherical_points(
    num_points: int,
    inner_radius: float,
    outer_radius: float,
    seed: int = None,
) -> np.ndarray:
    """
    Generates 3D points with logarithmic radial spacing and uniform Fibonacci angular
    distribution.
    """
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points must be a positive integer.")
    if not (isinstance(inner_radius, (int, float)) and inner_radius > 0):
        raise ValueError("inner_radius must be a positive number.")
    if not (isinstance(outer_radius, (int, float)) and outer_radius >= inner_radius):
        raise ValueError("outer_radius must be a positive number and >= inner_radius.")

    unit_vectors = pointing_vectors(num_points)
    radii = np.logspace(np.log10(inner_radius), np.log10(outer_radius), num_points)
    np.random.default_rng(seed).shuffle(radii)
    return unit_vectors * radii[:, np.newaxis]


def add_fixed_points(
    sim_data: Any,
    num_points: int = 100,
    size: float = 1.0,
    innerRadius: float = 2000000,
    outerRadius: float = 2 * GEO_RADIUS,
) -> None:
    """Adds fixed reference targets in the GCRS frame."""
    positions = generate_log_spherical_points(
        num_points=num_points, inner_radius=innerRadius, outer_radius=outerRadius,
    )
    if not sim_data.fixedpoints:
        sim_data.fixedpoints = FixedPointsState()
    sim_data.fixedpoints.add_target(positions, size=size, albedo=0.2)


def fixedSat(sim_data: Any, x: float, y: float, z: float, fov: float = 10 * DEGREE) -> None:
    """Adds a stationary test satellite at (x, y, z) m with a V-band detector."""
    pos = np.array([[x, y, z]], dtype=float)
    norm = np.linalg.norm(pos)
    det = makeDetector(1, 'V', fov, 1 * ARCSEC, 1.0)
    det.pointing[0] = pos[0] / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

    new_sat = SatellitesState(
        position=pos,
        velocity=np.zeros((1, 3), dtype=float),
        orbital_elements=np.zeros((1, 6), dtype=float),
        epochs=[sim_data.time],
        detector=det,
    )
    sim_data.add_component('satellites', new_sat)


def fixedTarget(sim_data: Any, size: float, x: float, y: float, z: float) -> None:
    """Adds a stationary Lambertian sphere target at (x, y, z) m."""
    if not sim_data.fixedpoints:
        sim_data.fixedpoints = FixedPointsState()
    sim_data.fixedpoints.add_target(position=np.array([x, y, z]), size=size, albedo=0.2)


def fixSun(sim_data: Any) -> None:
    """Places the Sun statically at -1 AU along the GCRS X-axis."""
    from celestialbodies import add_celestial_bodies

    if not sim_data.celestial:
        add_celestial_bodies(sim_data)
    sim_data.celestial.position[0] = np.array([-1.0 * u.au.to(u.m), 0.0, 0.0])
