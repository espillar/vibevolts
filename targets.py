import numpy as np
# from datetime import datetime, timezone
from typing import Dict, Any
from generate_log_spherical_points import generate_log_spherical_points
from constants import *
from minimalsimulation import FixedPointsState

def add_fixed_points(
    sim_data: Any,
    num_points: int = 100,
    size: float = 1.0,
    innerRadius: float = 2000000,
    outerRadius: float = 2 * GEO_RADIUS
) -> None:
    """
    Adds a structure for fixed reference points in the GCRS frame.

    Args:
        sim_data: The main simulation data structure (SimulationState).
        num_points: The number of fixed points to generate.
        size: The diameter of each object in meters.
        innerRadius: The minimum radius in meters at which test points will be created.
        outerRadius: The maximum radius in meters at which test points will be created.
    """
    positions = generate_log_spherical_points(
        num_points=num_points,
        inner_radius=innerRadius,
        outer_radius=outerRadius
    )
    if not sim_data.fixedpoints:
        sim_data.fixedpoints = FixedPointsState()
    sim_data.fixedpoints.add_target(positions, size=size, albedo=0.2)
