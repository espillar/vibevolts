import numpy as np
# from datetime import datetime, timezone
from typing import Dict, Any
from generate_log_spherical_points import generate_log_spherical_points
from constants import *

def add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0, innerRadius: float = 2000000, outerRadius: float = 2 * GEO_RADIUS) -> None:
    """
    Adds a structure for fixed reference points in the GCRS frame.

    Args:
        sim_data: The simulation data dictionary.
        num_points: The number of fixed points to generate.
        size: the size of the objects, 
        innerRadius: The minimum radius at which test points will be created.
        outerRadius: The maximum radius at which test points will be created.
    """
    positions = generate_log_spherical_points(
            num_points=num_points,
            inner_radius=innerRadius,
            outer_radius=outerRadius
        )
    sim_data['counts']['fixedpoints'] = num_points
    sim_data['fixedpoints'] = {
        'position': positions,        
        'exclusion': np.zeros(num_points, dtype=int), # Exclusion will be resized later,
        'size' : np.full(num_points, size, dtype=float ), 
        'albedo' : np.full(num_points, 0.2, dtype=float)
    }
