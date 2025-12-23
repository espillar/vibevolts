import numpy as np
# from datetime import datetime, timezone
from typing import Dict, Any
from generate_log_spherical_points import generate_log_spherical_points
from constants import *

def add_fixed_points(sim_data: Dict[str, Any], num_points: int = 100, size: float = 1.0) -> None:
    """
    Adds a structure for fixed reference points in the GCRS frame.

    Args:
        sim_data: The simulation data dictionary.
        num_points: The number of fixed points to generate.
        size: the size of the objects, 
    """
    sim_data['counts']['fixedpoints'] = num_points
    sim_data['fixedpoints'] = {
        'position': generate_log_spherical_points(
            num_points=num_points,
            inner_radius=2000000,
            outer_radius=84328000
        )[0],
        'exclusion': np.zeros(num_points, dtype=int), # Exclusion will be resized later,
        'size' : np.full( num_points, size) #
    }
