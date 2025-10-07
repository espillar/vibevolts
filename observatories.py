import numpy as np
from typing import Dict, Any

def add_observatories(sim_data: Dict[str, Any], num_observatories: int) -> None:
    """
    Adds observatory data structures to the simulation data.

    Args:
        sim_data: The main simulation data dictionary.
        num_observatories: The number of observatories to add.
    """
    if not isinstance(num_observatories, int) or num_observatories < 0:
        raise ValueError("num_observatories must be a non-negative integer.")

    sim_data['counts']['observatories'] = num_observatories
    sim_data['observatories'] = {
        'position': np.zeros((num_observatories, 3), dtype=float),
        'velocity': np.zeros((num_observatories, 3), dtype=float),
        'acceleration': np.zeros((num_observatories, 3), dtype=float),
        'pointing': np.zeros((num_observatories, 3), dtype=float),
        'detector': np.zeros((num_observatories, 9), dtype=float),
    }
