import numpy as np
from typing import Dict, Any

def jerk(data_struct: Dict[str, Any], satellite_number: int) -> Dict[str, Any]:
    """
    Moves the pointing vector of a specific satellite by 0.3 radians in a
    random direction.

    This function applies a random rotation to the satellite's pointing vector
    using a simplified version of Rodrigues' rotation formula.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_number: The index of the satellite to modify.

    Returns:
        The modified data_struct with the updated pointing vector.
    """
    p = data_struct['satellites']['pointing'][satellite_number]
    p_norm = p / np.linalg.norm(p)

    while True:
        r = np.random.rand(3) - 0.5
        k = np.cross(p_norm, r)
        norm_k = np.linalg.norm(k)
        if norm_k > 1e-9:
            k_hat = k / norm_k
            break

    theta = 0.3
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    p_new = p_norm * cos_theta + np.cross(k_hat, p_norm) * sin_theta

    data_struct['satellites']['pointing'][satellite_number] = p_new / np.linalg.norm(p_new)

    return data_struct

def find_and_jerk_blind_satellites(data_struct: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finds satellites with no visibility and applies the 'jerk' function to them.

    This function iterates through the visibility table. If any satellite (column)
    has no visible fixed points (i.e., the column sum is 0), the `jerk`
    function is called to randomly adjust its pointing vector.

    Args:
        data_struct: The main simulation data dictionary.

    Returns:
        The modified data_struct.
    """
    visibility_table = data_struct['fixedpoints']['visibility']

    column_sums = np.sum(visibility_table, axis=0)
    blind_satellite_indices = np.where(column_sums == 0)[0]

    for sat_index in blind_satellite_indices:
        print(f"Satellite {sat_index} has no visible points. Applying jerk.")
        data_struct = jerk(data_struct, sat_index)

    return data_struct
