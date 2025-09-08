import numpy as np
from typing import Dict, Any

from constants import POINTING_COUNT_IDX, POINTING_PLACE_IDX
from pointing_vectors import pointing_vectors

def pointing_place_update(data_struct: Dict[str, Any]) -> None:
    """
    Increments the pointing place for all satellites, wrapping around if necessary.
    """
    pointing_state = data_struct['satellites']['pointing_state']
    pointing_counts = pointing_state[:, POINTING_COUNT_IDX]

    # Increment pointing place
    pointing_state[:, POINTING_PLACE_IDX] += 1

    # Wrap around where place >= count
    wrap_around_indices = np.where(pointing_state[:, POINTING_PLACE_IDX] >= pointing_counts)
    pointing_state[wrap_around_indices, POINTING_PLACE_IDX] = 0

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

def generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int) -> None:
    """
    Generates a pointing sphere with n_points and stores it in the data_struct.
    If a sphere with the same number of points already exists, this function does nothing.
    """
    if n_points not in data_struct['pointing_spheres']:
        print(f"Generating pointing sphere with {n_points} points...")
        data_struct['pointing_spheres'][n_points] = pointing_vectors(n_points)

def update_satellite_pointing(data_struct: Dict[str, Any]) -> None:
    """
    Updates the pointing vector for each satellite based on its pointing state.
    """
    num_sats = data_struct['counts']['satellites']
    if num_sats == 0:
        return

    pointing_state = data_struct['satellites']['pointing_state']
    pointing_vectors_all = data_struct['satellites']['pointing']

    for i in range(num_sats):
        count = pointing_state[i, POINTING_COUNT_IDX]
        place = pointing_state[i, POINTING_PLACE_IDX]

        if count > 0:
            if count not in data_struct['pointing_spheres']:
                raise ValueError(f"Pointing sphere for {count} points not generated.")

            grid = data_struct['pointing_spheres'][count]
            pointing_vectors_all[i] = grid[place]

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
