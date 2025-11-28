import numpy as np
from typing import Dict, Any

from constants import POINTING_COUNT_IDX, POINTING_PLACE_IDX
from fibonacciSearch import pointing_vectors, resort_vectors_by_proximity
from exclusion import exclusion

def generate_pointing_sphere(data_struct: Dict[str, Any], n_points: int, debug: bool = False) -> None:
    """
    Generates a pointing sphere with n_points and stores it in the data_struct.
    These positions will be used by the update_satellte_pointing to
    point sensors incrementaly.
    The index is ['pointing_spheres'][n] 
    If a sphere with the same number of points already exists, this function does nothing.
    """
    if n_points not in data_struct['pointing_spheres']:
        print(f"Generating pointing sphere with {n_points} points...")
        data_struct['pointing_spheres'][n_points] = \
           resort_vectors_by_proximity(pointing_vectors(n_points))

    if debug:
        print(f"\n--- Debugging Pointing Sphere (n_points={n_points}) ---")
        generated_vectors = data_struct['pointing_spheres'][n_points]
        print(f"Total number of pointing vectors generated: {len(generated_vectors)}")

        num_to_show = min(5, len(generated_vectors))
        print(f"Showing first {num_to_show} pointing vectors and their norms:")
        print(f"{'Index':<7} {'Vector (x, y, z)':<35} {'Norm':<10}")
        print(f"{'-------':<7} {'-----------------------------------':<35} {'----------':<10}")
        for i in range(num_to_show):
            vec = generated_vectors[i]
            norm = np.linalg.norm(vec)
            print(f"{i:<7} {str(vec):<35} {norm:<10.6f}")
        print("-------------------------------------------\n")


def update_satellite_pointing(data_struct: Dict[str, Any]) -> None:
    """
    Updates the pointing vector for each satellite, skipping excluded pointing directions.
    """
    num_sats = data_struct['counts']['satellites']
    if num_sats == 0:
        return

    pointing_state = data_struct['satellites']['pointing_state']
    pointing_vectors_all = data_struct['satellites']['pointing']

    for i in range(num_sats):
        count = int(pointing_state[i, POINTING_COUNT_IDX])
        if count <= 0:
            continue

        if count not in data_struct['pointing_spheres']:
            raise ValueError(f"Pointing sphere for {count} points not generated.")

        grid = data_struct['pointing_spheres'][count]
        
        place = int(pointing_state[i, POINTING_PLACE_IDX])
        start_place = place

        while True:
            place += 1
            if place >= count:
                place = 0

            pointing_vectors_all[i] = grid[place]
            
            if exclusion(data_struct, i) == 0:
                pointing_state[i, POINTING_PLACE_IDX] = place
                break
            
            if place == start_place:
                print(f"Warning: Satellite {i} has all pointing vectors excluded.")
                pointing_state[i, POINTING_PLACE_IDX] = place
                break


def jerk(data_struct: Dict[str, Any], satellite_indices: np.ndarray) -> Dict[str, Any]:
    """
    Moves the pointing vector of specific satellites by 0.3 radians in a
    random direction.

    This function applies a random rotation to the satellites' pointing vectors.

    Args:
        data_struct: The main simulation data dictionary.
        satellite_indices: The indices of the satellites to modify.

    Returns:
        The modified data_struct with the updated pointing vectors.
    """
    if satellite_indices.size == 0:
        return data_struct

    p = data_struct['satellites']['pointing'][satellite_indices]
    p_norm = p / np.linalg.norm(p, axis=1)[:, np.newaxis]

    # Generate a random vector not parallel to p_norm
    r = np.random.randn(*p.shape)
    r -= np.sum(r * p_norm, axis=1)[:, np.newaxis] * p_norm
    k_hat = r / np.linalg.norm(r, axis=1)[:, np.newaxis]

    theta = 0.3
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rodrigues' rotation formula
    p_new = p_norm * cos_theta + np.cross(k_hat, p_norm) * sin_theta

    data_struct['satellites']['pointing'][satellite_indices] = p_new

    return data_struct



def find_and_jerk_blind_satellites(data_struct: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finds satellites with no exclusion and applies the 'jerk' function to them.

    This function finds satellites with no visible fixed points (i.e., the
    column sum in the exclusion table is 0) and calls the `jerk` function
    to randomly adjust their pointing vectors.

    Args:
        data_struct: The main simulation data dictionary.

    Returns:
        The modified data_struct.
    """
    exclusion_table = data_struct['fixedpoints']['exclusion']

    column_sums = np.sum(exclusion_table, axis=0)
    blind_satellite_indices = np.where(column_sums == 0)[0]

    if blind_satellite_indices.size > 0:
        print(f"Satellites {blind_satellite_indices} have no visible points. Applying jerk.")
        data_struct = jerk(data_struct, blind_satellite_indices)

    return data_struct
