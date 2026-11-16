import numpy as np
from typing import Any
import plotly.graph_objects as go

from constants import POINTING_COUNT_IDX, POINTING_PLACE_IDX
from exclusion import check_sim_exclusions
from plotting_3d import add_sphere_trace, apply_3d_scene_layout, tag_demo_figure


def pointing_vectors(n: int) -> np.ndarray:
    """
    Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    indices = np.arange(0, n, dtype=float) + 0.5
    z = 1 - 2 * indices / n
    radius_xy = np.sqrt(1 - z**2)
    theta = np.pi * (3.0 - np.sqrt(5.0)) * indices
    return np.stack([radius_xy * np.cos(theta), radius_xy * np.sin(theta), z], axis=1)


def resort_vectors_by_proximity(unit_vectors: np.ndarray) -> np.ndarray:
    """
    Resorts a list of vectors by making each subsequent vector the closest one
    in the remaining set to the previous one.
    """
    if unit_vectors.ndim != 2 or unit_vectors.shape[1] != 3:
        raise ValueError("unit_vectors array must have shape (n, 3).")

    n_vectors = unit_vectors.shape[0]
    if n_vectors == 0:
        return np.array([])

    remaining_indices = list(range(n_vectors))
    sorted_vectors = np.zeros_like(unit_vectors)
    sorted_vectors[0] = unit_vectors[remaining_indices.pop(0)]

    for i in range(1, n_vectors):
        rem_coords = unit_vectors[remaining_indices]
        dists = np.linalg.norm(rem_coords - sorted_vectors[i - 1], axis=1)
        best_idx_in_rem = int(np.argmin(dists))
        sorted_vectors[i] = unit_vectors[remaining_indices.pop(best_idx_in_rem)]

    return sorted_vectors


def plot_vectors_on_sphere(vectors: np.ndarray, title: str) -> go.Figure:
    """Creates a 3D plot of vectors on a unit sphere."""
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors array must have shape (n, 3).")

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
        mode='markers', marker=dict(size=2, color='red', opacity=0.8),
        name='Vectors',
    ))
    add_sphere_trace(fig, radius=1.0, name='Sphere')
    apply_3d_scene_layout(fig, title, axis_unit='', legend_title='Objects')
    return fig


def test_vector_resorting() -> go.Figure:
    """Tests vector resorting and plots the Euclidean distance between subsequent vectors."""
    vectors = pointing_vectors(100)
    diffs_unsorted = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
    diffs_sorted = np.linalg.norm(np.diff(resort_vectors_by_proximity(vectors), axis=0), axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=diffs_unsorted, mode='lines', name='Unsorted'))
    fig.add_trace(go.Scatter(y=diffs_sorted, mode='lines', name='Resorted by Proximity'))
    fig.update_layout(
        title="Euclidean Distance Between Subsequent Vectors",
        xaxis_title="Vector Index",
        yaxis_title="Euclidean Distance",
    )
    return tag_demo_figure(fig, "test_vector_resorting")


def generate_pointing_sphere(sim_data: Any, n_points: int, debug: bool = False) -> None:
    """Generates and caches a proximity-sorted Fibonacci pointing sphere in sim_data."""
    if n_points not in sim_data.pointing_spheres:
        print(f"Generating pointing sphere with {n_points} points...")
        sim_data.pointing_spheres[n_points] = resort_vectors_by_proximity(
            pointing_vectors(n_points)
        )

    if debug:
        generated = sim_data.pointing_spheres[n_points]
        print(f"\n--- Debugging Pointing Sphere (n_points={n_points}) ---")
        for i, vec in enumerate(generated[:5]):
            print(f"{i:<7} {str(vec):<35} {np.linalg.norm(vec):<10.6f}")


def update_detector_pointing(
    sim_data: Any,
    sat_category: str = None,
    debug: bool = False,
) -> None:
    """
    Updates detector pointing vectors across active categories in vectorized batches,
    skipping excluded pointing directions.
    """
    by_count: dict = {}
    for _, _, det, global_offset in sim_data.iter_detector_components(sat_category):
        for i_local in range(len(det.filt)):
            count = int(det.pointing_state[i_local, POINTING_COUNT_IDX])
            if count > 0:
                by_count.setdefault(count, []).append((det, i_local, global_offset + i_local))

    for count, entries in by_count.items():
        grid = sim_data.pointing_spheres[count]
        global_idxs = [e[2] for e in entries]
        u_pointing = np.broadcast_to(grid[:, np.newaxis, :], (count, len(global_idxs), 3))
        excl_mask = np.any(check_sim_exclusions(sim_data, u_pointing, mask=global_idxs), axis=2)

        for k, (det, i_local, i_global) in enumerate(entries):
            place = int(det.pointing_state[i_local, POINTING_PLACE_IDX])
            order = (np.arange(1, count + 1) + place) % count
            clear = order[~excl_mask[order, k]]
            if len(clear) > 0:
                new_place = int(clear[0])
            else:
                print(f"Warning: Detector {i_global} has all pointing vectors excluded.")
                new_place = place
            det.pointing_state[i_local, POINTING_PLACE_IDX] = new_place
            det.pointing[i_local] = grid[new_place]
            if debug:
                print(f"Detector {i_global} (local {i_local}): Pointing location {new_place}, Excluded: {len(clear) == 0}")


def detectorPointingInitialize(sim_data: Any, grid_points: int) -> None:
    """
    Initializes pointing and pointing_state inside component detectors,
    adds a pointing sphere to sim_data, and sets initial unexcluded directions.
    """
    generate_pointing_sphere(sim_data, grid_points)

    for _, _, det, _ in sim_data.iter_detector_components():
        sensor_count = len(det.filt)
        det.pointing_state = np.zeros((sensor_count, 2), dtype=int)
        det.pointing_state[:, POINTING_COUNT_IDX] = grid_points
        det.pointing_state[:, POINTING_PLACE_IDX] = np.random.randint(
            0, grid_points - 1, size=sensor_count
        )

    update_detector_pointing(sim_data)
