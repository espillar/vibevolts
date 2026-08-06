# fibonacciSearch
# this contains the functions necessary to support a set of detectors
# that searches the sphere using a fibonacci grid

import numpy as np
import plotly.graph_objects as go

# def searchStruct(sim_data,detect):
#     '''
#     creates the data structure for each of the satellite detectors,
#     adds the structure to the detector
#     '''
#     theta = detect[:,IFOV_IDX]/2

#     # Calculate solid angle 
#     theta = fov / 2
#     solid_angle = 2 * np.pi * (1 - np.cos(theta))
    
#     # Calculate grid_points - blow things up by 0.25 for overlap
#     grid_points = int(4 * np.pi / solid_angle * 1.25)

#     # Generate and store the pointing sphere and place in ['pointing_sphers'][n]
#     generate_pointing_sphere(sim_data, grid_points)

def pointing_vectors(n: int) -> np.ndarray:
    """
    Generates n equally spaced points on a unit sphere using the Fibonacci lattice algorithm.

    Args:
        n: The number of points to generate.

    Returns:
        A NumPy array of shape (n, 3) for the Cartesian coordinates of the points.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    indices = np.arange(0, n, dtype=float) + 0.5
    z = 1 - 2 * indices / n
    radius_xy = np.sqrt(1 - z**2)
    golden_angle = np.pi * (3. - np.sqrt(5.))
    theta = golden_angle * indices
    x = radius_xy * np.cos(theta)
    y = radius_xy * np.sin(theta)

    unit_vectors = np.stack([x, y, z], axis=1)
    return unit_vectors

def resort_vectors_by_proximity(unit_vectors: np.ndarray) -> np.ndarray:
    """
    Resorts a list of vectors by making each subsequent vector the closest one
    in the remaining set to the previous one.

    Args:
        unit_vectors: A NumPy array of shape (n, 3) representing the vectors.

    Returns:
        A new NumPy array with the reordered vectors.
    """
    if unit_vectors.ndim != 2 or unit_vectors.shape[1] != 3:
        raise ValueError("unit_vectors array must have shape (n, 3).")

    n_vectors = unit_vectors.shape[0]
    if n_vectors == 0:
        return np.array([])

    remaining_indices = list(range(n_vectors))
    sorted_vectors = np.zeros_like(unit_vectors)
    
    # Start with the first vector
    current_index = remaining_indices.pop(0)
    sorted_vectors[0] = unit_vectors[current_index]
    
    for i in range(1, n_vectors):
        last_vector = sorted_vectors[i-1]
        
        # Vectorized distance computation to all remaining candidates
        rem_coords = unit_vectors[remaining_indices]
        dists = np.linalg.norm(rem_coords - last_vector, axis=1)
        
        # Find index of minimum distance
        best_idx_in_rem = np.argmin(dists)
        best_index = remaining_indices[best_idx_in_rem]

        sorted_vectors[i] = unit_vectors[best_index]
        remaining_indices.pop(best_idx_in_rem)

    return sorted_vectors


def plot_vectors_on_sphere(vectors: np.ndarray, title: str) -> go.Figure:
    """
    Creates a 3D plot of vectors on a sphere.

    Args:
        vectors: A NumPy array of shape (n, 3) representing the vectors.
        title: The title of the plot.

    Returns:
        A Plotly figure object.
    """
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors array must have shape (n, 3).")

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',
            opacity=0.8
        ),
        name='Vectors'
    ))

    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = 1.0 * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = 1.0 * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = 1.0 * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Blues', showscale=False, opacity=0.5, name='Sphere'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Objects'
    )

    return fig

def test_vector_resorting():
    """
    Tests the vector resorting and plots the Euclidean distance between subsequent vectors.
    """
    # 1. Generate 100 vectors
    vectors = pointing_vectors(100)

    # 2. Calculate distances without resorting
    diffs_unsorted = np.linalg.norm(np.diff(vectors, axis=0), axis=1)

    # 3. Resort the vectors
    sorted_vectors = resort_vectors_by_proximity(vectors)
    
    # 4. Calculate distances with resorting
    diffs_sorted = np.linalg.norm(np.diff(sorted_vectors, axis=0), axis=1)

    # 5. Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=diffs_unsorted, mode='lines', name='Unsorted'))
    fig.add_trace(go.Scatter(y=diffs_sorted, mode='lines', name='Resorted by Proximity'))
    fig.update_layout(
        title="Euclidean Distance Between Subsequent Vectors",
        xaxis_title="Vector Index",
        yaxis_title="Euclidean Distance"
    )
    fig.add_annotation(
        text="Generated by: test_vector_resorting",
        xref="paper", yref="paper",
        x=0.5, y=1.01,
        showarrow=False,
        font=dict(size=10, color="gray"),
        xanchor="center", yanchor="bottom"
    )
    return fig

if __name__ == '__main__':
    fig = test_vector_resorting()
    fig.show()
