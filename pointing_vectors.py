import numpy as np
import plotly.graph_objects as go


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
