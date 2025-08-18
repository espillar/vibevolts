import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def generate_log_spherical_points(
    num_points: int, 
    inner_radius: float, 
    outer_radius: float,
    seed: int = None
) -> np.ndarray:
    """
    Generates a set of 3D points with logarithmic radial and uniform angular distribution.

    The function creates a point cloud where the distances of points from the origin
    are logarithmically spaced between an inner and outer radius. For any given
    radius, the points on the corresponding spherical shell are distributed
    uniformly using the Fibonacci lattice method, which ensures an equal-area
    distribution.

    Args:
        num_points: The total number of points to generate.
        inner_radius: The minimum distance from the origin. Must be positive.
        outer_radius: The maximum distance from the origin. Must be greater than
                      or equal to inner_radius.
        seed: An optional integer to seed the random number generator for
              reproducible shuffling.

    Returns:
        A NumPy array of shape (num_points, 3) containing the Cartesian
        coordinates (x, y, z) of the generated points.
    """
    # Input validation
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points must be a positive integer.")
    if not (isinstance(inner_radius, (int, float)) and inner_radius > 0):
        raise ValueError("inner_radius must be a positive number.")
    if not (isinstance(outer_radius, (int, float)) and outer_radius >= inner_radius):
        raise ValueError("outer_radius must be a positive number and >= inner_radius.")

    # --- 1. Generate uniformly distributed unit vectors (Fibonacci Lattice) ---
    # Create an array of indices for each point
    indices = np.arange(0, num_points, dtype=float) + 0.5

    # Uniformly distribute points along the z-axis (cos(phi))
    z = 1 - 2 * indices / num_points

    # Calculate the radius in the xy-plane for each point
    radius_xy = np.sqrt(1 - z**2)

    # Calculate the azimuthal angle using the golden angle
    golden_angle = np.pi * (3. - np.sqrt(5.))
    theta = golden_angle * indices

    # Convert to Cartesian coordinates for the unit sphere
    x = radius_xy * np.cos(theta)
    y = radius_xy * np.sin(theta)
    
    # Stack into an (N, 3) array of unit vectors
    unit_vectors = np.stack([x, y, z], axis=1)

    # --- 2. Generate logarithmically spaced radii ---
    # The start and stop parameters for logspace are exponents
    start_exp = np.log10(inner_radius)
    stop_exp = np.log10(outer_radius)
    radii = np.logspace(start_exp, stop_exp, num_points)

    # --- 3. Shuffle radii to decouple radius from spherical position ---
    # This prevents the smallest radii from always mapping to points near the
    # north pole, creating a more uniform visual distribution of densities.
    rng = np.random.default_rng(seed)
    rng.shuffle(radii)

    # --- 4. Scale unit vectors by radii using broadcasting ---
    # Reshape radii to (N, 1) to broadcast with (N, 3) unit_vectors
    points = unit_vectors * radii[:, np.newaxis]

    return points

def visualize_point_distribution(points: np.ndarray):
    """
    Visualizes the distribution of a 3D point cloud with plots.

    This function generates and displays plots for analysis:
    1. A 3D interactive scatter plot of the points.
    2. Histograms of the radial, longitude, and latitude distributions.

    Args:
        points: A NumPy array of shape (num_points, 3) with Cartesian coordinates.
    """
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input 'points' must be a NumPy array of shape (N, 3).")

    # --- 1. Display a sample of the data points ---
    print("--- Sample of Generated Points (X, Y, Z, Radius) ---")
    radii = np.linalg.norm(points, axis=1)
    sample_data = np.hstack((points[:15], radii[:15, np.newaxis]))
    for row in sample_data:
        print(f"[{row[0]:>8.2f} {row[1]:>8.2f} {row[2]:>8.2f} | {row[3]:>8.2f}]")
    print("-" * 45 + "\n")


    # --- 2. 3D Scatter Plot using Plotly ---
    print("Displaying 3D scatter plot...")
    fig_3d = px.scatter_3d(
        x=points[:, 0], 
        y=points[:, 1], 
        z=points[:, 2],
        color=radii,
        title="3D Scatter Plot of Generated Points",
        labels={'x': 'X-axis', 'y': 'Y-axis', 'z': 'Z-axis', 'color': 'Radius'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_3d.update_traces(marker=dict(size=2))
    fig_3d.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig_3d.show()

    # --- 3. Calculate Spherical Coordinates for Histograms ---
    # Longitude (Azimuthal angle, theta) from -180 to 180 degrees
    longitude_deg = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    # Latitude (Polar angle, phi) from -90 to 90 degrees
    polar_angle = np.arccos(points[:, 2] / radii)
    latitude_deg = np.degrees(np.pi / 2 - polar_angle)

    # --- 4. Create Histograms using Plotly ---
    print("Displaying distribution histograms...")
    fig_hist = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Distribution of Radial Distances',
            'Distribution of Longitude',
            'Distribution of Latitude'
        )
    )

    # Radii Histogram
    fig_hist.add_trace(
        go.Histogram(x=radii, nbinsx=50, name='Radius', marker_color='skyblue'),
        row=1, col=1
    )
    # Longitude Histogram
    fig_hist.add_trace(
        go.Histogram(x=longitude_deg, nbinsx=50, name='Longitude', marker_color='salmon'),
        row=1, col=2
    )
    # Latitude Histogram
    fig_hist.add_trace(
        go.Histogram(x=latitude_deg, nbinsx=50, name='Latitude', marker_color='lightgreen'),
        row=1, col=3
    )

    fig_hist.update_layout(
        title_text='Analysis of Point Cloud Distribution',
        xaxis1_title='Radius', yaxis1_title='Frequency',
        xaxis2_title='Longitude (Degrees)', yaxis2_title='Frequency',
        xaxis3_title='Latitude (Degrees)', yaxis3_title='Frequency',
        showlegend=False,
        height=500
    )
    fig_hist.show()


if __name__ == '__main__':
    # --- Example Usage ---
    # Define parameters for the point cloud
    NUM_POINTS = 5000
    INNER_RADIUS = 10.0
    OUTER_RADIUS = 100.0

    # Generate the points
    print(f"Generating {NUM_POINTS} points from radius {INNER_RADIUS} to {OUTER_RADIUS}...")
    generated_points = generate_log_spherical_points(
        num_points=NUM_POINTS,
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        seed=42 # Add seed for reproducibility
    )
    print("Point generation complete.\n")

    # Visualize the generated points
    visualize_point_distribution(generated_points)
