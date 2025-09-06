import numpy as np
from datetime import datetime, timezone
from plotting_3d import plot_3d_scatter


def generate_log_spherical_points(
    num_points: int,
    inner_radius: float,
    outer_radius: float,
    object_size_m: float = 1.0,
    seed: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates 3D points with logarithmic radial and uniform angular distribution.

    This function creates a point cloud where point distances from the origin are
    logarithmically spaced. On any given spherical shell, points are distributed
    uniformly using the Fibonacci lattice method. Each point is associated with a
    specified object size.

    Args:
        num_points: The total number of points to generate.
        inner_radius: The minimum distance from the origin (must be positive).
        outer_radius: The maximum distance from the origin (must be >= inner_radius).
        object_size_m: The size in meters to be associated with each point.
                       Defaults to 1.0.
        seed: An optional integer to seed the random number generator for
              reproducible shuffling.

    Returns:
        A tuple containing:
        - A NumPy array of shape (num_points, 3) for the Cartesian coordinates.
        - A NumPy array of shape (num_points,) for the object size in meters.
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

    # --- 5. Create an array for the object sizes ---
    sizes = np.full(num_points, object_size_m, dtype=float)

    return points, sizes

if __name__ == '__main__':
    # --- Demo of Point Generation and Visualization ---
    # This provides a consistent demonstration entry point, matching vibevolts_demo.py
    NUM_POINTS = 100
    # Using the same radii as the main simulation for consistency
    INNER_RADIUS = 2000000
    OUTER_RADIUS = 84328000
    OBJECT_SIZE = 1.0

    print(f"Generating {NUM_POINTS} points from radius {INNER_RADIUS} to {OUTER_RADIUS}...")
    generated_points, _ = generate_log_spherical_points(
        num_points=NUM_POINTS,
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        object_size_m=OBJECT_SIZE,
        seed=42
    )
    print("Point generation complete.\n")

    # Use the unified visualization function
    plot_3d_scatter(
        positions=generated_points,
        title="Fixed Points Distribution (from generate_log_spherical_points.py)",
        plot_time=datetime.now(timezone.utc),
        labels=[f"Point {i}" for i in range(len(generated_points))],
        marker_size=1,
        trace_name='Fixed Points'
    )
