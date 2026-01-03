import numpy as np
import plotly.graph_objects as go
from lambertian import lambertiansphere

def demo_lambertian():
    """
    Runs a demonstration of the lambertiansphere function,
    including example calculations and a plot.
    """
    SATELLITE_ALBEDO = 0.2
    SATELLITE_RADIUS = 1.5
    BASE_BRIGHTNESS = 1361  # Solar constant in W/m^2
    OBSERVER_DISTANCE = 1000e3  # 1000 km

    print(f"--- Simulating a sphere with Albedo={SATELLITE_ALBEDO}, Radius={SATELLITE_RADIUS}m, at a distance of {OBSERVER_DISTANCE / 1e3} km ---")
    print(f"--- Base Brightness (e.g., Solar Constant) = {BASE_BRIGHTNESS} W/m^2 ---")

    # --- Example 1: Full Illumination (Phase Angle = 0) ---
    print("--- Example 1: Full Illumination ---")
    vec_sun_1 = np.array([1, 0, 0])  # Light source direction
    vec_obs_1 = np.array([1, 0, 0]) * OBSERVER_DISTANCE  # Observer direction and distance
    brightness_1 = lambertiansphere(
        np.array([vec_sun_1]), np.array([vec_obs_1]), np.array([SATELLITE_ALBEDO]), np.array([SATELLITE_RADIUS]), np.array([BASE_BRIGHTNESS])
    )
    angle_1 = np.rad2deg(np.arccos(np.dot(vec_sun_1, vec_obs_1 / np.linalg.norm(vec_obs_1))))
    print(f"Phase Angle: {angle_1:.2f} degrees")
    print(f"Apparent Brightness: {brightness_1[0]:.4e} W/m^2\n")

    # --- Example 2: Half Illumination (Phase Angle = 90) ---
    print("--- Example 2: Half Illumination ---")
    vec_sun_2 = np.array([1, 0, 0])
    vec_obs_2 = np.array([0, 1, 0]) * OBSERVER_DISTANCE
    brightness_2 = lambertiansphere(
        np.array([vec_sun_2]), np.array([vec_obs_2]), np.array([SATELLITE_ALBEDO]), np.array([SATELLITE_RADIUS]), np.array([BASE_BRIGHTNESS])
    )
    angle_2 = np.rad2deg(np.arccos(np.dot(vec_sun_2, vec_obs_2 / np.linalg.norm(vec_obs_2))))
    print(f"Phase Angle: {angle_2:.2f} degrees")
    print(f"Apparent Brightness: {brightness_2[0]:.4e} W/m^2\n")

    # --- Example 3: No Illumination (Phase Angle = 180) ---
    print("--- Example 3: No Illumination ---")
    vec_sun_3 = np.array([1, 0, 0])
    vec_obs_3 = np.array([-1, 0, 0]) * OBSERVER_DISTANCE
    brightness_3 = lambertiansphere(
        np.array([vec_sun_3]), np.array([vec_obs_3]), np.array([SATELLITE_ALBEDO]), np.array([SATELLITE_RADIUS]), np.array([BASE_BRIGHTNESS])
    )
    angle_3 = np.rad2deg(np.arccos(np.dot(vec_sun_3, vec_obs_3 / np.linalg.norm(vec_obs_3))))
    print(f"Phase Angle: {angle_3:.2f} degrees")
    print(f"Apparent Brightness: {brightness_3[0]:.4e} W/m^2\n")

    # --- Generate Plot Data ---
    print("\n--- Generating Plot Data ---")
    angles_deg = np.linspace(0, 180, 200)
    angles_rad = np.deg2rad(angles_deg)

    # Prepare inputs for vectorized calculation
    num_points = len(angles_rad)
    plot_vec_light = np.tile([1, 0, 0], (num_points, 1))
    plot_vec_obs = np.zeros((num_points, 3))
    plot_vec_obs[:, 0] = np.cos(angles_rad)
    plot_vec_obs[:, 1] = np.sin(angles_rad)
    plot_vec_obs *= OBSERVER_DISTANCE

    albedos = np.full(num_points, SATELLITE_ALBEDO)
    radii = np.full(num_points, SATELLITE_RADIUS)
    base_brightnesses = np.full(num_points, BASE_BRIGHTNESS)

    brightness_values = lambertiansphere(plot_vec_light, plot_vec_obs, albedos, radii, base_brightnesses)

    # --- Create Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=angles_deg, y=brightness_values, mode='lines', name='Apparent Brightness'))
    title_text = (
        f'Lambertian Sphere Apparent Brightness vs. Phase Angle<br>'
        f'<sup>Albedo={SATELLITE_ALBEDO}, Radius={SATELLITE_RADIUS}m, Distance={OBSERVER_DISTANCE / 1e3}km</sup>'
    )
    fig.update_layout(
        title_text=title_text,
        xaxis_title="Physical Phase Angle (degrees)",
        yaxis_title=f"Apparent Brightness (W/m^2)",
        template="plotly_white",
        xaxis=dict(range=[0, 180]),
        yaxis_type="log",  # Use log scale for better visualization
        yaxis=dict(autorange=True)
    )

    print("\n--- Returning Plot ---")
    return fig
