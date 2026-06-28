import numpy as np
import plotly.graph_objects as go
from lambertian import lambertiansphere, includedAngle

def demo_lambertian():
    """
    Runs a demonstration of the lambertiansphere function,
    including example calculations and a plot.
    """
    SATELLITE_ALBEDO = 0.2
    SATELLITE_RADIUS = 1.5
    BASE_BRIGHTNESS = 1361  # Solar constant in W/m^2
    OBSERVER_DISTANCE = 1000e3  # 1000 km

    print(
        f"--- Simulating a sphere with Albedo={SATELLITE_ALBEDO}, "
        f"Radius={SATELLITE_RADIUS}m, at a distance of "
        f"{OBSERVER_DISTANCE / 1e3} km ---"
    )
    print(
        f"--- Base Brightness (e.g., Solar Constant) = "
        f"{BASE_BRIGHTNESS} W/m^2 ---"
    )

    # --- Example 1: Full Illumination (Phase Angle = 0) ---
    print("--- Example 1: Full Illumination ---")
    vec_sun_1 = np.array([1, 0, 0])  # Light source direction
    vec_obs_1 = np.array([1, 0, 0]) * OBSERVER_DISTANCE  # Observer direction
    
    # Calculate angle and observer distance for the first example
    angle_light_observer_1 = includedAngle(
        np.array([vec_sun_1]), np.array([vec_obs_1])
    )
    norm_observer_1 = np.linalg.norm(np.array([vec_obs_1]), axis=1)

    emitted_brightness_1 = lambertiansphere(
        angle_light_observer_1, np.array([SATELLITE_ALBEDO]),
        np.array([SATELLITE_RADIUS]), np.array([BASE_BRIGHTNESS])
    )
    brightness_1 = emitted_brightness_1 / (np.pi * norm_observer_1 ** 2)
    angle_1 = np.rad2deg(np.arccos(np.dot(vec_sun_1, vec_obs_1 / 
                                          np.linalg.norm(vec_obs_1))))
    print(f"Phase Angle: {angle_1:.2f} degrees")
    print(f"Apparent Brightness: {brightness_1[0]:.4e} W/m^2\n")

    # --- Example 2: Half Illumination (Phase Angle = 90) ---
    print("--- Example 2: Half Illumination ---")
    vec_sun_2 = np.array([1, 0, 0])
    vec_obs_2 = np.array([0, 1, 0]) * OBSERVER_DISTANCE
    
    # Calculate angle and observer distance for the second example
    angle_light_observer_2 = includedAngle(
        np.array([vec_sun_2]), np.array([vec_obs_2])
    )
    norm_observer_2 = np.linalg.norm(np.array([vec_obs_2]), axis=1)

    emitted_brightness_2 = lambertiansphere(
        angle_light_observer_2, np.array([SATELLITE_ALBEDO]),
        np.array([SATELLITE_RADIUS]), np.array([BASE_BRIGHTNESS])
    )
    brightness_2 = emitted_brightness_2 / (np.pi * norm_observer_2 ** 2)
    angle_2 = np.rad2deg(np.arccos(np.dot(vec_sun_2, vec_obs_2 / 
                                          np.linalg.norm(vec_obs_2))))
    print(f"Phase Angle: {angle_2:.2f} degrees")
    print(f"Apparent Brightness: {brightness_2[0]:.4e} W/m^2\n")

    # --- Example 3: No Illumination (Phase Angle = 180) ---
    print("--- Example 3: No Illumination ---")
    vec_sun_3 = np.array([1, 0, 0])
    vec_obs_3 = np.array([-1, 0, 0]) * OBSERVER_DISTANCE
    
    # Calculate angle and observer distance for the third example
    angle_light_observer_3 = includedAngle(
        np.array([vec_sun_3]), np.array([vec_obs_3])
    )
    norm_observer_3 = np.linalg.norm(np.array([vec_obs_3]), axis=1)

    emitted_brightness_3 = lambertiansphere(
        angle_light_observer_3, np.array([SATELLITE_ALBEDO]),
        np.array([SATELLITE_RADIUS]), np.array([BASE_BRIGHTNESS])
    )
    brightness_3 = emitted_brightness_3 / (np.pi * norm_observer_3 ** 2)
    angle_3 = np.rad2deg(np.arccos(np.dot(vec_sun_3, vec_obs_3 / 
                                          np.linalg.norm(vec_obs_3))))
    print(f"Phase Angle: {angle_3:.2f} degrees")
    print(f"Apparent Brightness: {brightness_3[0]:.4e} W/m^2\n")

    # --- Generate Plot Data ---
    print("\n--- Generating Plot Data ---")
    angles_deg = np.linspace(0, 180, 200)
    angles_rad = np.deg2rad(angles_deg)

    # Prepare inputs for vectorized calculation
    num_points = len(angles_rad)
    plot_vec_light = np.array([1, 0, 0])  # Single light source vector
    plot_vec_obs_raw = np.zeros((num_points, 3)) 
    plot_vec_obs_raw[:, 0] = np.cos(angles_rad)
    plot_vec_obs_raw[:, 1] = np.sin(angles_rad)
    
    # plot_vec_obs_for_angle is the vector from sphere to observer, 
    # magnitude is distance
    plot_vec_obs_for_angle = plot_vec_obs_raw * OBSERVER_DISTANCE

    # includedAngle expects (N,3) arrays for both arguments
    plot_vec_light_expanded = np.tile(plot_vec_light, (num_points, 1))

    # Calculate angles and observer distances for the plot data
    angle_light_observer_plot = includedAngle(
        plot_vec_light_expanded, plot_vec_obs_for_angle
    )
    norm_observer_plot = np.linalg.norm(plot_vec_obs_for_angle, axis=1)

    albedos = np.full(num_points, SATELLITE_ALBEDO)
    radii = np.full(num_points, SATELLITE_RADIUS)
    base_brightnesses = np.full(num_points, BASE_BRIGHTNESS)

    emitted_brightness_values = lambertiansphere(
        angle_light_observer_plot, albedos, radii, base_brightnesses
    )
    brightness_values = emitted_brightness_values / (np.pi * 
                                                     norm_observer_plot ** 2)

    # --- Create Plot ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=angles_deg, y=brightness_values, 
                             mode='lines', name='Apparent Brightness'))
    title_text = (
        f'Lambertian Sphere Apparent Brightness vs. Phase Angle<br>'
        f'<sup>Albedo={SATELLITE_ALBEDO}, Radius={SATELLITE_RADIUS}m, '
        f'Distance={OBSERVER_DISTANCE / 1e3}km</sup>'
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
