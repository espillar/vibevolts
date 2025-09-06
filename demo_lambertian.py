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

    print(f"--- Simulating a sphere with Albedo={SATELLITE_ALBEDO} and Radius={SATELLITE_RADIUS}m ---\n")

    print("--- Example 1: Full Illumination ---")
    vec_sun_1 = np.array([1, 0, 0])
    vec_obs_1 = np.array([1, 0, 0])
    brightness_1 = lambertiansphere(vec_sun_1, vec_obs_1, SATELLITE_ALBEDO, SATELLITE_RADIUS)
    angle_1 = np.rad2deg(np.arccos(np.dot(vec_sun_1, vec_obs_1)))
    print(f"Phase Angle: {angle_1:.2f} degrees")
    print(f"Effective Brightness: {brightness_1:.4f} m^2\n")

    print("--- Example 2: Half Illumination ---")
    vec_sun_2 = np.array([1, 0, 0])
    vec_obs_2 = np.array([0, 1, 0])
    brightness_2 = lambertiansphere(vec_sun_2, vec_obs_2, SATELLITE_ALBEDO, SATELLITE_RADIUS)
    angle_2 = np.rad2deg(np.arccos(np.dot(vec_sun_2/np.linalg.norm(vec_sun_2), vec_obs_2/np.linalg.norm(vec_obs_2))))
    print(f"Phase Angle: {angle_2:.2f} degrees")
    print(f"Effective Brightness: {brightness_2:.4f} m^2\n")

    print("--- Example 3: No Illumination ---")
    vec_sun_3 = np.array([1, 0, 0])
    vec_obs_3 = np.array([-1, 0, 0])
    brightness_3 = lambertiansphere(vec_sun_3, vec_obs_3, SATELLITE_ALBEDO, SATELLITE_RADIUS)
    angle_3 = np.rad2deg(np.arccos(np.dot(vec_sun_3, vec_obs_3)))
    print(f"Phase Angle: {angle_3:.2f} degrees")
    print(f"Effective Brightness: {brightness_3:.4f} m^2\n")

    print("\n--- Generating Plot Data ---")
    angles_deg = np.linspace(0, 180, 200)
    angles_rad = np.deg2rad(angles_deg)
    brightness_values = []
    plot_vec_light = np.array([1, 0, 0])
    for angle in angles_rad:
        plot_vec_obs = np.array([np.cos(angle), np.sin(angle), 0])
        brightness = lambertiansphere(plot_vec_light, plot_vec_obs, SATELLITE_ALBEDO, SATELLITE_RADIUS)
        brightness_values.append(brightness)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=angles_deg, y=brightness_values, mode='lines', name='Effective Brightness'))
    title_text = f'Lambertian Sphere Brightness vs. Phase Angle<br><sup>Albedo={SATELLITE_ALBEDO}, Radius={SATELLITE_RADIUS}m</sup>'
    fig.update_layout(
        title_text=title_text,
        xaxis_title="Physical Phase Angle (degrees)",
        yaxis_title="Effective Brightness (m^2)",
        template="plotly_white",
        xaxis=dict(range=[0, 180]),
        yaxis=dict(range=[0, max(brightness_values) * 1.1])
    )

    print("\n--- Displaying Plot ---")
    fig.show()

if __name__ == '__main__':
    demo_lambertian()
