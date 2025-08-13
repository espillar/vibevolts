import numpy as np
import plotly.graph_objects as go

def lambertiansphere(
    vec_from_sphere_to_light: np.ndarray,
    vec_from_sphere_to_observer: np.ndarray,
    albedo: float,
    radius: float
) -> float:
    """
    Calculates the effective brightness of a
    Lambertian sphere.

    This function determines the apparent brightness of
    a diffusely reflecting sphere based on the angle
    between the light source and the observer, the
    sphere's albedo (reflectivity), and its size.

    Args:
        vec_from_sphere_to_light: A 3-element NumPy
            array representing the direction vector from
            the sphere to the light source.
        vec_from_sphere_to_observer: A 3-element NumPy
            array representing the direction vector from
            the sphere to the observer.
        albedo: The fraction of incident light that is
            reflected (0.0 to 1.0).
        radius: The radius of the sphere in meters.

    Returns:
        The effective brightness cross-section in
        square meters. This value is proportional to
        the total light reflected towards the observer.
    """
    # --- Input Validation ---
    if not 0.0 <= albedo <= 1.0:
        raise ValueError(
            "Albedo must be between 0.0 and 1.0."
        )
    if radius < 0:
        raise ValueError("Radius cannot be negative.")

    # --- 1. Normalize the input vectors ---
    # This ensures we are only working with directions.
    norm_light = np.linalg.norm(vec_from_sphere_to_light)
    norm_observer = np.linalg.norm(
        vec_from_sphere_to_observer
    )

    if norm_light == 0 or norm_observer == 0:
        raise ValueError(
            "Input vectors cannot have zero length."
        )

    unit_vec_light = vec_from_sphere_to_light / norm_light
    unit_vec_observer = (
        vec_from_sphere_to_observer / norm_observer
    )

    # --- 2. Calculate the phase angle (alpha) ---
    # The dot product of the unit vectors gives the
    # cosine of the physical phase angle.
    cos_alpha = np.dot(unit_vec_light, unit_vec_observer)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    # --- 3. Calculate the phase function value ---
    # This formula integrates the Lambertian BRDF over
    # the visible, illuminated portion of the sphere.
    # It is maximal (value = pi) when alpha = 0,
    # and minimal (value = 0) when alpha = pi.
    term1 = np.sin(alpha)
    term2 = (np.pi - alpha) * np.cos(alpha)
    phase_function_value = (2 / (3 * np.pi)) * (term1 + term2)

    # --- 4. Calculate cross-sectional area ---
    cross_sectional_area = np.pi * (radius ** 2)

    # --- 5. Calculate total reflected brightness ---
    # This is often called the effective brightness
    # cross-section.
    effective_brightness = (
        albedo *
        cross_sectional_area *
        phase_function_value
    )

    return effective_brightness


# --- Main Execution Block for Demonstration ---
if __name__ == '__main__':
    # Define properties for a sample satellite
    SATELLITE_ALBEDO = 0.2
    SATELLITE_RADIUS = 1.5  # in meters

    print(
        f"--- Simulating a sphere with\n"
        f" Albedo={SATELLITE_ALBEDO} and "
        f"Radius={SATELLITE_RADIUS}m ---\n"
    )

    # --- Example 1: Full Illumination ('Full Moon') ---
    # The observer is between the sun and the sphere.
    # The light source and observer vectors point in the
    # same direction from the sphere's perspective.
    # Physical phase angle should be 0 degrees,
    # yielding maximum brightness.
    print("--- Example 1: Full Illumination ---")
    vec_sun_1 = np.array([1, 0, 0])
    vec_obs_1 = np.array([1, 0, 0])
    print(f"  Vec to Light:    {vec_sun_1}")
    print(f"  Vec to Observer: {vec_obs_1}")
    
    brightness_1 = lambertiansphere(
        vec_sun_1, vec_obs_1,
        SATELLITE_ALBEDO, SATELLITE_RADIUS
    )
    
    # Calculate the physical angle for the printout
    u_light_1 = vec_sun_1 / np.linalg.norm(vec_sun_1)
    u_obs_1 = vec_obs_1 / np.linalg.norm(vec_obs_1)
    dot_prod_1 = np.dot(u_light_1, u_obs_1)
    angle_1 = np.rad2deg(np.arccos(dot_prod_1))
    print(f"Phase Angle: {angle_1:.2f} degrees")
    print(f"Effective Brightness: {brightness_1:.4f} m^2\n")


    # --- Example 2: Half Illumination ('Half Moon') ---
    # The sun, sphere, and observer form a right
    # angle. Physical phase angle should be 90 degrees.
    print("--- Example 2: Half Illumination ---")
    vec_sun_2 = np.array([1, 0, 0])
    vec_obs_2 = np.array([0, 1, 0])
    print(f"  Vec to Light:    {vec_sun_2}")
    print(f"  Vec to Observer: {vec_obs_2}")

    brightness_2 = lambertiansphere(
        vec_sun_2, vec_obs_2,
        SATELLITE_ALBEDO, SATELLITE_RADIUS
    )

    # Calculate the physical angle for the printout
    u_light_2 = vec_sun_2 / np.linalg.norm(vec_sun_2)
    u_obs_2 = vec_obs_2 / np.linalg.norm(vec_obs_2)
    dot_prod_2 = np.dot(u_light_2, u_obs_2)
    angle_2 = np.rad2deg(np.arccos(dot_prod_2))
    print(f"Phase Angle: {angle_2:.2f} degrees")
    print(f"Effective Brightness: {brightness_2:.4f} m^2\n")

    # --- Example 3: No Illumination ('New Moon') ---
    # The sphere is between the sun and the observer.
    # The light source and observer vectors point in
    # opposite directions from the sphere's perspective.
    # Physical phase angle should be 180 degrees,
    # yielding minimum brightness.
    print("--- Example 3: No Illumination ---")
    vec_sun_3 = np.array([1, 0, 0])
    vec_obs_3 = np.array([-1, 0, 0])
    print(f"  Vec to Light:    {vec_sun_3}")
    print(f"  Vec to Observer: {vec_obs_3}")

    brightness_3 = lambertiansphere(
        vec_sun_3, vec_obs_3,
        SATELLITE_ALBEDO, SATELLITE_RADIUS
    )

    # Calculate the physical angle for the printout
    u_light_3 = vec_sun_3 / np.linalg.norm(vec_sun_3)
    u_obs_3 = vec_obs_3 / np.linalg.norm(vec_obs_3)
    dot_prod_3 = np.dot(u_light_3, u_obs_3)
    angle_3 = np.rad2deg(np.arccos(dot_prod_3))
    print(f"Phase Angle: {angle_3:.2f} degrees")
    print(f"Effective Brightness: {brightness_3:.4f} m^2\n")

    # --- Plotting Section ---
    # This section plots the effective brightness as a
    # function of the physical phase angle from 0 to
    # 180 degrees.
    print("\n--- Generating Plot Data ---")
    
    # Define a range of angles from 0 to 180
    angles_deg = np.linspace(0, 180, 200)
    angles_rad = np.deg2rad(angles_deg)
    
    brightness_values = []
    
    # Keep the light source vector constant
    plot_vec_light = np.array([1, 0, 0])
    
    # Calculate brightness for each angle
    for angle in angles_rad:
        # Rotate the observer vector around the z-axis
        plot_vec_obs = np.array(
            [np.cos(angle), np.sin(angle), 0]
        )
        
        brightness = lambertiansphere(
            plot_vec_light, 
            plot_vec_obs, 
            SATELLITE_ALBEDO, 
            SATELLITE_RADIUS
        )
        brightness_values.append(brightness)
        
    # Create the plot using Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=angles_deg, 
        y=brightness_values, 
        mode='lines',
        name='Effective Brightness'
    ))
    
    title_text = (
        f'Lambertian Sphere Brightness vs. Phase Angle'
        f'<br><sup>Albedo={SATELLITE_ALBEDO}, '
        f'Radius={SATELLITE_RADIUS}m</sup>'
    )
    
    fig.update_layout(
        title_text=title_text,
        xaxis_title="Physical Phase Angle (degrees)",
        yaxis_title="Effective Brightness (m^2)",
        template="plotly_white",
        xaxis=dict(range=[0, 180]),
        yaxis=dict(range=[0, max(brightness_values) * 1.1])
    )
    
    print(
        "\n--- Displaying Plot (requires Plotly "
        "in a notebook environment) ---"
    )
    # In a Jupyter notebook or similar environment,
    # this will render the plot.
    fig.show()
