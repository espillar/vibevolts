import numpy as np

def simple_lambertian(
    diameter: float,
    distance: float,
    albedo: float,
    angle: float,
    base_brightness: float
) -> float:
    """
    Calculates the apparent brightness of a Lambertian sphere.

    This function computes the apparent brightness of a diffusely
    reflecting sphere based on its physical properties, viewing
    geometry, and a given base incident brightness. It simplifies
    the calculation by taking the phase angle directly, rather than
    calculating it from vectors.

    Args:
        diameter: The diameter of the sphere in meters.
        distance: The distance from the sphere to the observer
            in meters.
        albedo: The fraction of incident light that is
            reflected (0.0 to 1.0).
        angle: The phase angle in radians. This is the angle
            between the light source and the observer as seen
            from the sphere's center (expected to be between 0 and pi).
        base_brightness: The incident flux or brightness of the
            light source at the sphere's location (e.g., in
            Watts per square meter or photons / s / m^2).

    Returns:
        The apparent brightness of the sphere as observed from
        the specified distance (e.g., in Watts per square meter
        or photons / s / m^2).
    """
    if not 0.0 <= albedo <= 1.0:
        raise ValueError("Albedo must be between 0.0 and 1.0.")
    if diameter < 0:
        raise ValueError("Diameter cannot be negative.")
    if distance <= 0:
        raise ValueError("Distance must be positive.")

    # Calculate radius from diameter
    radius = diameter / 2.0

    # The phase angle is physically constrained to be between 0 and pi.
    # We clip the value to handle out-of-range inputs gracefully.
    angle = np.clip(angle, 0, np.pi)

    # This is the phase function for a Lambertian sphere.
    # It describes how the brightness changes with the phase angle.
    term1 = np.sin(angle)
    term2 = (np.pi - angle) * np.cos(angle)
    phase_function_value = (2 / (3 * np.pi)) * (term1 + term2)

    # Cross-sectional area of the sphere.
    cross_sectional_area = np.pi * (radius ** 2)

    # The effective cross-section combines the sphere's size,
    # reflectivity (albedo), and the phase function.
    effective_cross_section = (
        albedo *
        cross_sectional_area *
        phase_function_value
    )

    # The apparent brightness is the incident brightness multiplied
    # by the effective reflecting area, with the resulting light
    # distributed according to the inverse square law.
    apparent_brightness = (
        (base_brightness * effective_cross_section) /
        (np.pi * distance ** 2)
    )

    return apparent_brightness




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
    if not 0.0 <= albedo <= 1.0:
        raise ValueError("Albedo must be between 0.0 and 1.0.")
    if radius < 0:
        raise ValueError("Radius cannot be negative.")

    norm_light = np.linalg.norm(vec_from_sphere_to_light)
    norm_observer = np.linalg.norm(vec_from_sphere_to_observer)

    if norm_light == 0 or norm_observer == 0:
        raise ValueError("Input vectors cannot have zero length.")

    unit_vec_light = vec_from_sphere_to_light / norm_light
    unit_vec_observer = vec_from_sphere_to_observer / norm_observer

    cos_alpha = np.dot(unit_vec_light, unit_vec_observer)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    term1 = np.sin(alpha)
    term2 = (np.pi - alpha) * np.cos(alpha)
    phase_function_value = (2 / (3 * np.pi)) * (term1 + term2)

    cross_sectional_area = np.pi * (radius ** 2)

    effective_brightness = (
        albedo *
        cross_sectional_area *
        phase_function_value
    )

    return effective_brightness



