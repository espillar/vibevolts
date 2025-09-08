import numpy as np

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
