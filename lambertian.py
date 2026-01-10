import numpy as np

def lambertiansphere(
    vec_from_sphere_to_light: np.ndarray,
    vec_from_sphere_to_observer: np.ndarray,
    albedo: np.ndarray,
    radius: np.ndarray,
    base_brightness: np.ndarray
) -> np.ndarray:
    """
    Calculates the apparent brightness of multiple Lambertian spheres from a single light source.

    This function determines the apparent brightness of diffusely reflecting spheres
    based on the angle between the light source and the observer, the spheres'
    albedos (reflectivity), their sizes, and the distance to the observer.

    Args:
        vec_from_sphere_to_light: A single (3,) NumPy array representing the
            vector from the sphere's position to the light source. NOT normalized.
        vec_from_sphere_to_observer: An (N, 3) NumPy array where each row is a
            vector from a sphere to the observer. The magnitude of this vector
            is the distance. NOT normalized.
        albedo: A 1D NumPy array of shape (N,) with the fraction of incident
            light that is reflected for each sphere (0.0 to 1.0).
        radius: A 1D NumPy array of shape (N,) with the radius of each sphere
            in meters.
        base_brightness: A 1D NumPy array of shape (N,) with the incident
            flux or brightness of the light source at each sphere's location.

    Returns:
        A 1D NumPy array of shape (N,) containing the apparent brightness
        for each sphere, e.g. in Watts per square meter.
    """
    if vec_from_sphere_to_light.shape != (3,):
        raise ValueError(f"vec_from_sphere_to_light must be a single vector of shape (3,), but got {vec_from_sphere_to_light.shape}")
    if not np.all((albedo >= 0.0) & (albedo <= 1.0)):
        raise ValueError("All albedo values must be between 0.0 and 1.0.")
    if np.any(radius < 0):
        raise ValueError("Radius cannot be negative.")

    norm_light = np.linalg.norm(vec_from_sphere_to_light)
    norm_observer = np.linalg.norm(vec_from_sphere_to_observer, axis=1)

    # To avoid division by zero, we'll suppress warnings and handle NaNs later
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vec_light = vec_from_sphere_to_light / norm_light
        unit_vec_observer = vec_from_sphere_to_observer / norm_observer[:, np.newaxis]

    # Dot product of the single light vector with each observer vector
    cos_alpha = np.einsum('j,ij->i', unit_vec_light, unit_vec_observer)

    # Handle cases where dot product resulted in NaN from zero-length vectors
    cos_alpha = np.nan_to_num(cos_alpha)

    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    term1 = np.sin(alpha)
    term2 = (np.pi - alpha) * np.cos(alpha)
    phase_function_value = (2 / (3 * np.pi)) * (term1 + term2)

    cross_sectional_area = np.pi * (radius ** 2)

    effective_cross_section = (
        albedo *
        cross_sectional_area *
        phase_function_value
    )

    # The apparent brightness is the incident brightness multiplied
    # by the effective reflecting area, with the resulting light
    # distributed according to the inverse square law.
    with np.errstate(divide='ignore', invalid='ignore'):
        apparent_brightness = (
            (base_brightness * effective_cross_section) /
            (np.pi * norm_observer ** 2)
        )

    # Ensure that entries corresponding to zero-norm vectors have zero brightness
    invalid_mask = (norm_light == 0) | (norm_observer == 0)
    apparent_brightness[invalid_mask] = 0.0

    return apparent_brightness


def simple_lambertian(
    diameter: float,
    distance: float,
    albedo: float,
    angle: float,
    base_brightness: float
) -> float:
    """
    Calculates the apparent brightness of a Lambertian sphere.
    See Also lambertiansphere, which works with complete vectors
    of objects in a more code friendly way!

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


