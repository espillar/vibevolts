import numpy as np

def lambertiansphere(
    angle_light_observer: np.ndarray,
    albedo: np.ndarray,
    radius: np.ndarray,
    base_brightness,
    debug: int = 0
) -> np.ndarray:
    """
    Calculates the emitted brightness from multiple lambertian spheres
    based on phase angle. This function computes the brightness emitted
    *from the sphere's surface* in the direction of the observer.
    The final apparent brightness at the observer's location must be
    calculated by dividing this result by pi * (distance_to_observer)^2
    to satisfy conservation of energy for a diffuse sphere.

    Args:
        angle_light_observer: A 1D NumPy array of shape (N,) with the
            phase angle in radians for each sphere. This is the angle
            between the light source and the observer as seen from
            the sphere's center.
        albedo: A 1D NumPy array of shape (N,) with the fraction of
            incident light that is reflected for each sphere (0.0 to 1.0).
        radius: A 1D NumPy array of shape (N,) with the radius of each
            sphere in meters.
        base_brightness: A 1D NumPy array of shape (N,) with the incident
            flux or brightness of the light source at each sphere's
            location.
        debug: An optional integer. If set to 1, a table of inputs and
            outputs will be printed before the function returns.
            Defaults to 0.

    Returns:
        A 1D NumPy array of shape (N,) containing the brightness emitted
        from each sphere's surface (e.g., in Watts per steradian per
        square meter). To get apparent brightness at the observer,
        divide by pi * (distance_to_observer)^2.
    """
    if not np.all((albedo >= 0.0) & (albedo <= 1.0)):
        raise ValueError("All albedo values must be between 0.0 and 1.0.")
    if np.any(radius < 0):
        raise ValueError("Radius cannot be negative.")
    if (not isinstance(angle_light_observer, np.ndarray) or
            angle_light_observer.ndim != 1):
        raise ValueError("angle_light_observer must be a 1D NumPy array.")

    # The phase angle is physically constrained to be between 0 and pi.
    # We clip the value to handle out-of-range inputs gracefully.
    alpha = np.clip(angle_light_observer, 0, np.pi)

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
    # by the effective reflecting area.
    # The division by distance squared will be handled by the caller,
    # as instructed by the user, to accommodate the 4*pi*r^2 factor.
    apparent_brightness = (base_brightness * effective_cross_section)

    if debug == 1:
        print("\n--- Debug Info: lambertiansphere ---")
        print("Effective Cross Section: ", effective_cross_section)
        num_spheres = len(albedo)
        h1 = f"{'Index':<5} {'Phase Angle (rad)':<18} {'Albedo':<10}"
        h2 = f"{'Radius (m)':<12} {'Base Brightness':<18} {'Emitted Brightness':<22}"
        header = h1 + " " + h2
        print(header)
        print("-" * len(header))

        display_limit = 5
        for i in range(num_spheres):
            if (num_spheres > 2 * display_limit and
                    display_limit <= i < num_spheres - display_limit):
                if i == display_limit:
                    p1 = f"{'.':<5} {'...':<18} {'...':<10}"
                    p2 = f"{'...':<12} {'...':<18} {'...':<22}"
                    print(p1 + " " + p2)
                continue

            brightness_val = (
                base_brightness[i]
                if np.ndim(base_brightness) > 0
                else base_brightness
            )
            r1 = f"{i:<5} {angle_light_observer[i]:<18.4e} {albedo[i]:<10.4f}"
            r2 = f"{radius[i]:<12.4e} {brightness_val:<18.4e}"
            r3 = f"{apparent_brightness[i]:<22.4e}"
            print(f"{r1} {r2} {r3}")
        print("-----------------------------------\n")
    elif debug == 2:
        print("\n--- Detailed Debug Info: lambertiansphere ---")
        num_spheres = len(albedo)
        header = (
            f"{'Index':<5} {'Input Angle':<15} {'Clipped Alpha':<15} "
            f"{'Phase Func Val':<15} {'Cross Sect Area':<15} "
            f"{'Effective CS':<15} {'Albedo':<10} {'Radius (m)':<12} "
            f"{'Base Brightness':<18} {'Emitted Brightness':<22}"
        )
        print(header)
        print("-" * len(header))

        display_limit = 5
        for i in range(num_spheres):
            if (num_spheres > 2 * display_limit and
                    display_limit <= i < num_spheres - display_limit):
                if i == display_limit:
                    dots = ["..."] * 9
                    print(f"{'.':<5} " + " ".join([f"{d:<15}" for d in dots]))
                continue

            brightness_val = (
                base_brightness[i]
                if np.ndim(base_brightness) > 0
                else base_brightness
            )
            print(
                f"{i:<5} {angle_light_observer[i]:<15.4e} {alpha[i]:<15.4e} "
                f"{phase_function_value[i]:<15.4e} "
                f"{cross_sectional_area[i]:<15.4e} "
                f"{effective_cross_section[i]:<15.4e} {albedo[i]:<10.4f} "
                f"{radius[i]:<12.4e} {brightness_val:<18.4e} "
                f"{apparent_brightness[i]:<22.4e}"
            )
        print("-------------------------------------------\n")

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


def includedAngle(
    vectors1: np.ndarray,
    vectors2: np.ndarray
) -> np.ndarray:
    """
    Calculates the included angle in radians between corresponding vectors
    in two input NumPy arrays.

    Args:
        vectors1 (np.ndarray): A NumPy array of shape (N, 3) representing the
                               first set of vectors.
        vectors2 (np.ndarray): A NumPy array of shape (N, 3) representing the
                               second set of vectors.

    Returns:
        np.ndarray: A 1D NumPy array of shape (N,) containing the included
                    angle in radians for each corresponding pair of vectors.
    """
    if vectors1.shape != vectors2.shape:
        raise ValueError("Input arrays must have the same shape.")
    if vectors1.shape[1] != 3:
        raise ValueError("Input vectors must be 3-dimensional.")

    from minimalsimulation import safe_normalize
    unit_vectors1 = safe_normalize(vectors1, axis=1)
    unit_vectors2 = safe_normalize(vectors2, axis=1)

    # Calculate the dot product of the normalized vectors
    dot_product = np.einsum('ij,ij->i', unit_vectors1, unit_vectors2)

    # Clip values to ensure they are within the valid range for arccos (-1 to 1)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle in radians
    angles = np.arccos(dot_product)

    return angles
