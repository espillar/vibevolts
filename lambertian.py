import numpy as np


def _print_lambertian_debug(
    debug: int,
    angle_in: np.ndarray,
    alpha: np.ndarray,
    phase: np.ndarray,
    cross_sect: np.ndarray,
    eff_cs: np.ndarray,
    albedo: np.ndarray,
    radius: np.ndarray,
    base_brightness,
    emitted: np.ndarray,
) -> None:
    """Prints level-1 or level-2 debug tables for lambertiansphere."""
    n = len(albedo)
    limit = 5
    scalar_base = np.ndim(base_brightness) == 0

    if debug == 1:
        print("\n--- Debug Info: lambertiansphere ---")
        print("Effective Cross Section: ", eff_cs)
        header = (
            f"{'Index':<5} {'Phase Angle (rad)':<18} {'Albedo':<10} "
            f"{'Radius (m)':<12} {'Base Brightness':<18} {'Emitted Brightness':<22}"
        )
        print(header)
        print("-" * len(header))
        for i in range(n):
            if n > 2 * limit and limit <= i < n - limit:
                if i == limit:
                    print(f"{'.':<5} {'...':<18} {'...':<10} {'...':<12} {'...':<18} {'...':<22}")
                continue
            b_val = base_brightness if scalar_base else base_brightness[i]
            print(
                f"{i:<5} {angle_in[i]:<18.4e} {albedo[i]:<10.4f} "
                f"{radius[i]:<12.4e} {b_val:<18.4e} {emitted[i]:<22.4e}"
            )
        print("-----------------------------------\n")
    elif debug == 2:
        print("\n--- Detailed Debug Info: lambertiansphere ---")
        header = (
            f"{'Index':<5} {'Input Angle':<15} {'Clipped Alpha':<15} "
            f"{'Phase Func Val':<15} {'Cross Sect Area':<15} {'Effective CS':<15} "
            f"{'Albedo':<10} {'Radius (m)':<12} {'Base Brightness':<18} {'Emitted Brightness':<22}"
        )
        print(header)
        print("-" * len(header))
        for i in range(n):
            if n > 2 * limit and limit <= i < n - limit:
                if i == limit:
                    print(f"{'.':<5} " + " ".join(f"{'...':<15}" for _ in range(9)))
                continue
            b_val = base_brightness if scalar_base else base_brightness[i]
            print(
                f"{i:<5} {angle_in[i]:<15.4e} {alpha[i]:<15.4e} {phase[i]:<15.4e} "
                f"{cross_sect[i]:<15.4e} {eff_cs[i]:<15.4e} {albedo[i]:<10.4f} "
                f"{radius[i]:<12.4e} {b_val:<18.4e} {emitted[i]:<22.4e}"
            )
        print("-------------------------------------------\n")


def lambertiansphere(
    angle_light_observer: np.ndarray,
    albedo: np.ndarray,
    radius: np.ndarray,
    base_brightness,
    debug: int = 0,
) -> np.ndarray:
    """
    Calculates emitted surface brightness for N Lambertian spheres from phase angles,
    albedos, radii, and incident base_brightness. Divide by pi * d^2 for apparent flux.
    """
    if not np.all((albedo >= 0.0) & (albedo <= 1.0)):
        raise ValueError("All albedo values must be between 0.0 and 1.0.")
    if np.any(radius < 0):
        raise ValueError("Radius cannot be negative.")
    if not isinstance(angle_light_observer, np.ndarray) or angle_light_observer.ndim != 1:
        raise ValueError("angle_light_observer must be a 1D NumPy array.")

    alpha = np.clip(angle_light_observer, 0, np.pi)
    phase = (2.0 / (3.0 * np.pi)) * (np.sin(alpha) + (np.pi - alpha) * np.cos(alpha))
    cross_sect = np.pi * (radius ** 2)
    eff_cs = albedo * cross_sect * phase
    emitted = base_brightness * eff_cs

    if debug in (1, 2):
        _print_lambertian_debug(
            debug, angle_light_observer, alpha, phase, cross_sect,
            eff_cs, albedo, radius, base_brightness, emitted,
        )
    return emitted


def simple_lambertian(
    diameter: float,
    distance: float,
    albedo: float,
    angle: float,
    base_brightness: float,
) -> float:
    """Scalar convenience wrapper around lambertiansphere() dividing by pi * distance^2."""
    if distance <= 0:
        raise ValueError("Distance must be positive.")
    emitted = lambertiansphere(
        np.array([angle], dtype=float),
        np.array([albedo], dtype=float),
        np.array([diameter / 2.0], dtype=float),
        np.array([base_brightness], dtype=float),
    )[0]
    return float(emitted / (np.pi * distance ** 2))


def includedAngle(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    """Calculates included angle (rad) between pairs of (N, 3) vectors."""
    if vectors1.shape != vectors2.shape:
        raise ValueError("Input arrays must have the same shape.")
    if vectors1.shape[1] != 3:
        raise ValueError("Input vectors must be 3-dimensional.")

    from minimalsimulation import safe_normalize
    dots = np.einsum('ij,ij->i', safe_normalize(vectors1, axis=1), safe_normalize(vectors2, axis=1))
    return np.arccos(np.clip(dots, -1.0, 1.0))
