import math
import numpy as np
import scipy.constants as const
from constants import ARCSEC


OPTICAL_NIR_BANDS = ('U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'g', 'r', 'i', 'z')


# --- Physical Constants ---
AU_M = const.au
RSUN_M = 6.957e8


# --- Filter Photometric Data ---
# Magnitudes are apparent magnitudes; sky/space are mag / arcsec^2;
# wavelengths/bandwidths in nm; zero_point in photons / s / m^2.
FILTER_DATA = {
    'U': {'sun': -26.03, 'sky': 22.0, 'space': 22.9, 'central_wavelength': 365.0, 'bandwidth': 66.0, 'zero_point': 4.96e9},
    'B': {'sun': -26.13, 'sky': 22.7, 'space': 23.3, 'central_wavelength': 445.0, 'bandwidth': 94.0, 'zero_point': 1.36e10},
    'V': {'sun': -26.78, 'sky': 21.9, 'space': 23.0, 'central_wavelength': 551.0, 'bandwidth': 88.0, 'zero_point': 8.79e9},
    'R': {'sun': -27.11, 'sky': 21.0, 'space': 22.0, 'central_wavelength': 658.0, 'bandwidth': 138.0, 'zero_point': 9.74e9},
    'I': {'sun': -27.47, 'sky': 20.0, 'space': 21.2, 'central_wavelength': 806.0, 'bandwidth': 149.0, 'zero_point': 7.11e9},
    'J': {'sun': -27.91, 'sky': 16.0, 'space': 21.5, 'central_wavelength': 1235.0, 'bandwidth': 162.0, 'zero_point': 3.15e9},
    'H': {'sun': -28.25, 'sky': 14.0, 'space': 20.5, 'central_wavelength': 1646.0, 'bandwidth': 251.0, 'zero_point': 2.35e9},
    'K': {'sun': -28.29, 'sky': 13.0, 'space': 19.5, 'central_wavelength': 2159.0, 'bandwidth': 262.0, 'zero_point': 1.17e9},
    'g': {'sun': -26.47, 'sky': 21.8, 'space': 23.0, 'central_wavelength': 477.0, 'bandwidth': 137.9, 'zero_point': 1.582e10},
    'r': {'sun': -26.93, 'sky': 20.8, 'space': 22.2, 'central_wavelength': 623.1, 'bandwidth': 138.2, 'zero_point': 1.214e10},
    'i': {'sun': -27.05, 'sky': 20.2, 'space': 21.5, 'central_wavelength': 762.5, 'bandwidth': 153.5, 'zero_point': 1.103e10},
    'z': {'sun': -27.07, 'sky': 19.0, 'space': 20.8, 'central_wavelength': 913.4, 'bandwidth': 140.9, 'zero_point': 8.455e9},
    'L': {'sun': None, 'sky': 3.5, 'space': 20.0, 'central_wavelength': 3500.0, 'bandwidth': 600.0, 'zero_point': 9.4e9},
    'M': {'sun': None, 'sky': 0.0, 'space': 19.0, 'central_wavelength': 4800.0, 'bandwidth': 600.0, 'zero_point': 6.8e9},
    'N': {'sun': None, 'sky': -6.0, 'space': 17.0, 'central_wavelength': 10200.0, 'bandwidth': 5000.0, 'zero_point': 2.7e10},
    'F560W': {'sun': None, 'sky': 24.4, 'central_wavelength': 5600.0, 'bandwidth': 1000.0, 'zero_point': 9.78e9},
    'F770W': {'sun': None, 'sky': 24.4, 'central_wavelength': 7700.0, 'bandwidth': 1950.0, 'zero_point': 1.39e10},
    'F1000W': {'sun': None, 'sky': 23.6, 'central_wavelength': 10000.0, 'bandwidth': 1800.0, 'zero_point': 9.86e9},
    'F1500W': {'sun': None, 'sky': 21.9, 'central_wavelength': 15000.0, 'bandwidth': 2940.0, 'zero_point': 1.08e10},
    'F2550W': {'sun': None, 'sky': 19.4, 'central_wavelength': 25500.0, 'bandwidth': 4050.0, 'zero_point': 8.70e9},
}


def mag(x: float) -> float:
    """Converts a linear flux ratio into a stellar magnitude."""
    if x <= 0:
        raise ValueError("Input for mag() must be positive.")
    return -2.5 * math.log10(x)


def amag(x: float) -> float:
    """Converts a stellar magnitude into a linear flux ratio."""
    return 10 ** (-0.4 * x)


def fluxes(band: str) -> tuple[float, float, float]:
    """
    Returns (sun, space, sky) photon fluxes for a filter band:
    sun in photons/s/m^2, space and sky in photons/s/sr/m^2.
    """
    x = FILTER_DATA[band]
    zp = x['zero_point']
    sun = amag(x['sun']) * zp if x.get('sun') is not None else 0.0
    space_mag = x.get('space', x.get('sky'))
    space = amag(space_mag) * zp / (ARCSEC**2) if space_mag is not None else 0.0
    sky = amag(x['sky']) * zp / (ARCSEC**2) if x.get('sky') is not None else 0.0
    return (sun, space, sky)


def _unit_vec(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("Input direction vector must be non-zero.")
    return arr / norm


def _airmass(cos_z: float) -> float:
    cos_clamped = max(float(cos_z), 0.0)
    return 1.0 / math.sqrt(1.0 - 0.96 * (1.0 - cos_clamped * cos_clamped))


def _scattered_nl(
    p_dot_src: float,
    x_src: float,
    mag_plus_1657: float,
    w_r: float,
    w_m: float,
    k_ext: float,
    xz: float,
) -> float:
    """Krisciunas & Schaefer scattered surface brightness in nanoLamberts."""
    rho_deg = math.degrees(math.acos(max(-1.0, min(1.0, p_dot_src))))
    f_rho = (
        (10.0 ** 5.36) * w_r * (1.06 + p_dot_src * p_dot_src)
        + (10.0 ** (6.15 - rho_deg / 40.0)) * w_m
    )
    i_star = 10.0 ** (-0.4 * mag_plus_1657)
    return (
        f_rho
        * i_star
        * (10.0 ** (-0.4 * k_ext * x_src))
        * (1.0 - 10.0 ** (-0.4 * k_ext * xz))
    )


def ground_sky_brightness(
    zenith_vec: np.ndarray,
    pointing_vec: np.ndarray,
    sun_vec: np.ndarray,
    moon_vec: np.ndarray,
    band: str | None = None,
    return_mag: bool = False,
) -> float | dict[str, float]:
    """
    Computes ground-based sky background brightness due to airglow, scattered
    moonlight (Krisciunas & Schaefer 1991), and daytime/twilight sunlight
    (Schaefer 1993; Patat et al. 2006), scaled across wavelength (Noll et al.
    2012; Jones et al. 2013).

    Args:
        zenith_vec: 3-element local zenith direction vector.
        pointing_vec: 3-element telescope line-of-sight pointing vector.
        sun_vec: 3-element vector from observatory toward the Sun.
        moon_vec: 3-element vector from observatory toward the Moon.
        band: Filter name (e.g. 'V') or None to return all 12 optical/NIR bands.
        return_mag: If True, returns surface brightness in mag/arcsec^2;
            otherwise returns photon radiance in photons / s / m^2 / sr.
    """
    if band is not None and band not in OPTICAL_NIR_BANDS:
        raise ValueError(f"Unsupported band '{band}'. Expected one of {OPTICAL_NIR_BANDS}.")

    z_hat = _unit_vec(zenith_vec)
    p_hat = _unit_vec(pointing_vec)
    s_hat = _unit_vec(sun_vec)
    m_hat = _unit_vec(moon_vec)

    cos_z = max(0.0, min(1.0, float(np.dot(p_hat, z_hat))))
    cos_zs = max(-1.0, min(1.0, float(np.dot(s_hat, z_hat))))
    cos_zm = max(-1.0, min(1.0, float(np.dot(m_hat, z_hat))))
    p_dot_s = max(-1.0, min(1.0, float(np.dot(p_hat, s_hat))))
    p_dot_m = max(-1.0, min(1.0, float(np.dot(p_hat, m_hat))))

    zs_deg = math.degrees(math.acos(cos_zs))
    alpha_deg = math.degrees(math.acos(max(-1.0, min(1.0, -float(np.dot(s_hat, m_hat))))))
    m_moon = -12.73 + 0.026 * alpha_deg + 4e-9 * (alpha_deg ** 4)
    xz = _airmass(cos_z)
    nl_to_ratio = 34.08 * math.exp(20.7233)

    bands = (band,) if band is not None else OPTICAL_NIR_BANDS
    out: dict[str, float] = {}
    for b in bands:
        fdata = FILTER_DATA[b]
        lam_nm = fdata['central_wavelength']
        w_r = (lam_nm / 550.0) ** -4.0
        w_m = (lam_nm / 550.0) ** -1.3
        k_ext = 0.11 * w_r + 0.05 * w_m

        r_dark = amag(fdata['sky']) * (10.0 ** (-0.4 * k_ext * (xz - 1.0))) * xz

        b_moon_nl = 0.0
        if cos_zm > 0.0:
            b_moon_nl = _scattered_nl(
                p_dot_m, _airmass(cos_zm), m_moon + 16.57, w_r, w_m, k_ext, xz
            )

        b_sun_nl = 0.0
        if zs_deg < 108.0:
            b_sun_horizon = _scattered_nl(
                p_dot_s, _airmass(max(cos_zs, 0.0)), -26.78 + 16.57, w_r, w_m, k_ext, xz
            )
            b_sun_nl = (
                b_sun_horizon
                if zs_deg <= 90.0
                else b_sun_horizon * (10.0 ** (-0.4 * 0.75 * (zs_deg - 90.0)))
            )

        r_tot = r_dark + (b_moon_nl + b_sun_nl) / nl_to_ratio
        out[b] = mag(r_tot) if return_mag else r_tot * fdata['zero_point'] / (ARCSEC ** 2)

    return out[band] if band is not None else out
