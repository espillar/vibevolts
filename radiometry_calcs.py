import math
import scipy.constants as const
from constants import ARCSEC


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
