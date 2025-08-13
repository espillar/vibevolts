# radiometry.py

"""
This module contains radiometric data for standard
astronomical filters and related conversion functions.

Disclaimer: The values provided in this module are compiled
from various sources and should be considered representative.
For critical applications, these values should be verified
against the primary literature for the specific instrument
and conditions of interest.
"""
import math
import scipy.integrate
import scipy.constants as const
import numpy as np
import plotly.graph_objects as go

# --- Module Contents ---
#
# Constants:
#   FILTER_DATA: Dictionary with radiometric data for various
#                astronomical filters.
#   AU_M: The astronomical unit in meters.
#   RSUN_M: The radius of the Sun in meters.
#
# Functions:
#   mag(x): Converts a linear flux ratio to a magnitude.
#   amag(x): Converts a magnitude to a linear flux ratio.
#   blackbody_flux(temp, l_short, l_long): Computes the
#       integrated blackbody radiance over a wavelength band.
#   stefan_boltzmann_law(temp): Calculates total power per
#       unit area using the Stefan-Boltzmann law.
#   plot_blackbody_spectrum(temp): Plots the blackbody
#       spectrum from 0.5 to 30 microns.
#   plot_blackbody_spectrum_visible_nir(temp): Plots the
#       blackbody spectrum from 0.1 to 1 micron.
#
# -------------------------

# --- Physical Constants ---
AU_M = const.au  # Astronomical Unit in meters
RSUN_M = 6.957e8 # Radius of the Sun in meters

# --- Radiometric Data ---

# Data is compiled for standard astronomical filters.
#
# Magnitudes are apparent magnitudes in the AB or Vega system.
# Sky brightness is for a dark site at new moon, in
# magnitudes per square arcsecond. For space-based IR
# telescopes, this is the zodiacal + telescope background.
# For ground-based IR, it's dominated by thermal emission.
# Wavelengths and bandwidths are in nanometers (nm).
# Zero point is the photon flux for a 0-magnitude
# object in photons per second per square meter.

FILTER_DATA = {
    # Johnson-Cousins UBVRI Filters
    'U': {
        'sun': -26.03,
        'sky': 22.0,
        'central_wavelength': 365.0,
        'bandwidth': 66.0,
        'zero_point': 4.96e9,
    },
    'B': {
        'sun': -26.13,
        'sky': 22.7,
        'central_wavelength': 445.0,
        'bandwidth': 94.0,
        'zero_point': 1.36e10,
    },
    'V': {
        'sun': -26.78,
        'sky': 21.9,
        'central_wavelength': 551.0,
        'bandwidth': 88.0,
        'zero_point': 8.79e9,
    },
    'R': {
        'sun': -27.11,
        'sky': 21.0,
        'central_wavelength': 658.0,
        'bandwidth': 138.0,
        'zero_point': 9.74e9,
    },
    'I': {
        'sun': -27.47,
        'sky': 20.0,
        'central_wavelength': 806.0,
        'bandwidth': 149.0,
        'zero_point': 7.11e9,
    },
    # Near-Infrared JHK Filters
    'J': {
        'sun': -27.91,
        'sky': 16.0,
        'central_wavelength': 1235.0,
        'bandwidth': 162.0,
        'zero_point': 3.15e9,
    },
    'H': {
        'sun': -28.25,
        'sky': 14.0,
        'central_wavelength': 1646.0,
        'bandwidth': 251.0,
        'zero_point': 2.35e9,
    },
    'K': {
        'sun': -28.29,
        'sky': 13.0,
        'central_wavelength': 2159.0,
        'bandwidth': 262.0,
        'zero_point': 1.17e9,
    },
    # SDSS Filters
    'g': {
        'sun': -26.47,
        'sky': 21.8,
        'central_wavelength': 477.0,
        'bandwidth': 137.9,
        'zero_point': 1.582e10,
    },
    'r': {
        'sun': -26.93,
        'sky': 20.8,
        'central_wavelength': 623.1,
        'bandwidth': 138.2,
        'zero_point': 1.214e10,
    },
    'i': {
        'sun': -27.05,
        'sky': 20.2,
        'central_wavelength': 762.5,
        'bandwidth': 153.5,
        'zero_point': 1.103e10,
    },
    'z': {
        'sun': -27.07,
        'sky': 19.0,
        'central_wavelength': 913.4,
        'bandwidth': 140.9,
        'zero_point': 8.455e9,
    },
    # Ground-Based Mid-IR Filters
    'L': {
        'sun': None, # Too bright, not a standard measurement
        'sky': 3.5,  # Dominated by thermal background
        'central_wavelength': 3500.0,
        'bandwidth': 600.0,
        'zero_point': 9.4e9,
    },
    'M': {
        'sun': None, # Too bright, not a standard measurement
        'sky': 0.0,  # Dominated by thermal background
        'central_wavelength': 4800.0,
        'bandwidth': 600.0,
        'zero_point': 6.8e9,
    },
    'N': {
        'sun': None, # Too bright, not a standard measurement
        'sky': -6.0, # Dominated by thermal background
        'central_wavelength': 10200.0,
        'bandwidth': 5000.0,
        'zero_point': 2.7e10,
    },
    # JWST MIRI Filters (5-25 microns)
    'F560W': {
        'sun': None,  # Too bright to be measured
        'sky': 24.4,
        'central_wavelength': 5600.0,
        'bandwidth': 1000.0,
        'zero_point': 9.78e9,
    },
    'F770W': {
        'sun': None,  # Too bright to be measured
        'sky': 24.4,
        'central_wavelength': 7700.0,
        'bandwidth': 1950.0,
        'zero_point': 1.39e10,
    },
    'F1000W': {
        'sun': None,  # Too bright to be measured
        'sky': 23.6,
        'central_wavelength': 10000.0,
        'bandwidth': 1800.0,
        'zero_point': 9.86e9,
    },
    'F1500W': {
        'sun': None,  # Too bright to be measured
        'sky': 21.9,
        'central_wavelength': 15000.0,
        'bandwidth': 2940.0,
        'zero_point': 1.08e10,
    },
    'F2550W': {
        'sun': None,  # Too bright to be measured
        'sky': 19.4,
        'central_wavelength': 25500.0,
        'bandwidth': 4050.0,
        'zero_point': 8.70e9,
    }
}

# --- Conversion Functions ---

def mag(x: float) -> float:
    """
    Calculates a magnitude value from a linear ratio.
    Uses the formula: magnitude = -2.5 * log10(ratio)
    """
    if x <= 0:
        raise ValueError(
            "Input for mag() must be positive."
        )
    return -2.5 * math.log10(x)

def amag(x: float) -> float:
    """
    Calculates the linear ratio from a magnitude value.
    This is the inverse of the mag() function.
    Uses the formula: ratio = 10**(-0.4 * magnitude)
    """
    return 10**(-0.4 * x)

def _planck_law(wav_m: float, temp_k: float) -> float:
    """
    Helper function for Planck's law for spectral radiance.
    Args:
        wav_m: Wavelength in meters.
        temp_k: Temperature in Kelvin.
    Returns:
        Spectral radiance in W / (m^2 * sr * m).
    """
    if wav_m <= 0:
        return 0.0
    # Use const.h, const.c, const.k from scipy.constants
    exponent = (const.h * const.c) / (wav_m * const.k * temp_k)
    # Avoid overflow for large exponents
    if exponent > 700:
        return 0.0
    numerator = 2.0 * const.h * const.c**2
    denominator = (wav_m**5) * (math.expm1(exponent))
    if denominator == 0:
        return 0.0
    return numerator / denominator

def blackbody_flux(
    temperature: float,
    lambda_short: float,
    lambda_long: float
) -> float:
    """
    Numerically computes the integrated spectral radiance of
    a blackbody over a given wavelength band.

    Requires the scipy library.

    Args:
        temperature: The temperature of the blackbody in
                     Kelvin.
        lambda_short: The short wavelength of the band
                      in microns.
        lambda_long: The long wavelength of the band
                     in microns.

    Returns:
        The integrated spectral radiance in units of
        Watts / (m^2 * steradian).
    """
    if lambda_short >= lambda_long:
        raise ValueError(
            "lambda_short must be less than lambda_long."
        )
    lambda_short_m = lambda_short * 1e-6
    lambda_long_m = lambda_long * 1e-6
    # Integrate Planck's law over the specified wavelength range
    integrated_radiance, _ = scipy.integrate.quad(
        lambda wav: _planck_law(wav, temperature),
        lambda_short_m,
        lambda_long_m
    )
    return integrated_radiance

def stefan_boltzmann_law(temperature: float) -> float:
    """
    Calculates the total power radiated per unit area by a
    blackbody using the Stefan-Boltzmann law.

    Args:
        temperature: The temperature of the blackbody in
                     Kelvin.

    Returns:
        The total radiated power per unit area in W / m^2.
    """
    # Use const.sigma, which is an alias for const.Stefan_Boltzmann
    return const.sigma * (temperature**4)

def plot_blackbody_spectrum(temperature: float):
    """
    Plots the spectral radiance of a blackbody from
    0.5 to 30 microns.

    Args:
        temperature: The temperature of the blackbody in
                     Kelvin.
    """
    # Generate wavelengths on a logarithmic scale for a better plot
    wavelengths_microns = np.geomspace(0.5, 30, 500)
    wavelengths_m = wavelengths_microns * 1e-6
    # Calculate spectral radiance, converting to per-micron units
    spectral_radiance = [
        _planck_law(w, temperature) * 1e-6
        for w in wavelengths_m
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelengths_microns,
        y=spectral_radiance,
        mode='lines',
        name=f'{temperature} K Blackbody'
    ))
    fig.update_layout(
        title_text=f'Blackbody Spectrum ({temperature} K)',
        xaxis_title="Wavelength (microns)",
        yaxis_title="Spectral Radiance (W / m^2 / sr / micron)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )
    fig.show()

def plot_blackbody_spectrum_visible_nir(temperature: float):
    """
    Plots the spectral radiance of a blackbody from
    0.1 to 1 micron.

    Args:
        temperature: The temperature of the blackbody in
                     Kelvin.
    """
    # Generate wavelengths on a linear scale for this range
    wavelengths_microns = np.linspace(0.1, 1.0, 500)
    wavelengths_m = wavelengths_microns * 1e-6
    # Calculate spectral radiance, converting to per-micron units
    spectral_radiance = [
        _planck_law(w, temperature) * 1e-6
        for w in wavelengths_m
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wavelengths_microns,
        y=spectral_radiance,
        mode='lines',
        name=f'{temperature} K Blackbody'
    ))
    fig.update_layout(
        title_text=f'Blackbody Spectrum ({temperature} K) - Visible/NIR',
        xaxis_title="Wavelength (microns)",
        yaxis_title="Spectral Radiance (W / m^2 / sr / micron)",
        template="plotly_white"
    )
    fig.show()


if __name__ == '__main__':
    import pprint

    print("--- Physical Constants ---")
    print(f"Astronomical Unit (AU): {AU_M:.3e} m")
    print(f"Radius of the Sun (R_sun): {RSUN_M:.3e} m")

    print("\n--- Radiometric Filter Data ---")
    pprint.pprint(FILTER_DATA)

    print("\n--- Example: r-band data ---")
    r_band_data = FILTER_DATA['r']
    print(f"Sun's magnitude: {r_band_data['sun']}")
    print(
        f"Sky brightness: {r_band_data['sky']} mag/arcsec^2"
    )
    print(
        f"Central Wavelength: "
        f"{r_band_data['central_wavelength']} nm"
    )
    print(f"Bandwidth: {r_band_data['bandwidth']} nm")
    print(
        f"Zero Point Photon Flux: "
        f"{r_band_data['zero_point']:.3e} photons/s/m^2"
    )

    print("\n--- Magnitude Conversion Examples ---")
    flux_ratio = 100.0
    calculated_mag = mag(flux_ratio)
    print(
        f"A flux ratio of {flux_ratio} corresponds to a "
        f"magnitude difference of {calculated_mag:.2f}."
    )

    mag_diff = 5.0
    calculated_ratio = amag(mag_diff)
    print(
        f"A magnitude difference of {mag_diff} "
        f"corresponds to a flux ratio of "
        f"{calculated_ratio:.1f}."
    )
    
    original_value = 250.0
    inverted_value = amag(mag(original_value))
    print(
        f"\nVerifying inverse: amag(mag({original_value}))"
        f" = {inverted_value:.1f}"
    )

    print("\n--- Blackbody Flux Calculation Example ---")
    sun_temp = 5778  # Kelvin
    # Convert filter nm values to microns for the function
    r_central = FILTER_DATA['r']['central_wavelength'] * 1e-3
    r_bw = FILTER_DATA['r']['bandwidth'] * 1e-3
    r_wav_short = r_central - r_bw / 2.0
    r_wav_long = r_central + r_bw / 2.0
    
    flux = blackbody_flux(sun_temp, r_wav_short, r_wav_long)
    
    print(
        f"Integrated spectral radiance of a {sun_temp}K "
        f"blackbody\n"
        f"between {r_wav_short:.4f} and "
        f"{r_wav_long:.4f} microns:"
    )
    print(f"{flux:.3e} W / (m^2 * sr)")

    print("\n--- Stefan-Boltzmann Law Example ---")
    total_flux = stefan_boltzmann_law(sun_temp)
    print(
        f"Total power radiated by a {sun_temp}K blackbody: "
        f"{total_flux:.3e} W / m^2"
    )

    print("\n--- Blackbody Spectrum Plot Examples ---")
    # These will display plots in a browser or notebook
    # plot_blackbody_spectrum(sun_temp)
    # plot_blackbody_spectrum_visible_nir(sun_temp)
    print("Plotting functions are commented out to prevent automatic display.")
    print("Uncomment the lines above to generate the plots.")
