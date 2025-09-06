import math
import scipy.integrate
import scipy.constants as const
import numpy as np
import plotly.graph_objects as go

def mag(x: float) -> float:
    """
    Calculates a magnitude value from a linear ratio.
    Uses the formula: magnitude = -2.5 * log10(ratio)
    """
    if x <= 0:
        raise ValueError("Input for mag() must be positive.")
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
    exponent = (const.h * const.c) / (wav_m * const.k * temp_k)
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
        temperature: The temperature of the blackbody in Kelvin.
        lambda_short: The short wavelength of the band in microns.
        lambda_long: The long wavelength of the band in microns.

    Returns:
        The integrated spectral radiance in units of
        Watts / (m^2 * steradian).
    """
    if lambda_short >= lambda_long:
        raise ValueError("lambda_short must be less than lambda_long.")
    lambda_short_m = lambda_short * 1e-6
    lambda_long_m = lambda_long * 1e-6
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
        temperature: The temperature of the blackbody in Kelvin.

    Returns:
        The total radiated power per unit area in W / m^2.
    """
    return const.sigma * (temperature**4)

def plot_blackbody_spectrum(temperature: float):
    """
    Plots the spectral radiance of a blackbody from
    0.5 to 30 microns.

    Args:
        temperature: The temperature of the blackbody in Kelvin.
    """
    wavelengths_microns = np.geomspace(0.5, 30, 500)
    wavelengths_m = wavelengths_microns * 1e-6
    spectral_radiance = [_planck_law(w, temperature) * 1e-6 for w in wavelengths_m]
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
        temperature: The temperature of the blackbody in Kelvin.
    """
    wavelengths_microns = np.linspace(0.1, 1.0, 500)
    wavelengths_m = wavelengths_microns * 1e-6
    spectral_radiance = [_planck_law(w, temperature) * 1e-6 for w in wavelengths_m]
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
