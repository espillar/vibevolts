import numpy as np
import plotly.graph_objects as go
# Import necessary data and functions from other files, including 'mag' for magnitude conversion
from radiometry_data import FILTER_DATA
from radiometry_calcs import amag, mag
from lambertian import lambertiansphere


def plot_satellite_brightness():
    """
    Plots the apparent V-band photon flux and magnitude of satellites
    with various diameters over a range of distances.

    This function calculates and plots two figures: one showing photon flux
    on a log-log scale, and another showing apparent magnitude (V-band)
    on a linear y-axis, for satellites of different sizes illuminated by
    the sun at a 90-degree phase angle.
    """
    # --- Satellite and Illumination Parameters ---
    DIAMETER_VALUES = [0.1, 0.4, 1.0, 3.0, 10.0]  # meters
    SATELLITE_ALBEDO = 0.3  # Typical albedo for satellite materials
    PHASE_ANGLE = np.pi / 2.0  # 90 degrees, as requested
    
    # Calculate the incident photon flux (photons / s / m^2) in the V-band.
    V_MAG_SUN = FILTER_DATA['V']['sun']
    # Zero-point flux is the flux of a 0-magnitude object in the V-band
    V_ZERO_POINT_FLUX = FILTER_DATA['V']['zero_point'] 
    
    # Incident photon flux: Flux = ZeroPoint * 10^(-0.4 * mag)
    INCIDENT_PHOTON_FLUX = amag(V_MAG_SUN) * V_ZERO_POINT_FLUX 

    # --- Distance Range ---
    # Generate 500 points from 1,000 km to 400,000 km on a logarithmic scale
    distances_km = np.geomspace(1000, 400000, 500)
    distances_m = distances_km * 1000  # Convert to meters for the calculation

    # --- Initialize Plot Figures ---
    fig_flux = go.Figure()
    fig_mag = go.Figure()

    # Calculate and plot a curve for each specified diameter
    for diameter in DIAMETER_VALUES:
        # 1. Calculate Apparent Brightness (Photon Flux)
        emitted_brightness = lambertiansphere(
            angle_light_observer=np.array([PHASE_ANGLE]),
            albedo=np.array([SATELLITE_ALBEDO]),
            radius=np.array([diameter / 2.0]),
            base_brightness=np.array([INCIDENT_PHOTON_FLUX])
        )[0]
        brightness_values = emitted_brightness / (np.pi * distances_m ** 2)

        # 2. Convert Photon Flux to Apparent Magnitude (V-band)
        # Magnitude = -2.5 * log10(Flux / ZeroPointFlux)
        magnitude_values = [
            mag(b / V_ZERO_POINT_FLUX) for b in brightness_values
        ]

        # 3. Add trace to FLUX plot
        fig_flux.add_trace(go.Scatter(
            x=distances_km,
            y=brightness_values,
            mode='lines',
            name=f'Diameter = {diameter} m'
        ))

        # 4. Add trace to MAGNITUDE plot
        fig_mag.add_trace(go.Scatter(
            x=distances_km,
            y=magnitude_values,
            mode='lines',
            name=f'Diameter = {diameter} m'
        ))

    # --- Configure FLUX Plot (Log-Log) ---
    fig_flux.update_layout(
        title_text='Apparent V-Band Photon Flux vs. Distance for Various Diameters',
        xaxis_title="Distance from Observer (km)",
        yaxis_title="Apparent Photon Flux (photons / s / m\u00b2)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )

    # --- Configure MAGNITUDE Plot (Log X, Reversed Linear Y) ---
    fig_mag.update_layout(
        title_text='Apparent V-Band Magnitude vs. Distance for Various Diameters',
        xaxis_title="Distance from Observer (km)",
        yaxis_title="Apparent V-Band Magnitude",
        xaxis_type="log",
        yaxis_type="linear",
        template="plotly_white",
        # Reverse y-axis so brighter objects (lower magnitude) are at the top
        yaxis={'autorange': "reversed"} 
    )
    
    # Display both figures
    fig_flux.show()
    fig_mag.show()
plot_satellite_brightness()
