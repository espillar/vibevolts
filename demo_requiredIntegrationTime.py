import numpy as np
import plotly.graph_objects as go
from constants import *
from detector import *

def demo_requiredIntegrationTime():
    """
    Demonstrates the requiredIntegrationTime function.
    Returns a graph
    """
    # Create a detector
    n = 1
    band = 'V'
    fov = 30 * DEGREE
    ifov = 1 * ARCSEC
    aper = 1
    qe = 1.0
    photfrac = 1.
    

    filt,detector = makeDetector(n, band, fov,ifov, aper, qe, photfrac)

    print(detector)
    
    # Print detector specs
    print("Detector Specifications:")
    print(f"  Aperture: {detector[0, APERTURE_IDX]} m")
    print(f"  Pixel Size: {detector[0, PIXEL_SIZE_IDX]} rad")
    print(f"  QE: {detector[0, QE_IDX]}")
    print(f"  Photometric Efficiency: {detector[0, PHOT_EFF_IDX]}")
    print(f"  Pixels: {detector[0, PIXELS_IDX]}")
    print(f"  Solar Exclusion: {detector[0, SOLAR_EXCL_IDX]} rad")
    print(f"  Lunar Exclusion: {detector[0, LUNAR_EXCL_IDX]} rad")
    print(f"  Earth Exclusion: {detector[0, EARTH_EXCL_IDX]} rad")
    print(f"  Sky Background: {detector[0, SKY_BACK_IDX]}")
    print(f"  Filter Band Cal: {filt[0]}")


    # Calculate required integration time for a range of limiting magnitudes
    limiting_mags = np.arange(15, 26)
    integration_times = []
    for mag in limiting_mags:
        integration_times.append(requiredIntegrationTime(mag, 10, filt, detector))

    # Create a plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=limiting_mags, y=np.array(integration_times).flatten(), mode='lines+markers'))
    fig.update_layout(
        title="Required Integration Time vs. Limiting Magnitude",
        xaxis_title="Limiting Magnitude",
        yaxis_title="Integration Time (s)",
    )
    fig.update_yaxes(type="log")
    return fig

if __name__ == '__main__':
    fig = demo_requiredIntegrationTime()
    fig.show()
