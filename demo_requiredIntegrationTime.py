import numpy as np
import plotly.graph_objects as go
from constellation import makeDetector, requiredIntegrationTime
from constants import *

def demo_requiredIntegrationTime():
    """
    Demonstrates the requiredIntegrationTime function.
    """
    # Create a detector
    n = 1
    band = 'V'
    fov = 1.0
    ifov = 5e-6
    aper = 0.1
    limitingmag = 20
    qe = 1.0
    photfrac = 0.7
    
    detector = makeDetector(n, band, fov, ifov, aper, limitingmag, qe, photfrac)

    # Print detector specs
    print("Detector Specifications:")
    print(f"  Aperture: {detector[APERTURE_IDX, 0]} m")
    print(f"  Pixel Size: {detector[PIXEL_SIZE_IDX, 0]} rad")
    print(f"  QE: {detector[QE_IDX, 0]}")
    print(f"  Photometric Efficiency: {detector[PHOT_EFF_IDX, 0]}")
    print(f"  Pixels: {detector[PIXELS_IDX, 0]}")
    print(f"  Solar Exclusion: {detector[SOLAR_EXCL_IDX, 0]} rad")
    print(f"  Lunar Exclusion: {detector[LUNAR_EXCL_IDX, 0]} rad")
    print(f"  Earth Exclusion: {detector[EARTH_EXCL_IDX, 0]} rad")
    print(f"  Sky Background: {detector[SKY_BACK_IDX, 0]}")
    print(f"  Filter Band: {detector[FILTER_BAND_IDX, 0]}")
    print(f"  Filter Band Cal: {detector[FILTER_BAND_CAL_IDX, 0]}")


    # Calculate required integration time for a range of limiting magnitudes
    limiting_mags = np.arange(15, 26)
    integration_times = []
    for mag in limiting_mags:
        integration_times.append(requiredIntegrationTime(mag, detector))

    # Create a plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=limiting_mags, y=np.array(integration_times).flatten(), mode='lines+markers'))
    fig.update_layout(
        title="Required Integration Time vs. Limiting Magnitude",
        xaxis_title="Limiting Magnitude",
        yaxis_title="Integration Time (s)",
    )
    return fig

if __name__ == '__main__':
    fig = demo_requiredIntegrationTime()
    fig.show()