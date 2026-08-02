import numpy as np
from typing import Dict, Any

def add_observatories(
    sim_data: Any,
    num_observatories: int,
    latitudes: np.ndarray = None,
    longitudes: np.ndarray = None,
    altitudes: np.ndarray = None
) -> None:
    """
    Adds observatory data structures to the simulation data.

    Args:
        sim_data: The main simulation data structure.
        num_observatories: The number of observatories to add.
        latitudes: Latitude coordinates in degrees.
        longitudes: Longitude coordinates in degrees.
        altitudes: Altitude coordinates in meters.
    """
    if not isinstance(num_observatories, int) or num_observatories < 0:
        raise ValueError("num_observatories must be a non-negative integer.")

    if latitudes is None:
        latitudes = np.zeros(num_observatories)
    if longitudes is None:
        longitudes = np.zeros(num_observatories)
    if altitudes is None:
        altitudes = np.zeros(num_observatories)

    from detector import makeBlankDetector, appendDetector
    detector = makeBlankDetector(num_observatories)
    detector.category = ['observatories'] * num_observatories
    
    from minimalsimulation import ObservatoriesState
    if sim_data.counts.get('observatories', 0) > 0:
        # Append to existing
        existing = sim_data.observatories
        detector.asset_index = np.arange(sim_data.counts.observatories, sim_data.counts.observatories + num_observatories, dtype=int)
        sim_data.counts.observatories += num_observatories
        
        existing.latitude = np.append(existing.latitude, np.array(latitudes, dtype=float))
        existing.longitude = np.append(existing.longitude, np.array(longitudes, dtype=float))
        existing.altitude = np.append(existing.altitude, np.array(altitudes, dtype=float))
        existing.position = np.vstack([existing.position, np.zeros((num_observatories, 3), dtype=float)])
        existing.velocity = np.vstack([existing.velocity, np.zeros((num_observatories, 3), dtype=float)])
        existing.acceleration = np.vstack([existing.acceleration, np.zeros((num_observatories, 3), dtype=float)])
        existing.pointing = np.vstack([existing.pointing, np.zeros((num_observatories, 3), dtype=float)])
    else:
        sim_data.counts.observatories = num_observatories
        detector.asset_index = np.arange(num_observatories, dtype=int)
        sim_data.observatories = ObservatoriesState(
            latitude=np.array(latitudes, dtype=float),
            longitude=np.array(longitudes, dtype=float),
            altitude=np.array(altitudes, dtype=float),
            position=np.zeros((num_observatories, 3), dtype=float),
            velocity=np.zeros((num_observatories, 3), dtype=float),
            acceleration=np.zeros((num_observatories, 3), dtype=float),
            pointing=np.zeros((num_observatories, 3), dtype=float),
        )
    
    if 'detector' not in sim_data or not sim_data.detector:
        sim_data.detector = detector
    else:
        appendDetector(sim_data.detector, detector)

    # Initial propagation of observatory positions to the current simulation time
    propagate_observatories(sim_data, sim_data.time)


def propagate_observatories(sim_data: Any, time_date: Any) -> None:
    """
    Propagates observatory positions in GCRS (ECI) based on Earth rotation.
    """
    obs = sim_data.observatories
    if not obs or len(obs.latitude) == 0:
        return

    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    import astropy.units as u

    # Define locations
    locations = EarthLocation(
        lat=obs.latitude * u.deg,
        lon=obs.longitude * u.deg,
        height=obs.altitude * u.m
    )

    t = Time(time_date)
    # Get coordinates in Geocentric Celestial Reference System (GCRS)
    gcrs_coords = locations.get_gcrs(t)

    # GCRS coordinates are in meters
    x = gcrs_coords.cartesian.x.to(u.m).value
    y = gcrs_coords.cartesian.y.to(u.m).value
    z = gcrs_coords.cartesian.z.to(u.m).value

    # If x, y, z are scalars (e.g. single observatory), stack them correctly
    if np.isscalar(x):
        obs.position = np.array([[x, y, z]], dtype=float)
    else:
        obs.position = np.vstack([x, y, z]).T

    # Compute velocity using Earth rotation angular velocity: v = omega x r
    # Earth rotation rate is ~7.292115e-5 rad/s
    omega = np.array([0.0, 0.0, 7.292115e-5])
    obs.velocity = np.cross(omega, obs.position)
