import numpy as np
from typing import Any

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

    from detector import makeBlankDetector
    from minimalsimulation import ObservatoriesState

    new_obs = ObservatoriesState(
        latitude=np.array(latitudes, dtype=float),
        longitude=np.array(longitudes, dtype=float),
        altitude=np.array(altitudes, dtype=float),
        position=np.zeros((num_observatories, 3), dtype=float),
        velocity=np.zeros((num_observatories, 3), dtype=float),
        detector=makeBlankDetector(num_observatories),
    )

    if sim_data.observatories is not None and len(sim_data.observatories) > 0:
        sim_data.observatories.append(new_obs)
    else:
        sim_data.observatories = new_obs

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
        obs.position = np.column_stack((x, y, z))

    # Compute velocity via finite-difference of GCRS positions.
    # The naive ω×r formula is only valid in the ECEF/ITRS
    # (body-fixed) frame, but obs.position is in GCRS (inertial).
    # A small Δt finite-difference gives the correct GCRS velocity
    # including precession and nutation effects.
    from datetime import timedelta
    dt_seconds = 1.0
    t2 = Time(time_date + timedelta(seconds=dt_seconds))
    gcrs_coords2 = locations.get_gcrs(t2)

    x2 = gcrs_coords2.cartesian.x.to(u.m).value
    y2 = gcrs_coords2.cartesian.y.to(u.m).value
    z2 = gcrs_coords2.cartesian.z.to(u.m).value

    if np.isscalar(x2):
        pos2 = np.array([[x2, y2, z2]], dtype=float)
    else:
        pos2 = np.column_stack((x2, y2, z2))

    obs.velocity = (pos2 - obs.position) / dt_seconds
