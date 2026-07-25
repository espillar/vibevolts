import numpy as np
from datetime import datetime, timedelta, timezone
from minimalsimulation import create_empty_simulation
from observatories import add_observatories, propagate_observatories
from celestialbodies import add_celestial_bodies, celestial_update
from detector import makeDetector
from exclusion import exclusion
from cadenceController import initCadence, nextIntegration
from targets import add_fixed_points

def test_observatories_all():
    print("=== STARTING OBSERVATORY TEST ===")
    
    # 1. Initialize Simulation
    start_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time, delta_time=60.0)
    
    # 2. Add Observatories (e.g., Table Mountain Observatory, CA: lat=34.38, lon=-117.68)
    latitudes = np.array([34.38])
    longitudes = np.array([-117.68])
    altitudes = np.array([2286.0]) # 2286 meters
    
    add_observatories(sim_data, num_observatories=1, latitudes=latitudes, longitudes=longitudes, altitudes=altitudes)
    add_celestial_bodies(sim_data)
    celestial_update(sim_data, sim_data.time)
    
    # Configure detector for observatory
    # Set earthEx (horizon limit) to 10 degrees = 10 * np.pi / 180
    d = makeDetector(1, band='V', fov=np.radians(10), ifov=np.radians(0.1), aper=1.0)
    d.earthEx = np.array([np.radians(10.0)]) # 10 deg minimum elevation
    sim_data.detector = d
    
    print(f"Observatory Count: {sim_data.counts.observatories}")
    print(f"Detector Count: {len(sim_data.detector.filt)}")
    
    # 3. Check Initial ECI Position
    pos_init = sim_data.observatories.position[0].copy()
    vel_init = sim_data.observatories.velocity[0].copy()
    print(f"Initial ECI Position (m): {pos_init}")
    print(f"Initial ECI Velocity (m/s): {vel_init}")
    assert np.any(pos_init != 0), "Observatory position should not be zero!"
    assert np.any(vel_init != 0), "Observatory velocity should not be zero!"
    
    # 4. Check Coordinate Change (Earth Rotation) after 6 hours
    later_time = start_time + timedelta(hours=6)
    propagate_observatories(sim_data, later_time)
    pos_later = sim_data.observatories.position[0].copy()
    print(f"ECI Position after 6 hours (m): {pos_later}")
    
    distance_moved = np.linalg.norm(pos_later - pos_init)
    print(f"Distance moved in inertial space (m): {distance_moved:.2f}")
    assert distance_moved > 1e3, "Observatory should have moved in ECI space due to Earth rotation!"
    
    # Reset position to start time for pointing tests
    propagate_observatories(sim_data, start_time)
    
    # 5. Horizon Exclusion Tests
    # Local Zenith Vector
    zenith_normal = pos_init / np.linalg.norm(pos_init)
    
    # Zenith Pointing (looking straight up)
    sim_data.detector.pointing[0] = zenith_normal
    ex_zenith = exclusion(sim_data, 0, print_debug=True)
    print(f"Zenith Pointing Excluded: {ex_zenith} (expected: 0)")
    assert ex_zenith == 0, "Looking straight up from observatory should not be excluded!"
    
    # Nadir Pointing (looking straight down)
    sim_data.detector.pointing[0] = -zenith_normal
    ex_nadir = exclusion(sim_data, 0, print_debug=True)
    print(f"Nadir Pointing Excluded: {ex_nadir} (expected: 1)")
    assert ex_nadir == 1, "Looking straight down from observatory must be excluded!"
    
    # Horizon Pointing (9 degrees elevation - below 10 degree threshold)
    # Generate vector perpendicular to zenith, tilt it up slightly (e.g. 5 degrees)
    perp = np.array([-zenith_normal[1], zenith_normal[0], 0.0])
    perp /= np.linalg.norm(perp)
    horizon_pointing = zenith_normal * np.sin(np.radians(5.0)) + perp * np.cos(np.radians(5.0))
    sim_data.detector.pointing[0] = horizon_pointing / np.linalg.norm(horizon_pointing)
    ex_horizon = exclusion(sim_data, 0, print_debug=True)
    print(f"Horizon Pointing (5 deg elev) Excluded: {ex_horizon} (expected: 1)")
    assert ex_horizon == 1, "Observing below 10 degrees elevation must be excluded!"
    
    # 6. Integration schedule test
    # Add a target
    add_fixed_points(sim_data, num_points=1, size=10.0, innerRadius=3e7, outerRadius=4e7)
    # Point observatory at the target
    target_pos = sim_data.fixedpoints.position[0]
    pointing_vector = target_pos - pos_init
    sim_data.detector.pointing[0] = pointing_vector / np.linalg.norm(pointing_vector)
    
    initCadence(sim_data)
    results = nextIntegration(sim_data, print_output=1)
    print(f"Scan Results at {sim_data.time}: {results}")
    
    print("=== ALL TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_observatories_all()
