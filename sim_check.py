import numpy as np
from datetime import datetime, timezone
from typing import Any

def sim_check(sim_data: Any) -> None:
    """
    Validates and prints a diagnostic summary of the simulation data structure.

    Checks key properties of the simulation state (such as start time, step time,
    pointing spheres, celestial bodies, satellites, and detector configurations)
    and reports warnings if crucial simulation elements are missing or uninitialized.

    Args:
        sim_data: The main simulation data structure (SimulationState).
    """
    print("========================================")
    print("--- Simulation Data Check ---")

    # Check for start_time
    if hasattr(sim_data, 'start_time') and sim_data.start_time is not None:
        print(f"          Start Time: {sim_data.start_time}")
    else:
        print("*** Start Time: Not found")

    # Check for delta_time
    if hasattr(sim_data, 'delta_time') and sim_data.delta_time is not None:
        print(f"          Delta Time: {sim_data.delta_time}")
    else:
        print("*** Delta Time: Not found")

    # Check for pointing_spheres
    if hasattr(sim_data, 'pointing_spheres') and sim_data.pointing_spheres:
        print(f"          Pointing Sphere Keys: {list(sim_data.pointing_spheres.keys())}")
    else:
        print("*** Pointing Spheres: Not found or empty")

    # Check for celestial
    if hasattr(sim_data, 'celestial') and sim_data.celestial:
        print(f"          Celestial Dictionary Keys: {list(sim_data.celestial.keys())}")
    else:
        print("*** Celestial Dictionary: Not found or empty")

    # Check for satellites
    if (hasattr(sim_data, 'satellites') and sim_data.satellites and
            hasattr(sim_data, 'counts') and sim_data.counts and
            hasattr(sim_data.counts, 'satellites') and sim_data.counts.satellites):
        num_sats = sim_data.counts.satellites
        print(f"          Number of Satellites: {num_sats}")

        if hasattr(sim_data.satellites, 'position') and sim_data.satellites.position is not None:
            if np.all(sim_data.satellites.position == 0):
                print("*** Satellite positions are all zero.")
            else:
                print("          Satellite positions are not all zero.")
        else:
            print("*** Satellite positions not found.")

        if hasattr(sim_data, 'detector') and sim_data.detector:
            detector = sim_data.detector
            for i in range(num_sats):
                # Check a few attributes to see if they are non-zero
                if (hasattr(detector, 'apertureArea') and detector.apertureArea[i] != 0 or
                    hasattr(detector, 'pixelOmega') and detector.pixelOmega[i] != 0 or
                    hasattr(detector, 'qe') and detector.qe[i] != 0):
                    print(f"            - Satellite {i} has a detector.")
                else:
                    print(f"***         - Satellite {i} does not have a detector.")
        else:
            print("*** Detector information not found for satellites.")
    else:
        print("*** Satellites: Not found")

    print("--- End of Check ---")

if __name__ == '__main__':
    # Create a dummy sim_data for demonstration
    from detector import makeBlankDetector
    from minimalsimulation import (
        create_empty_simulation,
        SatellitesState,
        CelestialState,
    )
    dummy_detector = makeBlankDetector(2)
    dummy_detector.apertureArea = np.array([0.785, 0.])
    dummy_detector.pixelOmega = np.array([1e-10, 0.])
    dummy_detector.qe = np.array([0.5, 0.5])

    dummy_sim_data = create_empty_simulation(
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        delta_time=60.0
    )
    dummy_sim_data.counts.satellites = 2
    dummy_sim_data.counts.celestial = 2
    dummy_sim_data.pointing_spheres = {
        'sphere1': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    }
    dummy_sim_data.celestial = CelestialState(
        position=np.zeros((2, 3))
    )
    dummy_sim_data.satellites = SatellitesState(
        position=np.zeros((2, 3))
    )
    dummy_sim_data.detector = dummy_detector
    sim_check(dummy_sim_data)

    print("\n--- Testing with a minimal structure ---")
    minimal_sim_data = create_empty_simulation(
        start_time=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    )
    sim_check(minimal_sim_data)

