import numpy as np
from datetime import datetime, timezone
from types import SimpleNamespace

def sim_check(sim_data):
    """
    Prints a brief summary of what's present in a sim_data structure.
    """
    print("========================================")
    print("--- Simulation Data Check ---")

    # Check for start_time
    if 'start_time' in sim_data:
        print(f"          Start Time: {sim_data['start_time']}")
    else:
        print("*** Start Time: Not found")

    # Check for delta_time
    if 'delta_time' in sim_data:
        print(f"          Delta Time: {sim_data['delta_time']}")
    else:
        print("*** Delta Time: Not found")

    # Check for pointing_spheres
    if 'pointing_spheres' in sim_data and sim_data['pointing_spheres']:
        print(f"          Pointing Sphere Keys: {list(sim_data['pointing_spheres'].keys())}")
    else:
        print("*** Pointing Spheres: Not found or empty")

    # Check for celestial dictionary
    if 'celestial' in sim_data and sim_data['celestial']:
        print(f"          Celestial Dictionary Keys: {list(sim_data['celestial'].keys())}")
    else:
        print("*** Celestial Dictionary: Not found or empty")

    # Check for satellites
    if 'satellites' in sim_data and 'counts' in sim_data and 'satellites' in sim_data['counts']:
        num_sats = sim_data['counts']['satellites']
        print(f"          Number of Satellites: {num_sats}")

        if 'position' in sim_data['satellites']:
            if np.all(sim_data['satellites']['position'] == 0):
                print("*** Satellite positions are all zero.")
            else:
                print("          Satellite positions are not all zero.")
        else:
            print("*** Satellite positions not found.")

        if 'detector' in sim_data:
            detector = sim_data['detector']
            for i in range(num_sats):
                # Check a few attributes to see if they are non-zero
                if (hasattr(detector, 'aperture') and detector.aperture[i] != 0 or
                    hasattr(detector, 'pixelArea') and detector.pixelArea[i] != 0 or
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
    dummy_detector = SimpleNamespace()
    dummy_detector.aperture = np.array([0.785, 0.])
    dummy_detector.pixelArea = np.array([1e-10, 0.])
    dummy_detector.qe = np.array([0.5, 0.])
    dummy_detector.photoEff = np.zeros(2)
    dummy_detector.pixCount = np.zeros(2)
    dummy_detector.solarEx = np.zeros(2)
    dummy_detector.lunarex = np.zeros(2)
    dummy_detector.earthEx = np.zeros(2)
    dummy_detector.skyBack = np.zeros(2)
    dummy_detector.zpCal = np.zeros(2)
    dummy_detector.itime = np.zeros(2)
    dummy_detector.fov = np.zeros(2)
    dummy_detector.ifov = np.zeros(2)

    dummy_sim_data = {
        'start_time': datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        'delta_time': 60.0,
        'counts': {
            'satellites': 2,
            'celestial': 2,
        },
        'pointing_spheres': {
            'sphere1': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        },
        'celestial': {
            'position': np.zeros((2, 3)),
        },
        'satellites': {
            'position': np.zeros((2, 3)),
        },
        'detector': dummy_detector,
    }
    sim_check(dummy_sim_data)

    print("\n--- Testing with a minimal structure ---")
    minimal_sim_data = {
        'start_time': datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    }
    sim_check(minimal_sim_data)

