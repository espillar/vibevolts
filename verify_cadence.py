
import numpy as np
from datetime import datetime, timezone, timedelta
from minimalsimulation import create_empty_simulation
from propagation import add_satellites_from_tle
from targets import add_fixed_points
from celestialbodies import add_celestial_bodies, celestial_update
from cadenceController import initCadence, nextIntegration
from detector import detectorPointingInitialize

def run_verification():
    print("--- Starting Cadence Controller Verification ---")
    
    # 1. Setup Simulation
    start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)
    
    # 2. Add Satellites (use a small number from a dummy TLE)
    # Let's assume dummy_tle.txt has at least 4 satellites.
    add_satellites_from_tle(sim_data, 'dummy_tle.txt', 'satellites')
    num_sats = sim_data['counts']['satellites']
    print(f"Added {num_sats} satellites.")
    
    # 3. Manually set detector properties
    # First half: 10s, Second half: 30s
    sim_data['detector'].integrationTime[:num_sats//2] = 10.0
    sim_data['detector'].integrationTime[num_sats//2:] = 30.0
    
    # Set other required properties for all detectors
    sim_data['detector'].filt = ["V"] * num_sats
    sim_data['detector'].apertureArea[:] = 0.01  # 10cm x 10cm roughly
    sim_data['detector'].qe[:] = 0.8
    sim_data['detector'].fov[:] = np.radians(10.0) # 10 degree FOV
    sim_data['detector'].pixelOmega[:] = 1e-9 # some small value
    
    # 4. Add Targets and Celestial Bodies
    add_fixed_points(sim_data, num_points=10)
    add_celestial_bodies(sim_data)
    celestial_update(sim_data)
    
    # Initialize detector pointing (required for scandetectors)
    detectorPointingInitialize(sim_data, 100) # 100 points in pointing sphere
    
    # 5. Initialize Cadence
    initCadence(sim_data)
    print("Cadence initialized.")
    for i, group in enumerate(sim_data['cadenceStructure']['cadenceList']):
        count = np.sum(group['scanMask'])
        print(f"Group {i}: Interval={group['scanInterval']}s, Count={count}")

    # 6. Run Iterations
    print("\n--- Running Iterations ---")
    for step in range(6):
        print(f"\nStep {step+1}:")
        next_time = sim_data['cadenceStructure']['nextTime']
        next_group = sim_data['cadenceStructure']['nextGroup']
        print(f"  Scheduled: Time={next_time}, Group={next_group}")
        
        results = nextIntegration(sim_data, print_output=1)
        
        print(f"  Executed:  Time={sim_data['time']}")
        if isinstance(results, dict):
            print(f"  Result Timestamp: {results.get('time')}")
            num_hits = len(results['sat_indices'])
            unique_sats = np.unique(results['sat_indices'])
            print(f"  Results:   {num_hits} detections from {len(unique_sats)} unique satellites.")
            print(f"  Sat IDs:   {unique_sats}")
        else:
            print("  Results:   No targets detected.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    run_verification()
