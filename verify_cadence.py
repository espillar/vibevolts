
import numpy as np
from datetime import datetime, timezone, timedelta
from minimalsimulation import create_empty_simulation
from propagation import add_satellites_from_tle
from targets import add_fixed_points
from celestialbodies import add_celestial_bodies, celestial_update
from cadenceController import initCadence, nextIntegration
from pointing import detectorPointingInitialize
from dataHandling import DataHandler
from constants import *

def run_verification():
    print("--- Starting Cadence Controller Verification ---")
    
    # 0. Initialize Data Handler
    handler = DataHandler()
    
    # 1. Setup Simulation
    start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time)
    
    # 2. Add Satellites (use a small number from a dummy TLE)
    # Let's assume dummy_tle.txt has at least 4 satellites.
    add_satellites_from_tle(sim_data, 'dummy_tle.txt', 'satellites')
    from propagation import propagate_satellites
    propagate_satellites(sim_data, sim_data['time'], 'satellites')
    num_sats = sim_data['counts']['satellites']
    print(f"Added {num_sats} satellites.")
    
    # 3. Manually set detector properties
    # First half: 10s, Second half: 30s
    sim_data['detector'].integrationTime[:num_sats//2] = 10.0
    sim_data['detector'].integrationTime[num_sats//2:] = 30.0
    
    # Set other required properties for all detectors
    sim_data['detector'].filt = ["V"] * num_sats
    sim_data['detector'].apertureArea[:] = 1.0  # 10cm x 10cm roughly
    sim_data['detector'].qe[:] = 0.8
    sim_data['detector'].fov[:] = np.radians(180) # 10 degree FOV
    sim_data['detector'].pixelOmega[:] = (3 * ARCSEC)**2 # some small value

    
    
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

    # Plot positions if in a notebook
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            print("\n--- Plotting Initial Positions ---")
            from plotting_3d import plot_3d_scatter
            import plotly.graph_objects as go
            
            sat_pos = sim_data['satellites']['position']
            target_pos = sim_data['fixedpoints']['position']
            sat_pointing = sim_data['detector'].pointing
            
            fig = plot_3d_scatter(
                positions=sat_pos,
                title="Initial Satellite and Target Positions",
                plot_time=sim_data['time'],
                trace_name="Satellites",
                marker_size=6,
                marker_color="blue"
            )
            
            # Add pointing vectors (cyan)
            vector_scale = 2e6  # 2000 km
            for i in range(len(sat_pos)):
                start = sat_pos[i]
                pvec = sat_pointing[i]
                norm = np.linalg.norm(pvec)
                if norm > 0:
                    pvec = pvec / norm
                end = start + pvec * vector_scale
                fig.add_trace(go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    line=dict(color='cyan', width=4),
                    showlegend=False
                ))

            fig.add_trace(go.Scatter3d(
                x=target_pos[:, 0], y=target_pos[:, 1], z=target_pos[:, 2],
                mode="markers",
                marker=dict(size=4, color="red", opacity=0.8),
                name="Targets"
            ))
            
            fig.show()
    except ImportError:
        pass

    # 6. Run Iterations
    print("\n--- Running Iterations ---")
    for step in range(6):
        print(f"\nStep {step+1}:")
        next_time = sim_data['cadenceStructure']['nextTime']
        next_group = sim_data['cadenceStructure']['nextGroup']
        print(f"  Scheduled: Time={next_time}, Group={next_group}")
        
        results = nextIntegration(sim_data, print_output=1)
        handler.add_results(results)
        
        print(f"  Executed:  Time={sim_data['time']}")
        if isinstance(results, dict):
            print(f"  Result Timestamp: {results.get('time')}")
            num_hits = len(results['sat_indices'])
            unique_sats = np.unique(results['sat_indices'])
            print(f"  Results:   {num_hits} detections from {len(unique_sats)} unique satellites.")
            print(f"  Sat IDs:   {unique_sats}")
        else:
            print("  Results:   No targets detected.")

    # 7. Save and Preview Results
    print("\n--- Saving Results ---")
    df = handler.get_dataframe()
    if not df.empty:
        print(f"Collected {len(df)} total detection rows.")
        handler.save_to_csv("verification_results.csv")
    else:
        print("No detections were collected.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    run_verification()
