from datetime import datetime, timezone, timedelta
import numpy as np
from minimalsimulation import create_empty_simulation
from constellation import geos, geosmod
from targets import add_fixed_points
from celestialbodies import add_celestial_bodies
from propagation import propagate_satellites
from cadenceController import initCadence, nextIntegration
from dataHandling import DataHandler
from constants import *

def run_simulation_template():
    """
    Simulation template using cadenceController and DataHandler.
    Shows the basic lifecycle: Initialize -> Add Components -> Loop
    """
    # 1. Initialize
    start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time, delta_time=600.0)

    # 2. Add Components
    # Add a GEO constellation with 1 satellite using geosmod for full
    # detector initialization:
    # n=1, band='V', fov=10*DEGREE, ifov=3*ARCSEC, aper=1.0, limitingmag=20.0
    geosmod(
        sim_data, n=1, band='V', fov=10*DEGREE,
        ifov=3*ARCSEC, aper=1.0, limitingmag=20.0
    )

    # Set integration time for the satellite detector (e.g., 600 seconds)
    sim_data.satellites.detector.integrationTime = np.full(
        sim_data.counts.satellites, 600.0
    )

    # Add fixed reference points (1m size, between 1.1 and 2.0 GEO_RADIUS)
    add_fixed_points(
        sim_data, num_points=500, size=1.0,
        innerRadius=1.1 * GEO_RADIUS, outerRadius=2.0 * GEO_RADIUS
    )

    # Verify targets
    fixed_pos = sim_data.fixedpoints.position
    fixed_radii = np.linalg.norm(fixed_pos, axis=1)
    print(f"Created {len(fixed_pos)} fixed points.")
    print(
        f"Radius range: {np.min(fixed_radii)/GEO_RADIUS:.2f} to "
        f"{np.max(fixed_radii)/GEO_RADIUS:.2f} GEO_RADIUS"
    )
    print(f"Target size: {sim_data.fixedpoints.size[0]} m")

    # Add Sun and Moon
    add_celestial_bodies(sim_data)

    # Initialize cadence controller and data handler
    initCadence(sim_data)
    data_handler = DataHandler()

    num_sats = sim_data.counts.satellites
    print(f"Simulation initialized with {num_sats} satellites.")

    # 3. Simulation Loop
    for step in range(20):  # Run for 20 steps

        # cadenceController.nextIntegration advances time, propagates
        # satellites, and scans. Now it also updates celestial body
        # positions.
        results = nextIntegration(sim_data)

        print(f"\n--- Step {step}: {sim_data.time.isoformat()} ---")

        # Access data (example: first satellite position)
        sat_pos = sim_data.satellites.position[0]
        print(f"Satellite 0 Position: {sat_pos}")

        if results is not None:
             next_grp = sim_data.cadenceStructure.nextGroup
             print(f"Scan performed for group {next_grp}")
             data_handler.add_results(results)

    # 4. Save Results
    data_handler.save_to_csv('simulation_results.csv')


if __name__ == "__main__":
    run_simulation_template()
