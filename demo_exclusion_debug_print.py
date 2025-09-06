import numpy as np
from datetime import datetime, timezone

from common import initialize_standard_simulation
from propagation import celestial_update
from visibility import update_visibility_table
from simulation import DETECTOR_SOLAR_EXCL_IDX, DETECTOR_LUNAR_EXCL_IDX, DETECTOR_EARTH_EXCL_IDX

def demo_exclusion_debug_print():
    """
    Demonstrates the debug printing feature of the exclusion function.

    This function runs the exclusion check for the first satellite against
    the first 100 fixed points and prints the detailed debug output for
    each check, as enabled by the `print_debug_for_sat` parameter.
    """
    print("\n--- Starting Demo: Exclusion Debug Print ---")
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = initialize_standard_simulation(sim_start_time)

    sim_data['satellites']['detector'][:, DETECTOR_SOLAR_EXCL_IDX] = np.deg2rad(30)
    sim_data['satellites']['detector'][:, DETECTOR_LUNAR_EXCL_IDX] = np.deg2rad(15)
    sim_data['satellites']['detector'][:, DETECTOR_EARTH_EXCL_IDX] = np.deg2rad(10)

    sim_data = celestial_update(sim_data, sim_start_time)

    print("\n--- Generating exclusion table for Satellite 0 vs First 100 Fixed Points (with debug print) ---")
    original_fixed_points = sim_data['fixedpoints']['position']
    sim_data['fixedpoints']['position'] = original_fixed_points[:100]

    update_visibility_table(sim_data, print_debug_for_sat=0)

    sim_data['fixedpoints']['position'] = original_fixed_points
    print("\n--- Debug Print Demo Complete ---")
