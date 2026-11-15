from datetime import datetime, timezone

from demo_common import initialize_standard_simulation, set_exclusion_deg
from celestialbodies import celestial_update
from exclusion import update_exclusion_table

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

    set_exclusion_deg(sim_data, 30, 15, 10)

    sim_data = celestial_update(sim_data, sim_start_time)

    print("\n--- Generating exclusion table for Satellite 0"
          " vs First 100 Fixed Points (with debug print) ---")
    original_fixedpoints = sim_data.fixedpoints
    sim_data.fixedpoints = original_fixedpoints.subset(
        slice(0, 100)
    )

    update_exclusion_table(sim_data, print_debug_for_sat=0)

    sim_data.fixedpoints = original_fixedpoints
    print("\n--- Debug Print Demo Complete ---")
