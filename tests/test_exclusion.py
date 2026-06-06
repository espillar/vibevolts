import numpy as np
import pytest
from datetime import datetime, timezone
from minimalsimulation import create_empty_simulation
from celestialbodies import add_celestial_bodies, celestial_update
from targets import add_fixed_points
from constellation import geos
from constants import MOON_RADIUS, EARTH_RADIUS
from exclusion import exclusion, update_exclusion_table

def reference_update_exclusion_table(data_struct):
    """
    Original loop-based update_exclusion_table used as reference.
    """
    num_sats = data_struct['counts'].get('satellites', 0)
    num_fixed_points = data_struct['counts'].get('fixedpoints', 0)
    
    exclusion_matrix = np.zeros((num_fixed_points, num_sats), dtype=int)
    original_pointing = data_struct['detector'].pointing.copy()
    
    for j in range(num_fixed_points):
        target_pos = data_struct['fixedpoints']['position'][j]
        for i in range(num_sats):
            sat_pos = data_struct['satellites']['position'][i]
            pointing_vector = target_pos - sat_pos
            data_struct['detector'].pointing[i] = pointing_vector
            is_excluded = exclusion(data_struct, i)
            exclusion_matrix[j, i] = is_excluded
            
    data_struct['detector'].pointing = original_pointing
    return exclusion_matrix

def test_vectorized_exclusion_table_correctness():
    """
    Test that the vectorized update_exclusion_table implementation matches the loop-based reference.
    """
    sim_start_time = datetime(2025, 8, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(sim_start_time)
    add_celestial_bodies(sim_data)
    sim_data = celestial_update(sim_data, sim_start_time)
    add_fixed_points(sim_data, 150)
    geos(sim_data, 15, 0.15)

    # Calculate using loop-based reference
    expected_matrix = reference_update_exclusion_table(sim_data)

    # Calculate using vectorized implementation
    update_exclusion_table(sim_data)
    actual_matrix = sim_data['fixedpoints']['exclusion']

    assert np.array_equal(expected_matrix, actual_matrix)
