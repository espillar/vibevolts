
import pytest
from datetime import datetime, timezone
from minimalsimulation import create_empty_simulation

def test_create_empty_simulation_structure():
    """
    Tests that create_empty_simulation returns a dictionary
    with the expected keys and initial values.
    """
    # 1. Set up the input
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    delta_time = 60.0

    # 2. Call the function
    sim_data = create_empty_simulation(start_time=start_time, delta_time=delta_time)

    # 3. Assert the output is correct
    assert isinstance(sim_data, dict)
    assert 'start_time' in sim_data
    assert sim_data['start_time'] == start_time
    assert 'delta_time' in sim_data
    assert sim_data['delta_time'] == delta_time
    assert 'counts' in sim_data
    assert isinstance(sim_data['counts'], dict)
    assert 'pointing_spheres' in sim_data
    assert isinstance(sim_data['pointing_spheres'], dict)

def test_create_empty_simulation_raises_errors():
    """
    Tests that create_empty_simulation raises appropriate errors for invalid input.
    """
    # Test for non-datetime object
    with pytest.raises(TypeError):
        create_empty_simulation(start_time="not a datetime", delta_time=60.0)

    # Test for timezone-naive datetime
    with pytest.raises(ValueError):
        naive_dt = datetime(2023, 1, 1, 12, 0, 0)
        create_empty_simulation(start_time=naive_dt, delta_time=60.0)
