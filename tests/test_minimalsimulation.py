
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


def test_hybrid_dataclass_access():
    """
    Tests that SimulationState and its sub-states behave correctly
    both as dataclasses (attribute access) and as dictionaries.
    """
    import numpy as np
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    sim_data = create_empty_simulation(start_time=start_time, delta_time=60.0)

    # 1. Verify isinstance checks
    assert isinstance(sim_data, dict)
    assert isinstance(sim_data.counts, dict)

    # 2. Test attribute read/write on SimulationState
    sim_data.time = datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
    assert sim_data['time'] == datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

    # 3. Test dictionary read/write on SimulationState
    sim_data['time'] = datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
    assert sim_data.time == datetime(2023, 1, 1, 14, 0, 0, tzinfo=timezone.utc)

    # 4. Test sub-state dictionary key write auto-promotion
    sim_data['satellites'] = {
        'position': np.array([[1.0, 2.0, 3.0]]),
        'velocity': np.array([[0.0, 0.0, 0.0]]),
    }
    # Should automatically convert to SatellitesState
    from minimalsimulation import SatellitesState
    assert isinstance(sim_data.satellites, SatellitesState)
    assert sim_data.satellites.position[0, 0] == 1.0
    assert sim_data['satellites']['position'][0, 0] == 1.0

    # 5. Test sub-state attribute write auto-promotion
    sim_data.fixedpoints = {
        'position': np.array([[4.0, 5.0, 6.0]]),
        'size': np.array([10.0]),
    }
    from minimalsimulation import FixedPointsState
    assert isinstance(sim_data.fixedpoints, FixedPointsState)
    assert sim_data.fixedpoints.position[0, 0] == 4.0
    assert sim_data['fixedpoints']['size'][0] == 10.0

    # 6. Test basic dictionary functions
    assert 'time' in sim_data
    assert 'satellites' in sim_data
    keys = list(sim_data.keys())
    assert 'time' in keys
    assert 'satellites' in keys
    assert len(sim_data) > 0

