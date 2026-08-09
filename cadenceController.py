import numpy as np
from datetime import timedelta
from propagation import propagate_satellites
from scandetectors import scandetectors
from celestialbodies import celestial_update
from minimalsimulation import CadenceGroup, CadenceState
from observatories import propagate_observatories


def initCadence(sim_data) -> None:
    """
    Initialises sim_data.cadenceStructure from detector integration times.

    Groups detectors that share the same integration time into CadenceGroup
    objects and stores them in a CadenceState on sim_data.cadenceStructure.
    All groups are initialised with scanNext equal to the current simulation
    time so that every group performs a scan on the first call to
    nextIntegration.

    Args:
        sim_data: The main SimulationState object.
    """
    all_detectors = sim_data.get_all_detectors()
    if all_detectors is None or len(all_detectors.integrationTime) == 0:
        return
    integration_times = all_detectors.integrationTime
    unique_intervals = np.unique(integration_times)

    cadence_list = [
        CadenceGroup(
            scanInterval=float(interval),
            scanMask=(integration_times == interval),
            scanNext=sim_data.time,
        )
        for interval in unique_intervals
    ]

    sim_data.cadenceStructure = CadenceState(cadenceList=cadence_list)
    _update_next_schedule(sim_data)


def nextIntegration(sim_data, print_output: int = 0) -> dict:
    """
    Advances the simulation to the next scheduled scan and executes it.

    Steps:
    1. Advances sim_data.time to the earliest scheduled group event.
    2. Propagates all satellites and updates celestial body positions.
    3. Runs scandetectors for only the active group's masked detectors.
    4. Updates that group's scanNext by one scanInterval.
    5. Recomputes which group fires next.

    Args:
        sim_data:      The main SimulationState object.
        print_output:  If > 0, scandetectors prints per-detection details.

    Returns:
        dict: The detection results returned by scandetectors, containing
              'time', 'sat_indices', 'target_indices', 'signal', 'noise',
              and 'snr'.
    """
    cadence = sim_data.cadenceStructure

    # 1. Advance simulation time to the next scheduled event.
    sim_data.time = cadence.nextTime

    # 2. Propagate all satellites and celestial bodies to the new time.
    propagate_satellites(sim_data, sim_data.time)
    celestial_update(sim_data, sim_data.time)
    propagate_observatories(sim_data, sim_data.time)

    # 3. Run the scan for the active group's detector subset.
    group = cadence.cadenceList[cadence.nextGroup]
    results = scandetectors(sim_data, print_output=print_output,
                            mask=group.scanMask)

    # 4. Schedule this group's next scan.
    group.scanNext = sim_data.time + timedelta(seconds=group.scanInterval)

    # 5. Find the next overall event across all groups.
    _update_next_schedule(sim_data)

    return results


def _update_next_schedule(sim_data) -> None:
    """
    Scans all CadenceGroups and updates sim_data.cadenceStructure with
    the index and datetime of the earliest upcoming scan.

    Args:
        sim_data: The main SimulationState object.
    """
    cadence = sim_data.cadenceStructure
    next_times = [g.scanNext for g in cadence.cadenceList]
    min_idx = min(range(len(next_times)), key=lambda i: next_times[i])
    cadence.nextGroup = min_idx
    cadence.nextTime = cadence.cadenceList[min_idx].scanNext
