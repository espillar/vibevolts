import numpy as np
from datetime import timedelta
from propagation import propagate_satellites
from scandetectors import scandetectors
from celestialbodies import celestial_update

def initCadence(sim_data: dict):
    """
    Initializes the cadenceStructure in sim_data based on detector
    integration times.
    
    This function groups detectors with identical integration times into cadence
    groups and calculates the initial schedule for the simulation.
    """
    # 1. Access integration times
    integration_times = sim_data['detector'].integrationTime
    
    # 2. Find unique integration times
    unique_intervals = np.unique(integration_times)
    
    # 3. Initialize cadenceStructure
    sim_data['cadenceStructure'] = {}
    cadence_list = []
    
    # 4 & 5. Create groups for each unique interval
    for interval in unique_intervals:
        cadence_group = {
            'scanInterval': float(interval),
            'scanMask': integration_times == interval,
            'scanNext': sim_data['time']  # Init to current time for initial scan
        }
        cadence_list.append(cadence_group)
    
    # 6. Store the list
    sim_data['cadenceStructure']['cadenceList'] = cadence_list
    
    # 7. Find initial nextTime and nextGroup
    _update_next_schedule(sim_data)

def nextIntegration(sim_data: dict, print_output: int = 0):
    """
    Finds and performs the next scheduled integration scan.
    
    Advances sim_data['time'] to the next scheduled event, propagates all satellites,
    and performs a vectorized scan for the specific detector group.
    """
    cadence_struct = sim_data['cadenceStructure']
    
    # 1. Advance simulation time
    sim_data['time'] = cadence_struct['nextTime']
    
    # 2. Propagate ALL satellites and update celestial bodies to the new time
    propagate_satellites(sim_data, sim_data['time'])
    celestial_update(sim_data, sim_data['time'])
    
    # 3. Get the active group
    group_idx = cadence_struct['nextGroup']
    group = cadence_struct['cadenceList'][group_idx]
    
    # 4. Perform the scan for the group's mask
    results = scandetectors(sim_data, print_output=print_output, mask=group['scanMask'])
    
    # 5. Update the group's next scheduled time
    group['scanNext'] = sim_data['time'] + timedelta(seconds=group['scanInterval'])
    
    # 6. Find the next overall event
    _update_next_schedule(sim_data)
    
    return results

def _update_next_schedule(sim_data: dict):
    """Helper to find the earliest scanNext among all groups."""
    cadence_list = sim_data['cadenceStructure']['cadenceList']
    
    # Find index of the group with the minimum scanNext
    next_times = [g['scanNext'] for g in cadence_list]
    min_idx = np.argmin(next_times)
    
    sim_data['cadenceStructure']['nextTime'] = cadence_list[min_idx]['scanNext']
    sim_data['cadenceStructure']['nextGroup'] = min_idx
