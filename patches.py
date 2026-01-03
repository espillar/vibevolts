# patches.py contains add hoc commands to manipulate the
# simulation
import numpy as np

def setDetectorFOV(sim_data, fovSize):
    """
    setDetectorFOV goes through the detectors in sim_data
    and changes the FOVs of all of them to size (radians).
    This is meant to be an ad-hoc function for test,
    not a regular operational thing.
    """
    count = len(sim_data['detector'].fov)
    sim_data['detector'].fov = np.full(count, fovSize)
    
