import numpy as np


def get_spherical_coords(arr):
    """
    given a n by 3 d array, return two 1D arrays
    with the theta and phi angles in radians
    """
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    r = np.linalg.norm(arr, axis=1)

    # theta: angle from the z-axis [0, pi]
    # We use np.divide to handle potential division by zero if r is 0
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))

    # phi: angle in the x-y plane [-pi, pi]
    phi = np.arctan2(y, x)

    return theta, phi



def scandetectors(sim_data: dict):
    """
    Scans for and processes detector data within the simulation.

    Args:
        sim_data (dict): The main simulation data dictionary.
                         This dictionary is expected to contain all
                         relevant simulation state and parameters.
    """
    print("scansensors function called with simulation data.")

    
    satpositions = sim_data['satellites']['position'] # all satellite positions
    detectorVect = sim_data['detector'] # detector pointings
    detectorFov = sim_data['detector'].fov # detector fields of view

    targets = sim_data['fixedpoints']['position'] # all target positions
    targetSize = sim_data['fixedpoints']['size'] # all target sizes
    sun = sim_data['celestial']['position'][0] # sun position
    


    
# #  Iterate over the satellites

#     for i in len(satpositions):
#         satposition = satpositions[i]
#         toTargets = targets - satposition
        
# #    Assuming V and W are (n, 3) arrays
#         dot_products = np.einsum('ij,ij->i', toTargets, detectorVect)
#         norms_V = np.linalg.norm(toTargets, axis=1)
#         norms_W = np.linalg.norm(detectorVect, axis=1)
#         angles = np.arccos(np.clip(dot_products / (norms_V * norms_W), -1.0, 1.0))

    return(detectorVect)
        
# Determine which ones are in the fov
# Skinny down the vector to be calculated to which ones are in the fov
# make sure you have some label so you can keep track
# calculate the SNR using the sun vector and the satellite vector and the
# function you have saved somewhere
# again clip to only the ones where the SNR is good enough
# save these to a database
    


    # Assume A and B are your (n, 3) arrays
    # A = np.array([[x1, y1, z1], ...])
    # B = np.array([[x2, y2, z2], ...])


    
    # # Calculate coords for both sets
    # theta_a, phi_a = get_spherical_coords(A)
    # theta_b, phi_b = get_spherical_coords(B)

    # # Calculate differences
    # delta_theta = theta_b - theta_a
    # delta_phi = phi_b - phi_a

    # # Optional: Normalize delta_phi to be within [-pi, pi]
    # # This handles the "wrap-around" at the 180-degree boundary
    # delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi    
        
