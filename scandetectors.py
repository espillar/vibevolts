import numpy as np
import lambertian
import radiometry_calcs


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

#######################################################


def scandetectors(sim_data: dict):
    """
    Scans for and processes detector data within the simulation.
    Args:
        sim_data (dict): The main simulation data dictionary.
                         This dictionary is expected to contain all
                         relevant simulation state and parameters.

    TODO: we are using the same filter for all the detectors!
    """
    print("scansensors function called with simulation data.")

    satpositions = sim_data['satellites']['position'] # all satellite positions
    detectorVect = sim_data['detector'].pointing # detector pointings
    
#    print('detectorVect ', detectorVect)
    detectorFov = sim_data['detector'].fov # detector fields of view
    targets = sim_data['fixedpoints']['position'] # all target positions
    targetSize = sim_data['fixedpoints']['size'] # all target sizes
    sunVect = sim_data['celestial']['position'][0] # sun position
    fovs = sim_data['detector'].fov # field of view of the detector
    sun, space, sky = radiometry_calcs.fluxes(sim_data['detector'].filt[0])
    albedo = sim_data['fixedpoints']['albedo']
    radius = sim_data['fixedpoints']['size']/2
  

# Scan over detectors
#    for i in range(len(satpositions)):
    for i in range(1): # Only one set for testing
         satposition = satpositions[i,:]
         ray = detectorVect[i,:]
         toTargets = targets - satposition
#         print(' toTargets, ray shapes ', toTargets.shape, ray.shape)
         dot_products = np.einsum('ij,j->i', toTargets, ray)
#         print(dot_products)
         norms_V = np.linalg.norm(toTargets, axis=1)
         norms_W = np.linalg.norm(ray, axis=0)
         angles = np.arccos(np.clip(dot_products / (norms_V * norms_W), -1.0, 1.0))
#         print('angles ', angles)
         fov = fovs[i]
         mask = angles < fov
         print(mask)
         # print('sunvect.shape ', sunVect.shape)
         # print( 'toTargets.shape ', toTargets.shape)
         # print( ' albed[mask].shape ' , albedo[mask].shape)
         signal = lambertian.lambertiansphere(
             -sunVect,
             -toTargets[mask],
             albedo[mask],
             radius[mask],
             sun)
         print(signal)
         


# Compare the the angles to the acceptance angle and create a mask for those
# For those in the mask, computer the SNR
# store an appropriately labeled vector with the detector number, the target number, the time,
# and the SNR in a pandas array, or maybe a numpy array.
# 
         
# Determine which ones are in the fov
# Skinny down the vector to be calculated to which ones are in the fov
# make sure you have some label so you can keep track
# calculate the SNR using the sun vector and the satellite vector and the
# function you have saved somewhere
# again clip to only the ones where the SNR is good enough
# save these to a database
    

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
        
    return(0)



#######################################################



def findVectorMask(values: np.ndarray, floorValue: float) -> np.ndarray:
    """
    Compares values in a 1D numpy array to a floorValue and returns a boolean mask.

    The mask will have True where values are greater than or equal to floorValue,
    and False otherwise.

    Args:
        values (np.ndarray): A 1D numpy array of numerical values.
        floorValue (float): The threshold value to compare against.

    Returns:
        np.ndarray: A boolean numpy array (mask) of the same shape as 'values'.
    """


    return values >= floorValue


