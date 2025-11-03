import numpy as np
from datetime import datetime
from typing import Dict, Any
import math
import random
import radiometry_data
from radiometry_data import FILTER_DATA
from constants import *

from pointing import generate_pointing_sphere

#########################################################

def geos(sim_data, n,  fov) -> None:
    """
    Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

    Args:
        sim_data: The main simulation data dictionary.
        n: The number of satellites to create.
        fov: The diameter of the field of view of the satellite in radians.
    """


    # Calculate solid angle 
    theta = fov / 2
    solid_angle = 2 * np.pi * (1 - np.cos(theta)) 

    # Calculate grid_points - blow things up by 0.25 for overlap
    grid_points = int(4 * np.pi / solid_angle * 1.25)

    # Generate and store the pointing sphere and place in ['pointing_sphers'][n]
    generate_pointing_sphere(sim_data, grid_points)

    orbital_elements_list = []
    epochs_list = []
    pointing_state_list = []

    # Geostationary orbit semi-major axis in meters
    a = 42164000.0

    # Create a set of elements evenly spaced around the equator in
    # orbital_elemens_list
    for i in range(n):
        elements = np.zeros(6)
        elements[ORBITAL_A_IDX] = a
        elements[ORBITAL_E_IDX] = 0.0
        elements[ORBITAL_I_IDX] = 0.0
        elements[ORBITAL_RAAN_IDX] = i * 2 * np.pi / n
        elements[ORBITAL_ARGP_IDX] = 0.0
        elements[ORBITAL_M_IDX] = 0.0
        orbital_elements_list.append(elements)
        epochs_list.append(sim_data['start_time'])

   # And these are the pointing state of the satellite. Make the position random.
   
        pointing_state = np.zeros(2, dtype=int)
        pointing_state[POINTING_COUNT_IDX] = grid_points
        pointing_state[POINTING_PLACE_IDX] = random.randint(0,grid_points-1)
    
        pointing_state_list.append(pointing_state)

    orbital_elements = np.array(orbital_elements_list, dtype=float)
    pointing_state_array = np.array(pointing_state_list, dtype=int)

    if 'satellites' not in sim_data:
        sim_data['counts']['satellites'] = n
        sim_data['satellites'] = {
            'position': np.zeros((n, 3), dtype=float),
            'velocity': np.zeros((n, 3), dtype=float),
            'acceleration': np.zeros((n, 3), dtype=float),
            'orbital_elements': orbital_elements,
            'epochs': epochs_list,
            'pointing': np.zeros((n, 3), dtype=float),
            'pointing_state': pointing_state_array,
            'detector': np.zeros((n, 9), dtype=float),
        }
    else:
        # Append to existing satellites
        sim_data['counts']['satellites'] += n
        sim_data['satellites']['position'] = np.vstack([sim_data['satellites']['position'], np.zeros((n, 3), dtype=float)])
        sim_data['satellites']['velocity'] = np.vstack([sim_data['satellites']['velocity'], np.zeros((n, 3), dtype=float)])
        sim_data['satellites']['acceleration'] = np.vstack([sim_data['satellites']['acceleration'], np.zeros((n, 3), dtype=float)])
        sim_data['satellites']['orbital_elements'] = np.vstack([sim_data['satellites']['orbital_elements'], orbital_elements])
        sim_data['satellites']['epochs'].extend(epochs_list)
        sim_data['satellites']['pointing'] = np.vstack([sim_data['satellites']['pointing'], np.zeros((n, 3), dtype=float)])
        sim_data['satellites']['pointing_state'] = np.vstack([sim_data['satellites']['pointing_state'], pointing_state_array])
        sim_data['satellites']['detector'] = np.vstack([sim_data['satellites']['detector'], np.zeros((n, 9), dtype=float)])

##########################################################


##########################################################

def makeDetector(n, band, fov,ifov, aper,limitingmag, qe = 0.5, photfrac=0.7, solarex = 0.5, lunarex =0.25,  earthex=0.25):
    '''
    makeDetector takes parameters of a sensor and stuffs a detector array, which it returns.
    I expect this function to be calld when a new satellite is created.
    '''
    detect = np.zeros((n,11), dtype=float)
    print(detect.shape)
    detect[:,DETECTOR_APERTURE_IDX] = aper  #aperture size me
    detect[:,DETECTOR_PIXEL_SIZE_IDX] = ifov  #pixel size rads
    detect[:,DETECTOR_QE_IDX] = qe   # Total QE
    detect[:,DETECTOR_PHOT_EFF_IDX] = photfrac  # fraction in photometry bucket
    pixels = (ifov/fov)**2 # pixels
    detect[:,DETECTOR_PIXELS_IDX] = pixels   # total pixels in the array
    detect[:,DETECTOR_SOLAR_EXCL_IDX] = solarex  # solar exclusion angle
    detect[:,DETECTOR_LUNAR_EXCL_IDX] = lunarex  # lunar exclusion angle
    detect[:,DETECTOR_EARTH_EXCL_IDX] = earthex  # eearth exclusion angle
    detect[:,DETECTOR_SKY_BACK_IDX] = FILTER_DATA[band]['space'] # photon backgroud
    detect[:,DETECTOR_FILTER_BAND_IDX] = band # Band
    detect[:,DETECTOR_FILTER_BAND_CAL_IDX] = FILTER_DATA[band]['zero_point'] # Filter Zero Point
    return(detect)

###########################################################

def requiredIntegrationTime(limitingMag, d):
    '''
    requiredIntegrationTime(limitingMag, d)
    takes a two dimensional detector array ("detect")and calculates all the integration tiemes
    and returns those as a vector.
    '''
    t  = (d[:,DETECTOR_SKY_BACK_IDX] * d[:,DETECTOR_PIXEL_SIZE_IDX]) / \
       ( d[:,DETECTOR_QE_IDX] *
         d[:,DETECTOR_PHOT_EFF_IDX]**2  *
         math.pi * 
         (d[:,DETECTOR_APERTURE_IDX]/2)**2 *
         amag(limitingMag)**2 *
         d[:,DETECTOR_FILTER_ZP_IDX])
    return(t)





###########################################################
        
def geosmod(sim_data, n, band,fov,ifov, aper, limitingmag) -> None:
    """
    Creates n equally spaced satellites in GEO and adds them to the 'satellites' group in the simulation.

    Args:
        sim_data: The main simulation data dictionary.
        n: The number of satellites to create.
        fov: The diameter of the field of view of the satellite in radians.
    """
#    detect = makeDetector(n, band, fov,ifov, aper,limitingmag):
        
    # Calculate solid angle 
    theta = fov / 2
    solid_angle = 2 * np.pi * (1 - np.cos(theta)) 

    # Calculate grid_points - blow things up by 0.25 for overlap
    grid_points = int(4 * np.pi / solid_angle * 1.25)

    # Generate and store the pointing sphere and place in ['pointing_sphers'][n]
    generate_pointing_sphere(sim_data, grid_points)

    orbital_elements_list = []
    epochs_list = []
    pointing_state_list = []

    # Geostationary orbit semi-major axis in meters
    a = 42164000.0

    # Create a set of elements evenly spaced around the equator in
    # orbital_elemens_list
    for i in range(n):
        elements = np.zeros(6)
        elements[ORBITAL_A_IDX] = a
        elements[ORBITAL_E_IDX] = 0.0
        elements[ORBITAL_I_IDX] = 0.0
        elements[ORBITAL_RAAN_IDX] = i * 2 * np.pi / n
        elements[ORBITAL_ARGP_IDX] = 0.0
        elements[ORBITAL_M_IDX] = 0.0
        orbital_elements_list.append(elements)
        epochs_list.append(sim_data['start_time'])

   # And these are the pointing state of the satellite. Make the position random.
   
        pointing_state = np.zeros(2, dtype=int)
        pointing_state[POINTING_COUNT_IDX] = grid_points
        pointing_state[POINTING_PLACE_IDX] = random.randint(0,grid_points-1)
    
        pointing_state_list.append(pointing_state)

    orbital_elements = np.array(orbital_elements_list, dtype=float)
    pointing_state_array = np.array(pointing_state_list, dtype=int)


    
    if 'satellites' not in sim_data:
        sim_data['counts']['satellites'] = n
        sim_data['satellites'] = {
            'position': np.zeros((n, 3), dtype=float),
            'velocity': np.zeros((n, 3), dtype=float),
            'acceleration': np.zeros((n, 3), dtype=float),
            'orbital_elements': orbital_elements,
            'epochs': epochs_list,
            'pointing': np.zeros((n, 3), dtype=float),
            'pointing_state': pointing_state_array,
            'detector': np.zeros((n, 9), dtype=float),
        }
    else:
        # Append to existing satellites
        sim_data['counts']['satellites'] += n
        sim_data['satellites']['position'] = np.vstack([sim_data['satellites']['position'], np.zeros((n, 3), dtype=float)])
        sim_data['satellites']['velocity'] = np.vstack([sim_data['satellites']['velocity'], np.zeros((n, 3), dtype=float)])
        sim_data['satellites']['acceleration'] = np.vstack([sim_data['satellites']['acceleration'], np.zeros((n, 3), dtype=float)])
        sim_data['satellites']['orbital_elements'] = np.vstack([sim_data['satellites']['orbital_elements'], orbital_elements])
        sim_data['satellites']['epochs'].extend(epochs_list)
        sim_data['satellites']['pointing'] = np.vstack([sim_data['satellites']['pointing'], np.zeros((n, 3), dtype=float)])
        sim_data['satellites']['pointing_state'] = np.vstack([sim_data['satellites']['pointing_state'], pointing_state_array])
        sim_data['satellites']['detector'] = np.vstack([sim_data['satellites']['detector'], np.zeros((n, 9), dtype=float)])
