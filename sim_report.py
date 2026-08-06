import numpy as np
from astropy.coordinates import EarthLocation, GCRS, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u

from constants import (
    ORBITAL_A_IDX,
    ORBITAL_E_IDX,
    ORBITAL_I_IDX,
    ORBITAL_RAAN_IDX,
    ORBITAL_ARGP_IDX,
    ORBITAL_M_IDX
)

def print_simulation_state(sim_data):
    """
    Prints the current state of the simulation in tabulated format.
    Includes satellites (orbital elements, pointing RA/Dec),
    observatories (lat/lon/alt, pointing Az/El), and exclusion parameters.
    """
    print("\n" + "="*80)
    print("SIMULATION STATE REPORT")
    print(f"Time: {sim_data.time}")
    print("="*80)

    # 1. Satellites
    print("\n--- SATELLITES ---")
    if 'satellites' in sim_data.counts and sim_data.counts.satellites > 0:
        num_sats = sim_data.counts.satellites
        print(f"{'Idx':<4} | {'Semi-Major (m)':<15} | {'Eccentricity':<12} | {'Inc (deg)':<9} | {'RAAN (deg)':<10} | {'Sensor Band':<11} | {'Pointing RA (deg)':<17} | {'Pointing Dec (deg)':<18}")
        print("-" * 115)
        
        # Build lookup array for detectors safely if detector is present
        has_detectors = (sim_data.detector is not None and len(sim_data.detector.filt) > 0)
        if has_detectors:
            det_cat = np.array(sim_data.detector.category)
            det_idx = sim_data.detector.asset_index
        else:
            det_cat, det_idx = np.array([]), np.array([])
        
        for i in range(num_sats):
            oe = sim_data.satellites.orbital_elements[i]
            a = oe[ORBITAL_A_IDX]
            e = oe[ORBITAL_E_IDX]
            inc = np.rad2deg(oe[ORBITAL_I_IDX])
            raan = np.rad2deg(oe[ORBITAL_RAAN_IDX])
            
            match = np.where((det_cat == 'satellites') & (det_idx == i))[0]
            if len(match) > 0:
                d_i = match[0]
                band = sim_data.detector.filt[d_i]
                pt = sim_data.detector.pointing[d_i]
                # Pointing vector is in GCRS
                ra = np.rad2deg(np.arctan2(pt[1], pt[0])) % 360
                dec = np.rad2deg(np.arcsin(pt[2]))
                band_str = band
                ra_str = f"{ra:.2f}"
                dec_str = f"{dec:.2f}"
            else:
                band_str = "None"
                ra_str = "N/A"
                dec_str = "N/A"
                
            print(f"{i:<4} | {a:<15.2f} | {e:<12.6f} | {inc:<9.2f} | {raan:<10.2f} | {band_str:<11} | {ra_str:<17} | {dec_str:<18}")
    else:
        print("No satellites in simulation.")

    # 2. Observatories
    print("\n--- OBSERVATORIES ---")
    if 'observatories' in sim_data.counts and sim_data.counts.observatories > 0:
        num_obs = sim_data.counts.observatories
        print(f"{'Idx':<4} | {'Lat (deg)':<9} | {'Lon (deg)':<10} | {'Alt (m)':<9} | {'Sensor Band':<11} | {'Pointing Az (deg)':<17} | {'Pointing El (deg)':<18}")
        print("-" * 95)
        
        if has_detectors:
            det_cat = np.array(sim_data.detector.category)
            det_idx = sim_data.detector.asset_index
        else:
            det_cat, det_idx = np.array([]), np.array([])
            
        obs_time = Time(sim_data.time)

        for i in range(num_obs):
            lat = sim_data.observatories.latitude[i]
            lon = sim_data.observatories.longitude[i]
            alt = sim_data.observatories.altitude[i]
            
            match = np.where((det_cat == 'observatories') & (det_idx == i))[0]
            if len(match) > 0:
                d_i = match[0]
                band = sim_data.detector.filt[d_i]
                pt = sim_data.detector.pointing[d_i]
                
                # Pointing vector is in GCRS. Convert to AltAz frame.
                loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt*u.m)
                coord = SkyCoord(x=pt[0], y=pt[1], z=pt[2], representation_type='cartesian', frame=GCRS(obstime=obs_time))
                altaz = coord.transform_to(AltAz(obstime=obs_time, location=loc))
                
                az = altaz.az.deg
                el = altaz.alt.deg
                
                band_str = band
                az_str = f"{az:.2f}"
                el_str = f"{el:.2f}"
            else:
                band_str = "None"
                az_str = "N/A"
                el_str = "N/A"
                
            print(f"{i:<4} | {lat:<9.4f} | {lon:<10.4f} | {alt:<9.1f} | {band_str:<11} | {az_str:<17} | {el_str:<18}")
    else:
        print("No observatories in simulation.")

    # 3. Exclusion Parameters
    print("\n--- EXCLUSION PARAMETERS ---")
    if has_detectors:
        print(f"{'Category':<15} | {'Asset Idx':<9} | {'Solar Excl (deg)':<16} | {'Lunar Excl (deg)':<16} | {'Earth Excl (deg)':<16}")
        print("-" * 80)
        
        for d_i in range(len(sim_data.detector.filt)):
            cat = sim_data.detector.category[d_i]
            a_idx = sim_data.detector.asset_index[d_i]
            solar_ex = np.rad2deg(sim_data.detector.solarEx[d_i])
            lunar_ex = np.rad2deg(sim_data.detector.lunarEx[d_i])
            earth_ex = np.rad2deg(sim_data.detector.earthEx[d_i])
            
            print(f"{cat:<15} | {a_idx:<9} | {solar_ex:<16.2f} | {lunar_ex:<16.2f} | {earth_ex:<16.2f}")
    else:
        print("No detectors configured.")
        
    print("="*80 + "\n")
