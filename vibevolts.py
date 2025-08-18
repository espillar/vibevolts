import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
import io

# The astropy library is required for accurate astronomical calculations.
# You can install it with: pip install astropy jplephem
from astropy.time import Time
from astropy.coordinates import get_body, GCRS, ITRS, EarthLocation, solar_system_ephemeris
import astropy.units as u

# The sgp4 library is required for parsing TLE files.
# You can install it with: pip install sgp4
from sgp4.api import Satrec

# The plotly library is required for 3D plotting.
# You can install it with: pip install plotly
import plotly.graph_objects as go


# --- Global Constants for Array Indices ---
# These constants define column indices for numpy arrays, making the
# code more readable and preventing errors from using "magic numbers".

# -- Detector Array Indices --
DETECTOR_APERTURE_IDX = 0      # Aperture size in meters
DETECTOR_PIXEL_SIZE_IDX = 1    # Pixel size in radians
DETECTOR_QE_IDX = 2            # Quantum efficiency as a fraction (0.0 to 1.0)
DETECTOR_PIXELS_IDX = 3        # Total number of pixels in the detector (count)
DETECTOR_SOLAR_EXCL_IDX = 4    # Solar exclusion angle in radians
DETECTOR_LUNAR_EXCL_IDX = 5    # Lunar exclusion angle in radians
DETECTOR_EARTH_EXCL_IDX = 6    # Earth exclusion angle (above the limb) in radians

# -- Orbital Elements Array Indices --
ORBITAL_A_IDX = 0              # Semi-major axis in meters
ORBITAL_E_IDX = 1              # Eccentricity (dimensionless)
ORBITAL_I_IDX = 2              # Inclination in radians
ORBITAL_RAAN_IDX = 3           # Right Ascension of the Ascending Node in radians
ORBITAL_ARGP_IDX = 4           # Argument of Perigee in radians
ORBITAL_M_IDX = 5              # Mean Anomaly in radians


def initializeStructures(
    num_satellites: int,
    num_observatories: int,
    num_red_satellites: int,
    start_time: datetime
) -> Dict[str, Any]:
    """
    Initializes categorized data structures for a space simulation.

    This function creates distinct sets of data components for different categories
    of entities. It includes kinematic data, orbital elements, sensor pointing
    information, and detailed detector characteristics.

    The 'orbital_elements' component contains the following columns:
    - 0: Semi-major axis (meters)
    - 1: Eccentricity
    - 2: Inclination (radians)
    - 3: Right Ascension of the Ascending Node (radians)
    - 4: Argument of Perigee (radians)
    - 5: Mean Anomaly (radians)

    The 'detector' component contains the following columns:
    - 0: Aperture size (meters)
    - 1: Pixel size (radians)
    - 2: Quantum Efficiency (fraction)
    - 3: Pixel count (total number)
    - 4: Solar exclusion angle (radians)
    - 5: Lunar exclusion angle (radians)
    - 6: Earth exclusion angle (radians, above the limb)

    All kinematic data is for the Geocentric Celestial Reference System (GCRS).
    Angles are relative to the International Celestial Reference System (ICRS).
    Distances are in meters, angles in radians, and time in seconds.

    Args:
        num_satellites: The number of regular space-based sensors.
        num_observatories: The number of ground-based sensors.
        num_red_satellites: The number of special "red" satellites.
        start_time: The starting time and date of the simulation. This must be a
                    timezone-aware datetime object set to UTC.

    Returns:
        A dictionary representing the simulation state, containing categorized data.
    """
    # --- Input Validation ---
    if not isinstance(num_satellites, int) or num_satellites < 0:
        raise ValueError("num_satellites must be a non-negative integer.")
    if not isinstance(num_observatories, int) or num_observatories < 0:
        raise ValueError("num_observatories must be a non-negative integer.")
    if not isinstance(num_red_satellites, int) or num_red_satellites < 0:
        raise ValueError("num_red_satellites must be a non-negative integer.")
    if not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime object.")
    if start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware. Please set tzinfo.")


    # --- Main Data Structure ---
    simulation_data: Dict[str, Any] = {
        'start_time': start_time,
        'counts': {
            'celestial': 2,  # Sun and Moon
            'satellites': num_satellites,
            'observatories': num_observatories,
            'red_satellites': num_red_satellites
        },

        'celestial': {
            'position': np.zeros((2, 3), dtype=float),
            'velocity': np.zeros((2, 3), dtype=float),
            'acceleration': np.zeros((2, 3), dtype=float),
        },

        'satellites': {
            'position': np.zeros((num_satellites, 3), dtype=float),
            'velocity': np.zeros((num_satellites, 3), dtype=float),
            'acceleration': np.zeros((num_satellites, 3), dtype=float),
            'orbital_elements': np.zeros((num_satellites, 6), dtype=float),
            'epochs': [], # List to store datetime epochs for each satellite
            'pointing': np.zeros((num_satellites, 3), dtype=float),
            'detector': np.zeros((num_satellites, 7), dtype=float),
        },

        'observatories': {
            'position': np.zeros((num_observatories, 3), dtype=float),
            'velocity': np.zeros((num_observatories, 3), dtype=float),
            'acceleration': np.zeros((num_observatories, 3), dtype=float),
            'pointing': np.zeros((num_observatories, 3), dtype=float),
            'detector': np.zeros((num_observatories, 7), dtype=float),
        },
        
        'red_satellites': {
            'position': np.zeros((num_red_satellites, 3), dtype=float),
            'velocity': np.zeros((num_red_satellites, 3), dtype=float),
            'acceleration': np.zeros((num_red_satellites, 3), dtype=float),
            'orbital_elements': np.zeros((num_red_satellites, 6), dtype=float),
            'epochs': [],
            'pointing': np.zeros((num_red_satellites, 3), dtype=float),
            'detector': np.zeros((num_red_satellites, 7), dtype=float),
        }
    }
    
    return simulation_data

def celestial_update(data_struct: Dict[str, Any], time_date: datetime) -> Dict[str, Any]:
    """
    Calculates and updates the positions of the Sun and Moon.

    This function uses the astropy library to get the precise GCRS coordinates
    of the Sun and Moon for the given time, and updates the 'celestial' position
    component in the main data structure. Velocities and accelerations are not
    calculated and remain zero.

    Args:
        data_struct: The main simulation data dictionary from initializeStructures.
        time_date: The timezone-aware datetime object (in UTC) for the calculation.

    Returns:
        The modified data_struct with updated celestial positions.
    """
    if time_date.tzinfo is None:
        raise ValueError("time_date must be timezone-aware.")

    # Convert the python datetime object to an astropy Time object
    astro_time = Time(time_date)

    # Get Sun position in the GCRS frame using the modern get_body function
    sun_coords = get_body("sun", astro_time)
    sun_gcrs = sun_coords.transform_to(GCRS(obstime=astro_time))
    
    # Get Moon position in the GCRS frame using the modern get_body function
    moon_coords = get_body("moon", astro_time)
    moon_gcrs = moon_coords.transform_to(GCRS(obstime=astro_time))

    # Update the celestial data arrays (index 0 for Sun, 1 for Moon)
    # Position data is converted from astropy's representation to a simple
    # numpy array in meters.
    celestial_pos = data_struct['celestial']['position']
    celestial_pos[0] = sun_gcrs.cartesian.xyz.to(u.m).value
    celestial_pos[1] = moon_gcrs.cartesian.xyz.to(u.m).value
    
    # Per request, velocity and acceleration are not calculated and remain as zeros.

    return data_struct

def readtle(tle_file_path: str) -> Tuple[np.ndarray, List[datetime]]:
    """
    Reads a TLE file and extracts orbital elements and epochs for each satellite.

    Args:
        tle_file_path: The path to the TLE file.

    Returns:
        A tuple containing:
        - A NumPy array of orbital elements.
        - A list of datetime objects representing the epoch for each satellite.
    """
    orbital_elements_list = []
    epochs_list = []
    with open(tle_file_path, 'r') as f:
        lines = f.readlines()

    # Iterate through the file, processing 3 lines at a time (name, line1, line2)
    for i in range(0, len(lines), 3):
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()

        # Create a satellite object from the TLE data
        satellite = Satrec.twoline2rv(line1, line2)

        # Use the pre-calculated Julian date from the satellite object
        jd, fr = satellite.jdsatepoch, satellite.jdsatepochF
        epoch_dt = Time(jd, fr, format='jd', scale='utc').to_datetime(timezone.utc)
        epochs_list.append(epoch_dt)

        # Extract and convert orbital elements
        a = satellite.a * satellite.radiusearthkm * 1000.0
        e = satellite.ecco
        inc = satellite.inclo
        raan = satellite.nodeo
        argp = satellite.argpo
        M = satellite.mo

        elements = np.zeros(6)
        elements[ORBITAL_A_IDX] = a
        elements[ORBITAL_E_IDX] = e
        elements[ORBITAL_I_IDX] = inc
        elements[ORBITAL_RAAN_IDX] = raan
        elements[ORBITAL_ARGP_IDX] = argp
        elements[ORBITAL_M_IDX] = M
        orbital_elements_list.append(elements)

    return np.array(orbital_elements_list, dtype=float), epochs_list

def propagate_satellites(data_struct: Dict[str, Any], time_date: datetime) -> Dict[str, Any]:
    """
    Updates satellite positions based on their orbital elements for a given time.

    This function propagates the orbits of all satellites ('satellites' and 
    'red_satellites') from their TLE epoch to the specified time_date
    using Kepler's laws.

    Args:
        data_struct: The main simulation data dictionary.
        time_date: The timezone-aware datetime object (in UTC) to propagate to.

    Returns:
        The modified data_struct with updated satellite positions.
    """
    # Earth's standard gravitational parameter (m^3/s^2)
    MU_EARTH = 3.986004418e14
    
    time_date_timestamp = time_date.timestamp()

    for sat_category in ['satellites', 'red_satellites']:
        if data_struct['counts'][sat_category] == 0:
            continue

        elements = data_struct[sat_category]['orbital_elements']
        epochs = data_struct[sat_category]['epochs']
        
        # Calculate the time difference from each satellite's own epoch
        epoch_timestamps = np.array([e.timestamp() for e in epochs])
        delta_t_array = time_date_timestamp - epoch_timestamps

        # Extract orbital elements for all satellites in the category using constants
        a = elements[:, ORBITAL_A_IDX]
        e = elements[:, ORBITAL_E_IDX]
        i = elements[:, ORBITAL_I_IDX]
        raan = elements[:, ORBITAL_RAAN_IDX]
        argp = elements[:, ORBITAL_ARGP_IDX]
        M0 = elements[:, ORBITAL_M_IDX]

        # --- Vectorized Orbit Propagation ---
        
        # 1. Calculate mean motion
        n = np.sqrt(MU_EARTH / a**3)
        
        # 2. Calculate new mean anomaly
        M = (M0 + n * delta_t_array) % (2 * np.pi)

        # 3. Solve Kepler's Equation for Eccentric Anomaly (E) using Newton's method
        E = M.copy()  # Initial guess
        for _ in range(10): # Iterate a few times for convergence
            f_E = E - e * np.sin(E) - M
            f_prime_E = 1 - e * np.cos(E)
            # Avoid division by zero for circular orbits
            f_prime_E[f_prime_E == 0] = 1e-10
            E = E - f_E / f_prime_E

        # 4. Calculate True Anomaly (ν)
        tan_nu_half = np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)
        nu = 2 * np.arctan(tan_nu_half)

        # 5. Calculate distance from center of Earth (r)
        r = a * (1 - e * np.cos(E))

        # 6. Calculate position in the perifocal (orbital) frame
        x_pqw = r * np.cos(nu)
        y_pqw = r * np.sin(nu)

        # 7. Rotate from perifocal to GCRS frame
        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)
        cos_argp = np.cos(argp)
        sin_argp = np.sin(argp)
        cos_i = np.cos(i)
        sin_i = np.sin(i)

        # Rotation matrix elements from PQW to GCRS (IJK)
        P_x = cos_argp * cos_raan - sin_argp * sin_raan * cos_i
        P_y = cos_argp * sin_raan + sin_argp * cos_raan * cos_i
        P_z = sin_argp * sin_i
        
        Q_x = -sin_argp * cos_raan - cos_argp * sin_raan * cos_i
        Q_y = -sin_argp * sin_raan + cos_argp * cos_raan * cos_i
        Q_z = cos_argp * sin_i

        # Perform the rotation
        x_gcrs = x_pqw * P_x + y_pqw * Q_x
        y_gcrs = x_pqw * P_y + y_pqw * Q_y
        z_gcrs = x_pqw * P_z + y_pqw * Q_z
        
        # Update the position array in the data structure
        data_struct[sat_category]['position'] = np.vstack((x_gcrs, y_gcrs, z_gcrs)).T

    return data_struct

def plot_positions_3d(positions: np.ndarray, title: str, plot_time: datetime, labels: Optional[List[str]] = None):
    """
    Displays a 3D interactive plot of object positions with Earth references.

    Args:
        positions: An (n x 3) NumPy array of (x, y, z) positions in meters.
        title: The title for the plot.
        plot_time: The UTC datetime for which the plot is generated. This is
                   used to correctly orient the Earth.
        labels: An optional list of names for each point to display on hover.
    """
    if positions.shape[1] != 3:
        raise ValueError("positions array must have 3 columns (x, y, z).")
        
    earth_radius = 6378137.0 # meters

    fig = go.Figure()

    # Add satellite markers
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=np.arange(len(positions)), # Color by index
            colorscale='Viridis',
            opacity=0.8
        ),
        text=labels,
        hoverinfo='text' if labels else 'none',
        name='Satellites'
    ))

    # Add a sphere to represent the Earth
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))

    # Add Equator line
    theta = np.linspace(0, 2 * np.pi, 100)
    x_eq = earth_radius * np.cos(theta)
    y_eq = earth_radius * np.sin(theta)
    z_eq = np.zeros_like(theta)
    fig.add_trace(go.Scatter3d(x=x_eq, y=y_eq, z=z_eq, mode='lines', line=dict(color='green', width=3), name='Equator'))

    # Add North Pole marker
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[earth_radius * 1.1], mode='text', text=['N'], textfont=dict(size=15, color='red'), name='North Pole'))

    # Add El Segundo marker
    lat_es = 33.92 * u.deg
    lon_es = -118.42 * u.deg
    el_segundo_loc = EarthLocation.from_geodetic(lon=lon_es, lat=lat_es)
    itrs_coords = el_segundo_loc.get_itrs(obstime=Time(plot_time))
    gcrs_coords = itrs_coords.transform_to(GCRS(obstime=Time(plot_time)))
    es_pos = gcrs_coords.cartesian.xyz.to(u.m).value * 1.05 # Scale slightly for visibility
    fig.add_trace(go.Scatter3d(x=[es_pos[0]], y=[es_pos[1]], z=[es_pos[2]], mode='text', text=['ES'], textfont=dict(size=15, color='yellow'), name='El Segundo'))


    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data' # Ensures a 1:1:1 aspect ratio
        ),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Objects'
    )
    fig.show()


def solarexclusion(data_struct: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates solar exclusion for all satellites based on their pointing vectors.

    This function operates in a vectorized manner on all satellites in the
    'satellites' category. It computes the angle between each satellite's
    pointing vector and the vector from the satellite to the Sun.

    Args:
        data_struct: The main simulation data dictionary.

    Returns:
        A tuple containing:
        - exclusion_vector (np.ndarray): An array of the same length as the
          number of satellites. An element is 1 if the satellite is within
          the solar exclusion angle, 0 otherwise.
        - angle_vector (np.ndarray): An array containing the calculated angle
          in radians for each satellite.
    """
    num_sats = data_struct['counts']['satellites']
    if num_sats == 0:
        return np.array([]), np.array([])

    # Get required data from the main structure
    sun_pos = data_struct['celestial']['position'][0]
    sat_pos = data_struct['satellites']['position']
    sat_pointing = data_struct['satellites']['pointing']
    solar_exclusion_angles = data_struct['satellites']['detector'][:, DETECTOR_SOLAR_EXCL_IDX]

    # Calculate the vector from each satellite to the Sun
    vec_sat_to_sun = sun_pos - sat_pos

    # Normalize the vectors
    norm_sat_to_sun = np.linalg.norm(vec_sat_to_sun, axis=1)
    norm_sat_pointing = np.linalg.norm(sat_pointing, axis=1)

    # Avoid division by zero for zero-length vectors.
    # A zero-length pointing vector can't have an angle, so we create a
    # mask to handle these cases safely.
    valid_norms = (norm_sat_to_sun > 1e-9) & (norm_sat_pointing > 1e-9)

    # Initialize angle vector with a default value (pi = 180 deg) for invalid cases
    angle_vector = np.full(num_sats, np.pi)

    # Calculate dot product only for vectors with valid norms
    if np.any(valid_norms):
        dot_product = np.einsum('ij,ij->i', vec_sat_to_sun[valid_norms], sat_pointing[valid_norms])

        # Calculate the angle where possible
        cos_angle = dot_product / (norm_sat_to_sun[valid_norms] * norm_sat_pointing[valid_norms])

        # Clip to handle potential floating point inaccuracies
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle_vector[valid_norms] = np.arccos(cos_angle)

    # Determine exclusion based on the angle
    exclusion_vector = (angle_vector < solar_exclusion_angles).astype(int)

    return exclusion_vector, angle_vector


def demo1():
    """
    Runs a full demonstration of the simulation tools: initialization,
    celestial updates, TLE reading, satellite propagation, and 3D plotting.
    """
    # Define the simulation start time. It must be timezone-aware and set to UTC.
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    
    # --- TLE Reading and Initialization ---
    # Create a dummy TLE file for demonstration with more satellites
    tle_data = """ISS (ZARYA)
1 25544U 98067A   25209.52203988  .00012111  00000+0  22159-3 0  9991
2 25544  51.6412 254.9961 0006733  98.4322 261.6813 15.49493393462383
NOAA 19
1 33591U 09005A   25209.38959223  .00000100  00000+0  97987-4 0  9993
2 33591  99.1533 244.3362 0013327 101.3725 258.7562 14.12510122810029
HST
1 20580U 90037B   25208.83160218  .00000113  00000+0  35999-4 0  9990
2 20580  28.4695 177.8391 0001259 138.5273 221.5822 15.09326468 23453
GEO-01
1 90001U 25001A   25209.50000000  .00000000  00000-0  00000-0 0  9991
2 90001   0.0500   0.0000 0001000   0.0000   0.0000  1.00270000    12
GEO-02
1 90002U 25001B   25209.50000000  .00000000  00000-0  00000-0 0  9992
2 90002   0.0500  36.0000 0001000   0.0000   0.0000  1.00270000    13
GEO-03
1 90003U 25001C   25209.50000000  .00000000  00000-0  00000-0 0  9993
2 90003   0.0500  72.0000 0001000   0.0000  45.0000  1.00270000    14
GEO-04
1 90004U 25001D   25209.50000000  .00000000  00000-0  00000-0 0  9994
2 90004   0.0500 108.0000 0001000   0.0000  90.0000  1.00270000    15
GEO-05
1 90005U 25001E   25209.50000000  .00000000  00000-0  00000-0 0  9995
2 90005   0.0500 144.0000 0001000   0.0000 135.0000  1.00270000    16
GEO-06
1 90006U 25001F   25209.50000000  .00000000  00000-0  00000-0 0  9996
2 90006   0.0500 180.0000 0001000   0.0000 180.0000  1.00270000    17
GEO-07
1 90007U 25001G   25209.50000000  .00000000  00000-0  00000-0 0  9997
2 90007   0.0500 216.0000 0001000   0.0000 225.0000  1.00270000    18
GEO-08
1 90008U 25001H   25209.50000000  .00000000  00000-0  00000-0 0  9998
2 90008   0.0500 252.0000 0001000   0.0000 270.0000  1.00270000    19
GEO-09
1 90009U 25001I   25209.50000000  .00000000  00000-0  00000-0 0  9999
2 90009   0.0500 288.0000 0001000   0.0000 315.0000  1.00270000    10
GEO-10
1 90010U 25001J   25209.50000000  .00000000  00000-0  00000-0 0  9990
2 90010   0.0500 324.0000 0001000   0.0000 360.0000  1.00270000    11
HEO-01 (MOLNIYA)
1 90011U 25002A   25209.50000000  .00000000  00000-0  00000-0 0  9995
2 90011  63.4000  50.0000 7500000  270.0000  45.0000  2.00560000    13
HEO-02 (MOLNIYA)
1 90012U 25002B   25209.50000000  .00000000  00000-0  00000-0 0  9996
2 90012  63.4000 110.0000 7500000  270.0000  90.0000  2.00560000    14
HEO-03 (MOLNIYA)
1 90013U 25002C   25209.50000000  .00000000  00000-0  00000-0 0  9997
2 90013  63.4000 170.0000 7500000  270.0000 135.0000  2.00560000    15
HEO-04 (MOLNIYA)
1 90014U 25002D   25209.50000000  .00000000  00000-0  00000-0 0  9998
2 90014  63.4000 230.0000 7500000  270.0000 180.0000  2.00560000    16
HEO-05 (MOLNIYA)
1 90015U 25002E   25209.50000000  .00000000  00000-0  00000-0 0  9999
2 90015  63.4000 290.0000 7500000  270.0000 225.0000  2.00560000    17
LEO-01 (POLAR)
1 90016U 25003A   25209.50000000  .00000000  00000-0  00000-0 0  9990
2 90016  98.0000 120.0000 0010000  90.0000  20.0000 14.50000000    18
LEO-02 (POLAR)
1 90017U 25003B   25209.50000000  .00000000  00000-0  00000-0 0  9991
2 90017  98.0000 240.0000 0010000  90.0000  40.0000 14.50000000    19
LEO-03
1 90018U 25003C   25209.50000000  .00000000  00000-0  00000-0 0  9992
2 90018  45.0000  80.0000 0010000  45.0000  60.0000 15.00000000    10
LEO-04
1 90019U 25003D   25209.50000000  .00000000  00000-0  00000-0 0  9993
2 90019  45.0000 200.0000 0010000  45.0000  80.0000 15.00000000    11
LEO-05
1 90020U 25003E   25209.50000000  .00000000  00000-0  00000-0 0  9994
2 90020  28.5000  10.0000 0010000  10.0000 100.0000 15.50000000    12
"""
    dummy_tle_path = "dummy_tle.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)
    
    # Read the TLEs to determine the number of satellites
    orbital_elements_from_tle, epochs_from_tle = readtle(dummy_tle_path)
    num_sats = len(orbital_elements_from_tle)
    num_obs = 3
    num_red_sats = 0 # No red satellites in this TLE file

    print(f"Initializing structures for simulation starting at {sim_start_time.isoformat()}")
    print(f"Counts: {num_sats} satellites, {num_obs} observatories, {num_red_sats} red satellites.\n")
    
    sim_data = initializeStructures(
        num_satellites=num_sats,
        num_observatories=num_obs,
        num_red_satellites=num_red_sats,
        start_time=sim_start_time
    )
    
    # Populate the satellite orbital elements from the TLE file
    sim_data['satellites']['orbital_elements'] = orbital_elements_from_tle
    sim_data['satellites']['epochs'] = epochs_from_tle
    
    # Ensure the required ephemeris data is available
    print("\nEnsuring planetary ephemeris data is available (may download on first run)...")
    solar_system_ephemeris.set('jpl')
    print("Ephemeris data is ready.")

    # --- Celestial Body Updates ---
    sim_data = celestial_update(sim_data, sim_start_time)
    print("\n--- Celestial Positions at Start Time ---")
    print(f"Time: {sim_start_time.isoformat()}")
    print(sim_data['celestial']['position'])

    # --- Satellite Propagation and Plotting ---
    # First propagation time
    time_t1 = sim_start_time + timedelta(hours=1, minutes=30)
    print(f"\n--- Propagating satellites to T1: {time_t1.isoformat()} ---")
    sim_data = propagate_satellites(sim_data, time_t1)
    positions_t1 = sim_data['satellites']['position'].copy() # Important to copy the data
    
    # Second propagation time, 10 minutes later
    time_t2 = time_t1 + timedelta(minutes=10)
    print(f"\n--- Propagating satellites to T2: {time_t2.isoformat()} ---")
    sim_data = propagate_satellites(sim_data, time_t2)
    positions_t2 = sim_data['satellites']['position'].copy()

    # --- 3D Plotting of Both Time Steps ---
    print("\n--- Generating 3D plot of satellite positions at two time steps ---")
    satellite_names = [line.strip() for i, line in enumerate(tle_data.strip().split('\n')) if i % 3 == 0]
    earth_radius = 6378137.0 # meters
    fig = go.Figure()

    # Add positions at T1
    fig.add_trace(go.Scatter3d(
        x=positions_t1[:, 0], y=positions_t1[:, 1], z=positions_t1[:, 2],
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.8),
        text=satellite_names, hoverinfo='text', name=f'Positions at T1'
    ))

    # Add positions at T2
    fig.add_trace(go.Scatter3d(
        x=positions_t2[:, 0], y=positions_t2[:, 1], z=positions_t2[:, 2],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8),
        text=satellite_names, hoverinfo='text', name=f'Positions at T2 (+10 min)'
    ))

    # Add Earth sphere and reference markers
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))
    
    theta = np.linspace(0, 2 * np.pi, 100)
    x_eq = earth_radius * np.cos(theta)
    y_eq = earth_radius * np.sin(theta)
    z_eq = np.zeros_like(theta)
    fig.add_trace(go.Scatter3d(x=x_eq, y=y_eq, z=z_eq, mode='lines', line=dict(color='green', width=3), name='Equator'))
    
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[earth_radius * 1.1], mode='text', text=['N'], textfont=dict(size=15, color='red'), name='North Pole'))
    
    lat_es = 33.92 * u.deg
    lon_es = -118.42 * u.deg
    el_segundo_loc = EarthLocation.from_geodetic(lon=lon_es, lat=lat_es)
    itrs_coords = el_segundo_loc.get_itrs(obstime=Time(time_t2)) # Use final time for Earth orientation
    gcrs_coords = itrs_coords.transform_to(GCRS(obstime=Time(time_t2)))
    es_pos = gcrs_coords.cartesian.xyz.to(u.m).value * 1.05
    fig.add_trace(go.Scatter3d(x=[es_pos[0]], y=[es_pos[1]], z=[es_pos[2]], mode='text', text=['ES'], textfont=dict(size=15, color='yellow'), name='El Segundo'))

    fig.update_layout(
        title=f"Satellite Positions at Two Time Steps",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Objects'
    )
    fig.show()


def demo2():
    """
    Runs a second demonstration with 10 LEO satellites, plotting their
    positions and celestial vectors at 0, 60, and 300 seconds.
    """
    # Define the simulation start time.
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    
    # --- TLE Reading and Initialization ---
    # Create a dummy TLE file for 10 LEO satellites
    tle_data = """LEO-DEMO-1
1 90101U 25004A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90101  97.5000  10.0000 0010000  90.0000  20.0000 15.50000000    11
LEO-DEMO-2
1 90102U 25004B   25210.50000000  .00000000  00000-0  00000-0 0  9992
2 90102  97.5000  46.0000 0010000  90.0000  20.0000 15.50000000    12
LEO-DEMO-3
1 90103U 25004C   25210.50000000  .00000000  00000-0  00000-0 0  9993
2 90103  97.5000  82.0000 0010000  90.0000  20.0000 15.50000000    13
LEO-DEMO-4
1 90104U 25004D   25210.50000000  .00000000  00000-0  00000-0 0  9994
2 90104  97.5000 118.0000 0010000  90.0000  20.0000 15.50000000    14
LEO-DEMO-5
1 90105U 25004E   25210.50000000  .00000000  00000-0  00000-0 0  9995
2 90105  97.5000 154.0000 0010000  90.0000  20.0000 15.50000000    15
LEO-DEMO-6
1 90106U 25004F   25210.50000000  .00000000  00000-0  00000-0 0  9996
2 90106  97.5000 190.0000 0010000  90.0000  20.0000 15.50000000    16
LEO-DEMO-7
1 90107U 25004G   25210.50000000  .00000000  00000-0  00000-0 0  9997
2 90107  97.5000 226.0000 0010000  90.0000  20.0000 15.50000000    17
LEO-DEMO-8
1 90108U 25004H   25210.50000000  .00000000  00000-0  00000-0 0  9998
2 90108  97.5000 262.0000 0010000  90.0000  20.0000 15.50000000    18
LEO-DEMO-9
1 90109U 25004I   25210.50000000  .00000000  00000-0  00000-0 0  9999
2 90109  97.5000 298.0000 0010000  90.0000  20.0000 15.50000000    19
LEO-DEMO-10
1 90110U 25004J   25210.50000000  .00000000  00000-0  00000-0 0  9990
2 90110  97.5000 334.0000 0010000  90.0000  20.0000 15.50000000    10
"""
    dummy_tle_path = "dummy_tle2.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    orbital_elements_from_tle, epochs_from_tle = readtle(dummy_tle_path)
    num_sats = len(orbital_elements_from_tle)

    print(f"\n--- Starting Demo 2 ---")
    print(f"Initializing structures for {num_sats} LEO satellites.")
    
    sim_data = initializeStructures(
        num_satellites=num_sats,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )
    sim_data['satellites']['orbital_elements'] = orbital_elements_from_tle
    sim_data['satellites']['epochs'] = epochs_from_tle
    
    # --- Satellite and Celestial Propagation ---
    time_t0 = sim_start_time
    sim_data = propagate_satellites(sim_data, time_t0)
    positions_t0 = sim_data['satellites']['position'].copy()
    sim_data = celestial_update(sim_data, time_t0)
    celestial_pos_t0 = sim_data['celestial']['position'].copy()
    
    time_t1 = sim_start_time + timedelta(seconds=60)
    sim_data = propagate_satellites(sim_data, time_t1)
    positions_t1 = sim_data['satellites']['position'].copy()
    sim_data = celestial_update(sim_data, time_t1)
    celestial_pos_t1 = sim_data['celestial']['position'].copy()

    time_t2 = sim_start_time + timedelta(seconds=300)
    sim_data = propagate_satellites(sim_data, time_t2)
    positions_t2 = sim_data['satellites']['position'].copy()
    sim_data = celestial_update(sim_data, time_t2)
    celestial_pos_t2 = sim_data['celestial']['position'].copy()

    # --- 3D Plotting of All Time Steps ---
    print("\n--- Generating 3D plot for Demo 2 ---")
    satellite_names = [line.strip() for i, line in enumerate(tle_data.strip().split('\n')) if i % 3 == 0]
    earth_radius = 6378137.0
    vector_scale = 2.5 * earth_radius
    fig = go.Figure()

    # Add positions at T0, T1, and T2
    fig.add_trace(go.Scatter3d(
        x=positions_t0[:, 0], y=positions_t0[:, 1], z=positions_t0[:, 2],
        mode='markers', marker=dict(size=5, color='blue'),
        text=satellite_names, hoverinfo='text', name='Sats (T=0s)'
    ))
    fig.add_trace(go.Scatter3d(
        x=positions_t1[:, 0], y=positions_t1[:, 1], z=positions_t1[:, 2],
        mode='markers', marker=dict(size=5, color='red'),
        text=satellite_names, hoverinfo='text', name='Sats (T=60s)'
    ))
    fig.add_trace(go.Scatter3d(
        x=positions_t2[:, 0], y=positions_t2[:, 1], z=positions_t2[:, 2],
        mode='markers', marker=dict(size=5, color='green'),
        text=satellite_names, hoverinfo='text', name='Sats (T=300s)'
    ))
    
    # Add Celestial Vectors
    # T=0s
    sun_vec_t0 = celestial_pos_t0[0] / np.linalg.norm(celestial_pos_t0[0]) * vector_scale
    moon_vec_t0 = celestial_pos_t0[1] / np.linalg.norm(celestial_pos_t0[1]) * vector_scale
    fig.add_trace(go.Scatter3d(x=[0, sun_vec_t0[0]], y=[0, sun_vec_t0[1]], z=[0, sun_vec_t0[2]], mode='lines', line=dict(color='blue', width=4), name='Sun (T=0s)'))
    fig.add_trace(go.Scatter3d(x=[0, moon_vec_t0[0]], y=[0, moon_vec_t0[1]], z=[0, moon_vec_t0[2]], mode='lines', line=dict(color='cyan', width=4), name='Moon (T=0s)'))
    
    # T=60s
    sun_vec_t1 = celestial_pos_t1[0] / np.linalg.norm(celestial_pos_t1[0]) * vector_scale
    moon_vec_t1 = celestial_pos_t1[1] / np.linalg.norm(celestial_pos_t1[1]) * vector_scale
    fig.add_trace(go.Scatter3d(x=[0, sun_vec_t1[0]], y=[0, sun_vec_t1[1]], z=[0, sun_vec_t1[2]], mode='lines', line=dict(color='red', width=4), name='Sun (T=60s)'))
    fig.add_trace(go.Scatter3d(x=[0, moon_vec_t1[0]], y=[0, moon_vec_t1[1]], z=[0, moon_vec_t1[2]], mode='lines', line=dict(color='magenta', width=4), name='Moon (T=60s)'))

    # T=300s
    sun_vec_t2 = celestial_pos_t2[0] / np.linalg.norm(celestial_pos_t2[0]) * vector_scale
    moon_vec_t2 = celestial_pos_t2[1] / np.linalg.norm(celestial_pos_t2[1]) * vector_scale
    fig.add_trace(go.Scatter3d(x=[0, sun_vec_t2[0]], y=[0, sun_vec_t2[1]], z=[0, sun_vec_t2[2]], mode='lines', line=dict(color='green', width=4), name='Sun (T=300s)'))
    fig.add_trace(go.Scatter3d(x=[0, moon_vec_t2[0]], y=[0, moon_vec_t2[1]], z=[0, moon_vec_t2[2]], mode='lines', line=dict(color='lime', width=4), name='Moon (T=300s)'))


    # Add Earth and reference markers
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))

    fig.update_layout(
        title=f"LEO Satellite Positions at 0, 60, and 300 seconds",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Time Step'
    )
    fig.show()

def demo3():
    """
    Runs a third demonstration with a single LEO satellite, plotting its
    trajectory over 90 minutes.
    """
    # Define the simulation start time.
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    
    # --- TLE Reading and Initialization ---
    # Create a dummy TLE file for one LEO satellite
    tle_data = """LEO-TRAJECTORY
1 90201U 25005A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90201  51.6000  10.0000 0010000  90.0000  20.0000 15.50000000    11
"""
    dummy_tle_path = "dummy_tle3.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    orbital_elements_from_tle, epochs_from_tle = readtle(dummy_tle_path)
    num_sats = len(orbital_elements_from_tle)

    print(f"\n--- Starting Demo 3 ---")
    print(f"Initializing structures for {num_sats} LEO satellite.")
    
    sim_data = initializeStructures(
        num_satellites=num_sats,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )
    sim_data['satellites']['orbital_elements'] = orbital_elements_from_tle
    sim_data['satellites']['epochs'] = epochs_from_tle
    
    # --- Satellite Propagation ---
    positions_over_time = []
    time_steps = np.arange(0, 91, 10) # 0 to 90 minutes in 10 minute steps
    
    for minutes in time_steps:
        prop_time = sim_start_time + timedelta(minutes=int(minutes))
        sim_data = propagate_satellites(sim_data, prop_time)
        positions_over_time.append(sim_data['satellites']['position'][0])

    positions_array = np.array(positions_over_time)

    # --- 3D Plotting of the Trajectory ---
    print("\n--- Generating 3D plot for Demo 3 ---")
    earth_radius = 6378137.0
    fig = go.Figure()

    # Add the trajectory line
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='lines', line=dict(color='red', width=4), name='Trajectory'
    ))
    
    # Add markers for each time step
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='markers', marker=dict(size=5, color='blue'),
        text=[f'T={t} min' for t in time_steps], hoverinfo='text', name='Time Steps'
    ))

    # Add Earth and reference markers
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))

    fig.update_layout(
        title=f"Single LEO Satellite Trajectory over 90 Minutes",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Trace'
    )
    fig.show()

def demo5():
    """
    Demonstrates the use of the solarexclusion function.

    This test sets up a scenario with 3 satellites and specific pointing
    vectors to verify the solar exclusion logic.
    - Sat 1: Points directly at the Sun (should be excluded).
    - Sat 2: Points perpendicular to the Sun (should not be excluded).
    - Sat 3: Points away from the Sun (should not be excluded).
    """
    print("\n--- Starting Demo 5: Solar Exclusion Test ---")

    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)

    # --- Initialization for 3 satellites ---
    tle_data = """SAT-1
1 90401U 25007A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90401  51.6412 254.9961 0006733  98.4322 261.6813 15.49493393462383
SAT-2
1 90402U 25007B   25210.50000000  .00000000  00000-0  00000-0 0  9992
2 90402  99.1533 244.3362 0013327 101.3725 258.7562 14.12510122810029
SAT-3
1 90403U 25007C   25210.50000000  .00000000  00000-0  00000-0 0  9993
2 90403  28.4695 177.8391 0001259 138.5273 221.5822 15.09326468 23453
"""
    dummy_tle_path = "dummy_tle5.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    orbital_elements_from_tle, epochs_from_tle = readtle(dummy_tle_path)
    num_sats = len(orbital_elements_from_tle)

    sim_data = initializeStructures(
        num_satellites=num_sats,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )
    sim_data['satellites']['orbital_elements'] = orbital_elements_from_tle
    sim_data['satellites']['epochs'] = epochs_from_tle

    # --- Set up test conditions ---
    # Propagate to get initial positions
    sim_data = propagate_satellites(sim_data, sim_start_time)
    sim_data = celestial_update(sim_data, sim_start_time)

    sun_pos = sim_data['celestial']['position'][0]
    sat_pos = sim_data['satellites']['position']

    # Define pointing vectors for the test case
    vec_sat_to_sun = sun_pos - sat_pos

    # Sat 1: Point directly at the Sun
    sim_data['satellites']['pointing'][0] = vec_sat_to_sun[0]

    # Sat 2: Point perpendicular to the Sun vector
    # Create a perpendicular vector by swapping and negating components
    perp_vec = np.array([-vec_sat_to_sun[1, 1], vec_sat_to_sun[1, 0], 0])
    sim_data['satellites']['pointing'][1] = perp_vec

    # Sat 3: Point directly away from the Sun
    sim_data['satellites']['pointing'][2] = -vec_sat_to_sun[2]

    # Set solar exclusion angles (in radians)
    # Sat 1 should be excluded, Sat 2 and 3 should not.
    solar_exclusion_angles = np.array([
        0.2,  # 11.5 degrees, angle should be ~0, so excluded
        0.1,  #  5.7 degrees, angle should be ~pi/2, so not excluded
        0.1   #  5.7 degrees, angle should be ~pi, so not excluded
    ])
    sim_data['satellites']['detector'][:, DETECTOR_SOLAR_EXCL_IDX] = solar_exclusion_angles

    # --- Run the function ---
    exclusion_vec, angle_vec = solarexclusion(sim_data)

    # --- Print results ---
    print("\n--- Input Data ---")
    print(f"Solar Exclusion Angles (rad): {np.round(solar_exclusion_angles, 2)}")
    # Pointing vectors are too long to print neatly, but their setup is described above.

    print("\n--- Output Data ---")
    print(f"Calculated Angles (rad): {np.round(angle_vec, 2)}")
    print(f"Exclusion Vector: {exclusion_vec}")

    print("\n--- Verification ---")
    # Expected: Sat 1 angle is ~0, Sat 2 is ~pi/2, Sat 3 is ~pi
    # Expected: Exclusion vector is [1 0 0]
    expected_exclusion = np.array([1, 0, 0])
    if np.array_equal(exclusion_vec, expected_exclusion):
        print("SUCCESS: Exclusion vector matches expected output [1 0 0].")
    else:
        print(f"FAILURE: Exclusion vector was {exclusion_vec}, expected {expected_exclusion}.")

def demo4():
    """
    Runs a fourth demonstration with a single GEO satellite, plotting its
    trajectory over 23 hours.
    """
    # Define the simulation start time.
    sim_start_time = datetime(2025, 7, 27, 22, 27, 0, tzinfo=timezone.utc)
    
    # --- TLE Reading and Initialization ---
    # Create a dummy TLE file for one GEO satellite
    tle_data = """GEO-TRAJECTORY
1 90301U 25006A   25210.50000000  .00000000  00000-0  00000-0 0  9991
2 90301   0.0500  45.0000 0001000  90.0000  20.0000  1.00270000    11
"""
    dummy_tle_path = "dummy_tle4.txt"
    with open(dummy_tle_path, "w") as f:
        f.write(tle_data)

    orbital_elements_from_tle, epochs_from_tle = readtle(dummy_tle_path)
    num_sats = len(orbital_elements_from_tle)

    print(f"\n--- Starting Demo 4 ---")
    print(f"Initializing structures for {num_sats} GEO satellite.")
    
    sim_data = initializeStructures(
        num_satellites=num_sats,
        num_observatories=0,
        num_red_satellites=0,
        start_time=sim_start_time
    )
    sim_data['satellites']['orbital_elements'] = orbital_elements_from_tle
    sim_data['satellites']['epochs'] = epochs_from_tle
    
    # --- Satellite Propagation ---
    positions_over_time = []
    time_steps = np.arange(0, 24, 1) # 0 to 23 hours in 1 hour steps
    
    for hours in time_steps:
        prop_time = sim_start_time + timedelta(hours=int(hours))
        sim_data = propagate_satellites(sim_data, prop_time)
        positions_over_time.append(sim_data['satellites']['position'][0])

    positions_array = np.array(positions_over_time)

    # --- 3D Plotting of the Trajectory ---
    print("\n--- Generating 3D plot for Demo 4 ---")
    earth_radius = 6378137.0
    fig = go.Figure()

    # Add the trajectory line
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='lines', line=dict(color='purple', width=4), name='Trajectory'
    ))
    
    # Add markers for each time step
    fig.add_trace(go.Scatter3d(
        x=positions_array[:, 0], y=positions_array[:, 1], z=positions_array[:, 2],
        mode='markers', marker=dict(size=5, color='orange'),
        text=[f'T={t} hr' for t in time_steps], hoverinfo='text', name='Time Steps'
    ))

    # Add Earth and reference markers
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 100)
    x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    fig.add_trace(go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale='Blues', showscale=False, opacity=0.5, name='Earth'))

    fig.update_layout(
        title=f"Single GEO Satellite Trajectory over 23 Hours",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(r=20, b=10, l=10, t=40),
        legend_title_text='Trace'
    )
    fig.show()

# --- Main Execution Block ---
if __name__ == '__main__':
    demo1()
    demo2()
    demo3()
    demo4()
    demo5()
