# --- Global Constants for VibeVolts ---
# Physical constants (radii, angles) and column indices for NumPy arrays,
# making the code more readable and avoiding magic numbers.
from math import pi


# -- Radii in Meters --
EARTH_RADIUS = 6378137.0
MOON_RADIUS = 1737400.0
GEO_RADIUS = 42164140.0
MOON_ORBIT_RADIUS = 384400000.0


# -- Some UsefulConstants --
ARCSEC = 2.0 *pi/(360*3600)
DEGREE = 3600 * ARCSEC



# -- Orbital Elements Array Indices --
ORBITAL_A_IDX = 0              # Semi-major axis in meters
ORBITAL_E_IDX = 1              # Eccentricity (dimensionless)
ORBITAL_I_IDX = 2              # Inclination in radians
ORBITAL_RAAN_IDX = 3           # Right Ascension of the Ascending Node in radians
ORBITAL_ARGP_IDX = 4           # Argument of Perigee in radians
ORBITAL_M_IDX = 5              # Mean Anomaly in radians

# -- Pointing State Array Indices --
POINTING_COUNT_IDX = 0         # Number of points in the pointing grid
POINTING_PLACE_IDX = 1         # Current index in the pointing sequence
