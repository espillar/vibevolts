# --- Global Constants for Array Indices ---
# These constants define column indices for numpy arrays, making the
# code more readable and preventing errors from using "magic numbers".

# -- Radii in Meters --
EARTH_RADIUS = 6378137.0
MOON_RADIUS = 1737400.0

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

# -- Pointing State Array Indices --
POINTING_COUNT_IDX = 0         # Number of points in the pointing grid
POINTING_PLACE_IDX = 1         # Current index in the pointing sequence
