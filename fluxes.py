from radiometry_data import *
from radiometry_calcs import *
from constants import ARCSEC
def fluxes(band):
    """
    uses the FILTER_DATA table from radiometry_data.py for data
    Looks up in formation based on the argument band, which is 
    usually something like an astronomical band... U, B, V, etc.
    It returns three numbers:
    sun which is the solar flux at earth in photons/s/m^2
    sky which is the sky brightness at earth in p/s/asec^2/m^2
    space, sky brightness in space in p/s/asec^2/m^2
    """
    x = FILTER_DATA[band]
    zp = x['zero_point']
    sun = amag(x['sun']) * zp
    space = amag(x['space']) * zp / (ARCSEC**2)
    sky = amag(x['sky']) * zp / (ARCSEC**2)
    return(sun, space, sky)
    
