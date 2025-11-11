# fibonacciSearch
# this contains the functions necessary to support a set of detectors
# that searches the sphere using a fibonachhi grid

import numpy as np
from pointing_vectors import pointing_vectors

def searchStruct(detect):
    '''
    creates the data structure for each of the satellite detectors,
    adds the structure to the detector
    '''
