from functools import lru_cache
from numba import njit
import math
import numpy as np

@njit
def gammaNew(modelDeath,t,a1=0,b1=0,c1=0,d1=0,e1=0,f1=0):
    if modelDeath=='constant':
        return a1+b1
    if modelDeath=='linear':
        return a1*t+c1+b1
    if modelDeath=='harmonic':
        return a1*np.cos(d1*t+e1)+c1+b1
    if modelDeath=='exponential':
        return -a1*np.exp(-d1*t)+c1+b1
    else:
        return a1+b1