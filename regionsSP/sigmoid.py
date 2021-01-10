from functools import lru_cache
from numba import njit
import math
import numpy as np

@njit #(parallel=True)
def sigmoid(x,betai,betaf):
    if betai<betaf:
        rx=1-(1 / (1 + math.exp(-x)))
        return betai*rx+betaf*(1-rx)
    else:
        rx=1 / (1 + math.exp(-x))
        return betaf*rx+betai*(1-rx)  
    
@lru_cache(maxsize=None)
@njit
def sigmoid2(x,xff,betai,betaf,betaff,half):        
    
    if half<=0:
        return sigmoid(x,betai,betaf)
    else:
        return sigmoid(xff,betaf,betaff)