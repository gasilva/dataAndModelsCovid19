from functools import lru_cache
from numba import njit
import math
import numpy as np
import latexify

@lru_cache(maxsize=None)
@njit #(parallel=True)
def sigmoid(x,betai,betaf):
    if betai<betaf:
        rx=1-(1 / (1 + math.exp(-x)))
        return betai*rx+betaf*(1-rx)
    else:
        rx=1 / (1 + math.exp(-x))
        return betaf*rx+betai*(1-rx)  
        
