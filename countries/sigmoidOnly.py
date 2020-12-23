from functools import lru_cache
from numba import njit
import math
import numpy as np

@lru_cache(maxsize=None)
@njit #(parallel=True)
def sigmoid(x,betai,betaf):
    if betai<betaf:
        return 1-(1 / (1 + math.exp(-x)))
    else:
        return 1 / (1 + math.exp(-x))
        
