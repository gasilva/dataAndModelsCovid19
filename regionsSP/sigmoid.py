from functools import lru_cache
from numba import njit
import math
import numpy as np

@njit #(parallel=True)
def sigmoid(x,vari,varf):
    rx=1 / (1 + math.exp(-x))
    return vari*(1-rx)+varf*rx

@lru_cache(maxsize=None)
@njit
def sigmoid2(x,xff,vari,varf,varff,half):        
    if half<=0:
        return sigmoid(x,vari,varf)
    else:
        return sigmoid(xff,varf,varff)