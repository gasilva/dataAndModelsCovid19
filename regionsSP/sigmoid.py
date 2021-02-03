from functools import lru_cache
from numba import njit
import math
import numpy as np

@njit #(parallel=True)
def sigmoid(x,vari,varf):
    if vari<varf:
        rx=1-(1 / (1 + math.exp(-x)))
        return vari*rx+varf*(1-rx)
    else:
        rx=1 / (1 + math.exp(-x))
        return varf*rx+vari*(1-rx)  
    
@lru_cache(maxsize=None)
@njit
def sigmoid2(x,xff,vari,varf,varff,half):        
    
    if half<=0:
        return sigmoid(x,vari,varf)
    else:
        return sigmoid(xff,varf,varff)