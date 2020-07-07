from functools import lru_cache
from numba import njit
import math
import numpy as np

# @lru_cache(maxsize=None)
@njit #(parallel=True)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# @lru_cache(maxsize=None)
@njit #(parallel=True)
def betaX(t,tmax=280,beta0=0,beta1=0,beta01=0,t0=0,t1=0,twoPoints=False):
    if twoPoints:
        tfrac=t/tmax
        beta=0    
        if tfrac<0.5:
            beta1=max(beta0,beta1)
            rx=sigmoid(t-t0)
            beta=beta1*rx+beta0*(1-rx)
        if tfrac>=0.5:
            beta1=max(beta1,beta01)
            rx=sigmoid(t-t1)
            beta=beta01*rx+beta1*(1-rx)
    else:
        rx=sigmoid(t-t0)
        beta=beta1*rx+beta0*(1-rx)
    return beta
        
