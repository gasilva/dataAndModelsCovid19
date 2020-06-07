from functools import lru_cache
from numba import njit
import math

@lru_cache(maxsize=None)
@njit(parallel=True)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))