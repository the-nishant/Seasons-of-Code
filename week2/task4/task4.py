import numpy as np
from numpy import random as rnd
import math

def mean_filter(arr, k):
    pad = np.zeros(len(arr)+2*k,dtype = np.float64)
    pad[k:-k] = arr
    cum = (np.cumsum(pad))/(2*k+1)
    mean = np.zeros(len(arr))
    mean[0]=cum[2*k]
    mean[1:] = (cum[2*k+1:]-cum[:-2*k-1])
    return mean

def generate_sin_wave(period, range_, num):
    step = (range_[1]-range_[0])/num ;
    x = np.arange(range_[0],range_[1],step,dtype = np.float64)
    return np.sin(2*math.pi*x/period)

def noisify(array, var):
    return array + rnd.normal(0,math.sqrt(var),array.shape)






