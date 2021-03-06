# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial.distance as scipy_spatial

def compute(X,Y,length_scale):
    #sqdist=scipy.spatial.distance.cdist(X,Y,'euclidean')
    sqdist = scipy_spatial.cdist(X,Y,'euclidean')
    sqdist=sqdist*sqdist
    return np.exp(sqdist*(-1/(2*length_scale*length_scale)))