# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 18:48:57 2016

@author: frey
"""

"""
v, tau are the parameters we learned before
X is our training dataset
y is our label vector with labels +1/-1 (ie. 6 or 8, 3 or 5)
k can be user chosen, but we assume that it is the function from kernel(.py) that we used earlier to compute K

x is our "unknown" input we want to classify/get the probabilities for.
x is not a single point but of the shape of X, but with only a single entry, otherwise kernel.compute seems to complain
about the dimensions.
"""
def classify(v, tau, X, y, k, x):
    pass