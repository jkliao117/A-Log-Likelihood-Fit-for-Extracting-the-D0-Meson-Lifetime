#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:45:48 2018

@author: RoroLiao
"""
"""computing total decay function and its NLL"""
###############################################################################
import math
import numpy as np
from part1 import t, sigma, true_func

# compute the total decay function
# t,sigma,tau,a - floating point number 
def decay_func(t,sigma,tau,a):
    A = true_func(t, sigma, tau)
    # eq(7)
    B = 1/(sigma*((2*np.pi)**0.5))
    C = -0.5*((t/sigma)**2)
    # eq(8)
    f = a*A + (1-a)*B*np.exp(C)
    return f

# compute the negative log likelihood function for the total decay function
# tau,a - numpy array
# for observing how NLL changes with tau and a 
def NLL_total(tau,a):
    # find the dimensions of tau, a, and t,sigma to create a NLL array M
    n = tau.shape
    l = a.shape
    N = t.shape
    M = np.zeros([n[0],l[0]])
    # for each value of tau
    for i in range(n[0]):
        # for each value of a
        for j in range(l[0]):
            # calculate the sum of log of the true function over all (t, sigma)
            M[i][j] = -sum(math.log(decay_func(t[k],sigma[k],tau[i],a[j])) for k in range(N[0]))
    return M

# compute the negative log likelihood function for the total decay function
# tau,a - floating point number 
# for evaluating NLL for specific tau and a 
def NLL_func(tau,a): 
    # check the dimensions of t, sigma
    N = t.shape
    # calculate the sum of log of the true function over all (t, sigma)
    m = 0
    for k in range(N[0]):
        z = decay_func(t[k],sigma[k],tau,a)
        m += -math.log(z)
    return m