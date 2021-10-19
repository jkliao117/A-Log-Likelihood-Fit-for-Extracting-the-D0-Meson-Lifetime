#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 00:22:03 2018

@author: RoroLiao
"""
"""importing data, computing true decay function and its NLL"""
###############################################################################
import math
import numpy as np
import scipy as sp
import scipy.special

# read data in a text file
def read_data(filename):
    # import the data as a numpy array
    data = np.loadtxt('lifetime.txt')
    # rearrange in the ascending order of time
    # data = data[np.argsort(data[:,0])]
    # extract the time and error data individually
    t = data[:,0]
    sigma = data[:,1]
    return t, sigma

t, sigma = read_data('lifetime.txt')

# compute the true decay function
# t,sigma,tau - floating point number
def true_func(t,sigma,tau):
    # eq(3)
    A = 1/(2*tau)
    b = ((sigma/tau)**2)/2 - t/tau
    B = np.exp(b)    
    c = (sigma/tau-t/sigma)/(2**0.5)
    # erfc from the scipy special library
    C = sp.special.erfc(c)    
    f = A * B * C
    return f

# compute the negative log likelihood function for the true decay function
# tau - numpy array
def NLL_true(tau):
    # find the dimensions of tau and t,sigma to create a NLL array M
    n = tau.shape
    N = t.shape
    M = np.zeros(n)
    # for each value of tau
    for i in range(n[0]):
        # calculate the sum of log of the true function over all (t,sigma)
        M[i]= -sum(math.log(true_func(t[j],sigma[j],tau[i])) for j in range(N[0]))
    return M
