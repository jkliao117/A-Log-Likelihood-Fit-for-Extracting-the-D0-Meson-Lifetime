#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 11:23:12 2018

@author: RoroLiao
"""
"""performing error analysis for different number of measurement"""
###############################################################################
import numpy as np
import math
from part1 import t, sigma, true_func

# modified NLL function
# tau - numpy array
# sample_number - floating point number showing the number of measurements included
def NLL_true_mod(tau,sample_number):
    # find the dimensions of tau to create a NLL array M
    n = tau.shape
    M = np.zeros(n)
    # for each value of tau
    for i in range(n[0]):
        # calculate the sum of log of the true function over all (t,sigma)
        M[i]= -sum(math.log(true_func(t[j],sigma[j],tau[i])) for j in range(int(sample_number)))
    return M

# modified parabolic minimisaion
# tau - numpy array with 3 elements representing the chosen initial values
# sample_number - floating point number showing the number of measurements included
def parabolic_minimisation_mod(tau, sample_number):
    # for any tau data set, take its min, midpoint, and max values
    np.sort(tau)
    x = np.array([tau[0],tau[1],tau[2]])
    # find the corresponding ys
    y = NLL_true_mod(tau,sample_number)
    # before tau converges to this extent
    while (x[2]-x[0])>1e-3:
        # estimate tau at the minimum NNL
        x3 = ((x[2]**2 -x[1]**2)*y[0] + (x[0]**2 - x[2]**2)*y[1] + (x[1]**2 - x[0]**2)*y[2]) \
        /(2*((x[2]-x[1])*y[0] + (x[0]-x[2])*y[1] + (x[1]-x[0])*y[2]))
        y3 = NLL_true_mod(np.array([x3]),sample_number)
        # consider 4 data points (original 3 + new 1)
        temp_x = np.append(x,x3)
        temp_y = np.append(y,y3)
        # keep the lowest 3
        # find the maximum NNL value and remove it and its tau
        max_y_index = np.argwhere(temp_y == max(temp_y))
        x = np.delete(temp_x, max_y_index)
        y = np.delete(temp_y, max_y_index)
        # rearrange in the ascending order of tau
        sort_indeces = x.argsort()
        x = x[sort_indeces]
        y = y[sort_indeces]
    # obtain the final minimum estimate of tau    
    q_min,nll_min = x3,y3[0]
    return q_min,nll_min

# sample - numpy array containing the numbers of meausrements interetsed
def parabolic_error_mod(sample):
    # input for parabolic minimisation
    tau0 = np.linspace(0.2,0.6,3)
    # find the dimensions of sample to create a tau error array 
    s = sample.shape
    tau_error = np.zeros(s)
    # for each number of meausrements
    for i in range(s[0]):
        n = sample[i]
        # find tau_min
        tau_avg = parabolic_minimisation_mod(tau0, n)[0]
        # find tau_error using parobolic estimates
        tau = (1+1e-3)*tau_avg
        nll_min = NLL_true_mod(np.array([tau_avg]),n)
        a = (NLL_true_mod(np.array([tau]),n)-nll_min)/((tau_avg*1e-3)**2)
        tau_error[i] = np.sqrt(0.5/a)
    return tau_error

def gaussian_error_mod(sample):
    # input for parabolic minimisation
    tau0 = np.linspace(0.2,0.6,3)
    # find the dimensions of sample to create a tau error array 
    s = sample.shape
    tau_error = np.zeros(s)
    # for each number of meausrements
    for i in range(s[0]):
        n = sample[i]
        # find tau_min
        tau_avg = parabolic_minimisation_mod(tau0, n)[0]
        # find tau_error using gussian estimates
        x0, x1, x2 = np.array([tau_avg]), np.array([tau_avg*(1+1e-3)]), np.array([tau_avg*(1-1e-3)])
        f0, f1, f2 = NLL_true_mod(x0,n), NLL_true_mod(x1,n), NLL_true_mod(x2,n) 
        alpha = f0/((x0-x1)*(x0-x2)) + f1/((x1-x0)*(x1-x2)) + f2/((x2-x0)*(x2-x1)) 
        tau_error[i] = np.sqrt(0.5/alpha)[0]
    return tau_error