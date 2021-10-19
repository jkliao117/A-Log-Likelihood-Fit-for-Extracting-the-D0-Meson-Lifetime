#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 00:25:42 2018

@author: RoroLiao
"""
"""computing parabolic minimisation, finding errors of the result"""
###############################################################################
import numpy as np
from part1 import NLL_true

# perform parabolic minimisation on NLL(tau)
# q - numpy array with 3 elements representing the chosen initial values
# f - a one variable function
def parabolic_minimisation(q,f):
    # for any tau data set, take its min, midpoint, and max values
    np.sort(q)
    x = np.array([q[0],q[1],q[2]])
    # find the corresponding ys
    y = f(x)
    # before tau converges to this extent
    while (x[2]-x[0])>1e-5:
        # estimate tau at the minimum NNL
        x3 = ((x[2]**2 -x[1]**2)*y[0] + (x[0]**2 - x[2]**2)*y[1] + (x[1]**2 - x[0]**2)*y[2]) \
        /(2*((x[2]-x[1])*y[0] + (x[0]-x[2])*y[1] + (x[1]-x[0])*y[2]))
        y3 =  f(np.array([x3]))
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
        if len(x)<3:
            break     
    # obtain the final minimum estimate of tau    
    q_min,f_min = x3,y3[0]
    return q_min,f_min

# evaluate the error of the result of minimisation 1
def error(tau_avg,nll_min): 
    # the values tau+ where the NLL is changed by 0.5
    nll_std = nll_min+0.5
    # choose tau_min as the starting point
    # estimate the range where tau+ lies in
    x = np.linspace(tau_avg,tau_avg+0.1,11)
    nll_plus = nll_min
    # i is the magnitude of the range examined
    i = -1
    # before the NLL at the estimated error converges to the NLL_std to this extent
    while np.abs(nll_plus-nll_std)>1e-5:
        # calculate the NLL values for this range
        y = NLL_true(x)
        # calculate the differences between these values and the NLL_std 
        d = nll_std-y
        # find the closest value that is less than the NLL_std and its index
        j = np.argwhere(d == min(a for a in d if a>0))[0][0]
        # update the estimate of tau+
        tau_plus = x[j]
        nll_plus = y[j]
        # look at a smaller range with values larger than the new tau+ estimate in the next round 
        i -= 1
        x = np.linspace(tau_plus,tau_plus+10**(i),11) 
    # calculate the std/error
    error_plus = tau_plus-tau_avg     
    # repeat the similar process for tau-
    x = np.linspace(tau_avg-0.1,tau_avg,11)
    nll_minus = nll_min
    i = -1
    while np.abs(nll_minus-nll_std)>1e-5:
        y = NLL_true(x)
        d = nll_std - y
        j = np.argwhere(d == max(a for a in d if a<0))[0][0]
        tau_minus = x[j]
        nll_minus = y[j]
        i -= 1
        x = np.linspace(tau_minus,tau_minus+10**(i),11)
    error_minus = tau_avg-tau_minus
    return error_plus, error_minus

# evaluate the error 2
def parabolic_error(tau_avg,nll_min):
    # choose a point close to the turning point to fit the parabolic curve
    tau = np.array([tau_avg*(1+1e-3)])
    # NLL = a(tau - tau_min)^2 + nll_min
    a = (NLL_true(tau)-nll_min)/((tau_avg*1e-3)**2)
    # 0.5 = a(tau_error)^2
    tau_error = np.sqrt(0.5/a)[0]
    return tau_error

# evaluate the error 3
def gaussian_error(tau_avg,nll_min):
    # find three points near the minmimum
    x0, x1, x2 = np.array([tau_avg]), np.array([tau_avg*(1+1e-5)]), np.array([tau_avg*(1-1e-5)])
    f0, f1, f2 = NLL_true(x0), NLL_true(x1), NLL_true(x2) 
    # y = A(x-xmin)^2+B
    # calculate the parameter for the 2nd order polynomial
    alpha = f0/((x0-x1)*(x0-x2)) + f1/((x1-x0)*(x1-x2)) + f2/((x2-x0)*(x2-x1)) 
    # compute sigma
    tau_error = np.sqrt(0.5/alpha)[0]
    return tau_error