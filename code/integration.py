#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:28:09 2018

@author: RoroLiao
"""
"""performing integration using Trapezoidal Rule"""
###############################################################################
import numpy as np
from part1 import true_func

# compute the integral of true dunction wrt t using the Extended Trapezoidal Rule
# for a given (t,sigma) 
# t - numpy array with 2 elements defining the lower and upper bounds
# sigma,tau - flating point number
def integral_ft(t,sigma,tau):
    # find t_min, t_max and f(t_min), f(t_max)
    a,b = t[0],t[1]
    f_a,f_b = true_func(t, sigma, tau)
    # the area of a trapezium
    A = (f_a+f_b)/2
    h1 = b-a
    I1 = A*h1
    # n is the number of intervals, c is the convergence limit
    n,c = 1,1
    # before ith intergal value converges to this extent
    while c > 1e-5:
        # update the number of interval
        n += 2
        # x is sample point
        x = np.linspace(a,b,(n+1))[1:-2]
        # the Extended Trapezoidal Rule formula
        S = A + sum(true_func(x, sigma, tau))
        h2 = h1/n
        I2 = S*h2
        # check the convergence
        c = abs(I2/I1-1)
        # update the estimated integral
        I1 = I2
    return I1

# compute the integral of true dunction wrt t using the Extended Trapezoidal Rule
# for many different (t,sigma) 
# tau - numpy array with 2 elements defining the lower and upper bounds
# t, sigma - numpy array
def integral_f_total(t,sigma,tau):
    # check the dimensions of sigma and tau to create an intergral array I
    n = sigma.shape
    m = tau.shape
    I = np.zeros([n[0],m[0]])
    # evaluate the integral for each (t,sigma)
    for i in range(n[0]):
        for j in range(m[0]):
            I[i][j] = integral_ft(t, sigma[i], tau[j])
    return I

# integral for any one variable function
# for testing
def integral(x,f):
    # find x_min, x_max and f(x_min), f(x_max)
    a,b = x[0],x[1]
    f_a,f_b = f(x)
    # the area of a trapezium
    A = (f_a+f_b)/2
    h1 = b-a
    I1 = A*h1
    # n is the number of intervals, c is the convergence limit
    n,c = 1,1
    # before ithe ntergal value converges to this extent
    while c > 1e-5:
        # update the number of interval
        n += 2
        # x is smaple points
        y = np.linspace(a,b,(n+1))[1:-2]
        # the Extended Trapezoidal Rule formula
        S = A + sum(f(y))
        h2 = h1/n
        I2 = S*h2
        # check the convergence
        c = abs(I2/I1-1)
        # update the estimated integral
        I1 = I2
    return I1
