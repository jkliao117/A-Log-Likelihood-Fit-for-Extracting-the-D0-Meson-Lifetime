#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 12:36:27 2018

@author: RoroLiao
"""
"""computing Quasi-Newton minimisation, finding errors of the result"""
###############################################################################
import numpy as np
from part3 import NLL_total, NLL_func

# compute the gradient of any function at any point (x,y) from first principles
def gradient_f(x,y,f):
    # find f(x,y)
    f0 = f(x,y)
    # generate points x1 = x + delta_x and y1 = y + delta_y 
    x1 = (1+1e-5)*x
    y1 = (1+1e-5)*y
    # evaluate the partial differentiation rwt tau and a from first principles
    df_x = (f(x1,y) - f0)/ (x1-x)
    df_y = (f(x,y1) - f0)/ (y1-y)
    # get gradient
    return np.array([df_x,df_y]) 

# compute new G using Davidon– Fletcher–Power algorithm    
def new_G(delta,gamma,G):
    # eq(7.20)
    delta_t = np.transpose(delta)
    gamma_t = np.transpose(gamma)
    A = np.cross(delta, delta_t)/np.dot(gamma_t, delta)
    B = np.dot(G, np.dot(np.cross(delta, delta_t), G))
    C = np.dot(gamma_t,np.dot(G,gamma))
    G1 = G + A - B/C
    return G1

# perform Quasi-Newton minimisation
# x,y - floating point number representing the chosen initial value 
def quasi_newton_minimisation(x,y,f):
    # alpha - the parameter determining the madnitude of the step
    alpha = 1e-5
    # evaluate the gradient at v0 = (x,y)
    v0 = np.array([x,y])
    grad_f0 = gradient_f(x,y,f)
    # set initial G0 and delta
    G0 = np.identity(2)
    delta = np.array([1,1])    
    # before tau,a converge to these extents
    while (delta[0]>1e-6) or (abs(delta[1])>1e-6):
        # find the new estimates of minimum f = (x,y) using eq (7.18)
        delta = -alpha * np.dot(G0,grad_f0)  
        v1 = v0 + delta
        # calculate the new gradient and G
        grad_f1 = gradient_f(v1[0],v1[1],f)
        gamma = grad_f1 - grad_f0
        G1 = new_G(delta,gamma, G0)
        # update the v0 ,gradient, and G
        v0 = v1
        grad_f0 = grad_f1
        G0 = G1
    # obtain the final minimum estimate of x and y
    x_min, y_min = v0[0],v0[1]
    f_min = f(x_min, y_min)
    return x_min, y_min, f_min

# evaluate the error of tau_min
def error_tau(tau_avg,a_avg,nll_min):  
    # the values tau+ where the NLL is changed by 0.5
    nll_s = nll_min+0.5
    # choose tau_min as the starting point
    # estimate the range where tau+ lies in
    x = np.linspace(tau_avg,tau_avg+0.1,11)
    a = np.array([a_avg])
    # i is the magnitude of the range examined
    i = -1
    nll_plus = nll_min
    # before the NLL at the estimated error converges to the NLL_std to this extent
    while np.abs(nll_plus-nll_s)>1e-5:
        # calculate the NLL values for this range
        y = NLL_total(x,a)
        # calculate the differences between these values and the NLL_std 
        d = nll_s-y
        # find the closest value that is less than the NLL_std and its index
        j = np.argwhere(d == min(i for i in d if i>0))[0][0]
        # update the estimate of tau+
        tau_plus = x[j]
        nll_plus = y[j]
        # look at a smaller range with values larger than the new tau+ estimate in the next round 
        i -= 1
        x = np.linspace(tau_plus,tau_plus+10**(i),11) 
    # calculate the std/error
    error_plus = tau_plus-tau_avg
    # repeat the similar process for tau-      
    x = np.linspace(tau_avg -0.1,tau_avg,11)
    nll_minus = nll_min
    i = -1
    while np.abs(nll_minus-nll_s)>1e-5:
        y = NLL_total(x,a)
        d = nll_s-y
        j = np.argwhere(d == max(i for i in d if i<0))[0][0]
        tau_minus = x[j]
        nll_minus = y[j]
        i -= 1
        x = np.linspace(tau_minus,tau_minus+10**(i),11)
    error_minus = tau_avg-tau_minus
    
    return error_plus, error_minus

# evaluate the error of a_min
def error_a(tau_avg,a_avg,nll_min):  
    # the values a+ where the NLL is changed by 0.5
    nll_s = np.transpose(nll_min)+0.5
    # choose a_min as the starting point
    # estimate the range where a+ lies in
    x = np.linspace(a_avg,1,11)
    tau = np.array([tau_avg])
    # i is the magnitude of the range examined
    i = -1
    nll_plus = np.transpose(nll_min)
    # before the NLL at the estimated error converges to the NLL_std to this extent
    while np.abs(nll_plus-nll_s)>1e-5:
        # calculate the NLL values for this range
        y = np.transpose(NLL_total(tau,x))
        # calculate the differences between these values and the NLL_std 
        d = nll_s-y
        # find the closest value that is less than the NLL_std and its index
        j = np.argwhere(d == min(i for i in d if i>0))[0][0]
        # update the estimate of a+
        a_plus = x[j]
        nll_plus = y[j]
        # look at a smaller range with values larger than the new tau+ estimate in the next round 
        i -= 1
        x = np.linspace(a_plus,a_plus+10**(i), 11) 
    # calculate the std/error
    error_plus = a_plus-a_avg 
    # repeat the similar process for a-     
    x = np.linspace(a_avg-0.1,a_avg,11)
    nll_minus = np.transpose(nll_min)
    i = -1
    while np.abs(nll_minus - nll_s)>1e-5:
        y = np.transpose(NLL_total(tau,x))
        d = nll_s-y
        j = np.argwhere(d == max(i for i in d if i<0))[0][0]
        a_minus = x[j]
        nll_minus = y[j]
        i -= 1
        x = np.linspace(a_minus,a_minus+10**(i),11)
    error_minus = a_avg-a_minus

    return error_plus, error_minus

# evaluate the error 2
def parabolic_error2d(tau_avg, a_avg, nll_min):
    # choose a point close to the turning point to fit the parabolic curve
    tau = np.array([tau_avg*(1+1e-3)])
    # NLL = a(tau - tau_min)^2 + nll_min
    p = (NLL_func(tau,a_avg)-nll_min)/((tau_avg*1e-3)**2)
    # 0.5 = a(tau_error)^2
    tau_error = np.sqrt(0.5/p)
    # repeat the similar process for a
    a = np.array([a_avg*(1+1e-3)])
    q = (NLL_func(tau_avg,a)-nll_min)/((a_avg*1e-3)**2)
    a_error = np.sqrt(0.5/q)
    return tau_error, a_error

# evaluate the error 3
def gaussian_error2d(tau_avg, a_avg):
    # find three points near the minmimum
    x0, x1, x2 = np.array([tau_avg]), np.array([tau_avg*(1+1e-5)]), np.array([tau_avg*(1-1e-5)])
    f0, f1, f2 = NLL_func(x0,a_avg), NLL_func(x1,a_avg), NLL_func(x2,a_avg) 
    # y = A(x-xmin)^2+B
    # calculate the parameter for the 2nd order polynomial
    alpha = f0/((x0-x1)*(x0-x2)) + f1/((x1-x0)*(x1-x2)) + f2/((x2-x0)*(x2-x1)) 
    # compute sigma
    tau_error = np.sqrt(0.5/alpha)[0]
    # repeat the similar process for a
    x0, x1, x2 = np.array([a_avg]), np.array([a_avg*(1+1e-5)]), np.array([a_avg*(1-1e-5)])
    f0, f1, f2 = NLL_func(tau_avg,x0), NLL_func(tau_avg,x1), NLL_func(tau_avg,x2) 
    alpha = f0/((x0-x1)*(x0-x2)) + f1/((x1-x0)*(x1-x2)) + f2/((x2-x0)*(x2-x1)) 
    a_error = np.sqrt(0.5/alpha)[0]
    return tau_error, a_error