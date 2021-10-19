#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 00:29:23 2018

@author: RoroLiao
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from part1 import t, sigma, true_func, NLL_true
from part2 import parabolic_minimisation, error, parabolic_error, gaussian_error
from part3 import NLL_total, NLL_func
from part4 import quasi_newton_minimisation, error_tau, error_a, parabolic_error2d, gaussian_error2d
from integration import integral_f_total
from error_analysis import parabolic_error_mod, gaussian_error_mod



"""decay time histogram"""
"""error of decay time histogram"""
fig1 = plt.figure()
plt.subplots_adjust(wspace=0.4)
ax = plt.subplot(1,2,1)
weights = np.ones_like(t)/float(len(t))
ax.hist(t, bins=100, color = 'DarkCyan', stacked=True, normed=True)
ax.plot(np.sort(t), true_func(np.sort(t),np.mean(sigma),np.mean(t)), color = 'Gold')
ax.set_title('measurements of decay time')
ax.set_xlabel('decay time (ps)')
ax.set_ylabel('probability density')
ax.set_xlim([-1.5,3])
ax.legend(['true decay function'])

ax = plt.subplot(1,2,2)
weights = np.ones_like(sigma)/float(len(sigma))
ax.hist(sigma, bins=100, color = 'CornFlowerBlue', stacked=True, normed=True)
ax.set_title('measurement errors')
ax.set_xlabel('error (ps)')
ax.set_ylabel('probability density')



"""decay function vs t, sigma, tau graphs"""
fig2 = plt.figure()
ax = plt.subplot(1,2,1)
t_0 = np.linspace(min(t),max(t),500)
sigma_0 = np.mean(sigma)
tau_0 = np.mean(t)
f_t1= true_func(t_0, sigma_0, tau_0)
f_t2= true_func(t_0, sigma_0*0.5, tau_0)
f_t3= true_func(t_0, sigma_0*2, tau_0)
ax.plot(t_0,f_t1,'Blue')
ax.plot(t_0,f_t2,'LightSeaGreen' )
ax.plot(t_0,f_t3,'Navy')
ax.legend(['average','half average', 'double average'])
ax.set_xlim([-2,4])
ax.set_xlabel('decay time (ps)')
ax.set_ylabel('probability density')
ax.set_title('true decay function vs decay time for different errors')

ax = plt.subplot(1,2,2)
t_0 = np.linspace(min(t),max(t),500)
sigma_0 = np.mean(sigma)
tau_0 = np.mean(t)
f_t4= true_func(t_0, sigma_0, tau_0)
f_t5= true_func(t_0, sigma_0, tau_0*0.5)
f_t6= true_func(t_0, sigma_0, tau_0*2)
ax.plot(t_0,f_t4,'Red')
ax.plot(t_0,f_t5,'Salmon' )
ax.plot(t_0,f_t6,'Maroon')
ax.legend(['average','half average', 'double average'])
ax.set_xlim([-2,4])
ax.set_xlabel('decay time (ps)')
ax.set_ylabel('probability density')
ax.set_title('true decay function vs decay time for different lifetimes')



"""decay function integral over t vs sigma, tau"""
t_2 = np.linspace(min(t)*1.5,max(t)*1.5,2)
sigma_2 = np.linspace(0.05,1,50)
tau_2 = np.linspace(0.05,1,50)
sigma_0 = np.array([np.mean(sigma)])
tau_0 = np.array([np.mean(t)])
I = integral_f_total(t_2,sigma_2,tau_2)
I_sigma = integral_f_total(t_2,sigma_2,tau_0)
I_tau = integral_f_total(t_2,sigma_0,tau_2)

fig3 = plt.figure()
plt.subplots_adjust(hspace=0.7)
ax = plt.subplot(2,1,1)
ax.plot(sigma_2, np.transpose(I_sigma)[0], 'DarkGoldenRod')
ax.set_ylim([0.9,1.1])
ax.set_title('integral of true decay function wrt t vs error')
ax.set_xlabel('error (ps)')
ax.set_ylabel('integral')

ax = plt.subplot(2,1,2)
ax.plot(tau_2, I_tau[0], 'BlueViolet')
ax.set_ylim([0.9,1.1])
ax.set_title('integral of true decay function wrt t vs lifetime')
ax.set_xlabel('lifetime (ps)')
ax.set_ylabel('integral')


 
"""NLL vs tau graph"""
tau = np.linspace(0.1,1,100)
nll = NLL_true(tau)

fig4 = plt.figure()
plt.plot(tau, nll/1e3, 'Navy')
plt.title('negative log-likelihood function (true decay)')
plt.xlabel('lifetime (ps)')
plt.ylabel('NNL (10^3)')
plt.legend(['NLL'])



"""parabolic minimisation test"""
q = np.linspace(-1,1,3)
def cosh(q):
    return np.cosh(q)
q_min, f_min = parabolic_minimisation(q,cosh)
print ('q_min=', q_min)
print ('f_min=', f_min)



"""minimum and error of tau"""
tau = np.linspace(0.35,0.45,3)
# tau_min = 0.40454, nll_min = 6220.44689
tau_avg, nll_min = parabolic_minimisation(tau,NLL_true)  
error_plus,error_minus = error(tau_avg, nll_min)    
error_parabolic = parabolic_error(tau_avg,nll_min)  
error_gaussian = gaussian_error(tau_avg,nll_min)
print ("for the true decay function only")
print ("tau_avg =", tau_avg)
print ("error_plus =", error_plus)
print ("error_minus =", error_minus)
print ("error_parabolic = ", error_parabolic)
print ("error_gaussian = ", error_gaussian)

a = np.array([tau_avg + error_plus, tau_avg - error_minus])
b = NLL_true(a)
c = np.array([tau_avg + error_parabolic, tau_avg - error_parabolic])
d = NLL_true(c)
e = np.array([tau_avg + error_gaussian, tau_avg - error_gaussian])
f = NLL_true(e)
tau_0 = np.linspace(0.395,0.415,100)
nll_0 = NLL_true(tau_0)

fig5 = plt.figure()
plt.plot(tau_0, nll_0/1e3, 'Chocolate')
plt.plot(tau_0, (np.zeros(tau_0.shape)+nll_min+0.5)/1e3, 'LightSeaGreen')
plt.plot(tau_avg, nll_min/1e3, 'yo')
plt.plot(a, b/1e3, 'ro')
plt.plot(c, d/1e3, 'go')
plt.plot(e, f/1e3, 'bo')
plt.title('negative log likelihood function (true decay)')
plt.ylim(6.2200, 6.2215)
plt.xlabel('lifetime (ps)')
plt.ylabel('NNL (10^3)')
plt.legend(['NLL','minimum NLL +0.5', 'minimum lifetime', 'standard deviation 1', 'standard deviation 2', 'standard deviation 3'])



"""error vs smaple number"""
sample = np.arange(1e2,len(t),100)
tau_error_all_p = parabolic_error_mod(sample)
tau_error_all_g = gaussian_error_mod(sample)

fig6 = plt.figure()
plt.plot(sample, tau_error_all_p/1e-3, 'MediumTurquoise')
#plt.plot(sample, tau_error_all_g/1e-3)
M = np.mean(tau_error_all_p*(sample**0.5))
# sigma = M/((sample number)**0.5)
plt.plot(sample, M/(sample**0.5)/1e-3, 'tomato')
plt.title("standard deviation vs number of measurements")
plt.xlabel("number of measurements")
plt.ylabel("standard deviation (fs)")
plt.legend(['standard deviation','emprical fit'])



"""NLL vs tau,a graph"""
tau = np.linspace(0.1,1,20)
a = np.linspace(0.1,1,20)
nll = NLL_total(tau,a)/1e4

fig7 = plt.figure()
ax = plt.axes(projection ='3d')
tauv, av = np.meshgrid(tau, a)
ax.plot_surface(tauv, av, np.transpose(nll), cmap=plt.cm.jet)
ax.set_xlabel('tau (ps)')
ax.set_ylabel('a')
ax.set_zlabel('NLL (10^4)')



"""quasi-Newton minimisation test"""
x = 2
y = 2
def f(x,y):
    return (x-1)**2+(y-3)**2
x_min, y_min, f_min = quasi_newton_minimisation(x,y,f) 
print ("x_min = ", x_min)
print ("y_min = ", y_min)
print ("f_min = ", f_min)



"""minimum and error of tau and a"""
tau,a = 0.4,1
# tau_avg = 0.4096830, a_avg = 0.9836775, nll_min = 6218.3944179
tau_avg, a_avg, nll_min = quasi_newton_minimisation(tau,a,NLL_func)
tau_plus, tau_minus = error_tau(tau_avg,a_avg,nll_min)
a_plus, a_minus = error_a(tau_avg,a_avg,nll_min)
tau_error_p, a_error_p = parabolic_error2d(tau_avg, a_avg, nll_min)
tau_error_g, a_error_g = gaussian_error2d(tau_avg, a_avg)
print ("for the total decay function")
print ("tau_avg =", tau_avg)
print ("tau_error =", (tau_plus, tau_minus))
print ("tau_error_p =", tau_error_p)
print ("tau_error_g =", tau_error_g)
print ("a_avg =", a_avg)
print ("a_error =", (a_plus, a_minus))
print ("a_error_p =", a_error_p)
print ("a_error_g =", a_error_g)
