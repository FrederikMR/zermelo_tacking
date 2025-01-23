#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 22:30:32 2024

@author: fmry
"""

#%% Modules

import jax.numpy as jnp
from jax import jacfwd
import jax.scipy as jscipy

#%% Finsler Metric

def Falpha(t,x1,v):
    
    x,y = v[0], v[1]
    
    return (1.0*(2+1/2*jnp.sin(t))**2*jnp.sin(1/4*t*jnp.pi)*y \
            +1.0*(3/2+3/8*jnp.sin(t))**2*jnp.cos(1/4*t*jnp.pi)*x \
                +((2+1/2*jnp.sin(t))**4*(3/2+3/8*jnp.sin(t))**2*y**2 \
                  +(2+1/2*jnp.sin(t))**2*(3/2+3/8*jnp.sin(t))**4*x**2-\
                      1.00*(2+1/2*jnp.sin(t))**2*(3/2+3/8*jnp.sin(t))**2*jnp.cos(1/4*t*jnp.pi)**2*y**2 \
                          +2.00*(2+1/2*jnp.sin(t))**2*(3/2+3/8*jnp.sin(t))**2*jnp.cos(1/4*t*jnp.pi) \
                              *jnp.sin(1/4*t*jnp.pi)*x*y-1.00*(2+1/2*jnp.sin(t))**2*\
                                  (3/2+3/8*jnp.sin(t))**2*jnp.sin(1/4*t*jnp.pi)**2*x**2)**(1/2)) \
        /((2+1/2*jnp.sin(t))**2*(3/2+3/8*jnp.sin(t))**2-1.00*(2+1/2*jnp.sin(t))**2\
          *jnp.sin(1/4*t*jnp.pi)**2-1.00*(3/2+3/8*jnp.sin(t))**2*jnp.cos(1/4*t*jnp.pi)**2)

#%% Defining start and end point

z0 = jnp.zeros(2)
zT = jnp.array([5*jnp.pi, 0.0])

#%% Defining discrte curve

N = 10000
zt = jnp.linspace(0,1,N).reshape(-1,1)*(zT-z0)+z0

#%% Defining continious curve

def z_cont(s):
    
    return (zT-z0)*s+z0

def dz_cont(s):
    
    return jacfwd(z_cont)(s)

#%% Computing integral

def disc_integral(F, zt):
    
    t = 0.0
    for i in range(len(zt)-1):
        t += F(t,zt[i], zt[i+1]-zt[i]) 
    
    return t

print(disc_integral(Falpha, zt))

#%% Computing continous integral

def cont_integral(F, z_fun, dz_fun, T=1.0, N=100):
    
    h = T/N
    t = 0.0
    for i in range(N):
        t += F(t,z_fun(t), dz_fun(t))*h
    
    return t

print(cont_integral(Falpha, z_cont, dz_cont, T=1.0, N=N))