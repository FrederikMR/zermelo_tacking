#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:10:55 2023

@author: fmry
"""

#%% Sources

#https://en.wikipedia.org/wiki/List_of_numerical_analysis_topics

#%% Modules

from JaxMan.initialize import *

#%% Private functions

@jit
def rectangle(x:jnp.ndarray, dt:float) -> float:
   
    return jnp.sum(x[1:]*dt)
   
@jit 
def trapezoidal(x:jnp.ndarray, dt:float) -> float:
   
    return .5*dt*(x[0]+x[-1]+2*jnp.sum(x[1:-1]))

@jit
def simpson(x:jnp.ndarray, dt:float) -> float:
   
    return (dt/3) * (x[0] + 2*jnp.sum(x[:-2:2]) \
            + 4*sum(x[1:-1:2]) + x[-1])
        