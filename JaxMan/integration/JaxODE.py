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

def euler(f_fun: Callable[[float, jnp.ndarray], jnp.ndarray], 
          y0: jnp.ndarray,
          grid:jnp.ndarray) -> jnp.ndarray:
    
    def step(yn:jnp.ndarray, time:tuple[float, float]) -> tuple[jnp.ndarray, jnp.ndarray]:
        
        tn, dt = time
        
        yn += dt*f_fun(tn,yn)
        
        return yn, yn
    
    _, y = lax.scan(step, init=y0, xs=(grid[1:], jnp.diff(grid)))
    
    return jnp.concatenate((y0.reshape(1,-1), y))

def heun(f_fun: Callable[[float, jnp.ndarray], jnp.ndarray], 
          y0: jnp.ndarray,
          grid:jnp.ndarray) -> jnp.ndarray:
    
    def step(y0:jnp.ndarray, time:tuple[float, float, float]) -> tuple[jnp.ndarray, jnp.ndarray]:
        
        t0, t1, dt = time
        
        y1 = y0 + dt*f_fun(t0,y0)
        y1 = y0 + 0.5*dt*(f_fun(t0,y0)+f_fun(t1,y1))
        
        return y1, y1
    
    _, y = lax.scan(step, init=y0, xs=(grid[:-1], grid[1:], jnp.diff(grid)))
    
    return jnp.concatenate((y0.reshape(1,-1), y))

def rk4(f_fun: Callable[[float, jnp.ndarray], jnp.ndarray], 
          y0: jnp.ndarray,
          grid:jnp.ndarray) -> jnp.ndarray:
    
    def step(yn:jnp.ndarray, time:tuple[float, float, float]) -> tuple[jnp.ndarray, jnp.ndarray]:
        
        t0, t1, dt = time
        
        dt2 = 0.5*dt
        dt6 = dt/6
        
        tn2 = t0+dt2
        k1 = f_fun(t0, yn)
        k2 = f_fun(tn2, yn+dt2*k1)
        k3 = f_fun(tn2, yn+dt2*k2)
        k4 = f_fun(t1, yn+dt2*k3)
        
        y1 = y0 + dt*f_fun(t0,y0)
        yn += dt6*(k1+2*k2+2*k3+k4)
        
        return yn, yn

    _, y = lax.scan(step, init=y0, xs=(grid[:-1], grid[1:], jnp.diff(grid)))
    
    return jnp.concatenate((y0.reshape(1,-1), y))
