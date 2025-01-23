#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
import jax.random as jrandom
from jax import jit, lax

from geometry.manifolds import EllipticFinsler, PointcarreLeft, PointcarreRight, TimeOnly

#%% Load manifolds

def load_stochastic_manifold(manifold:str="direction_only", 
                             seed:int=2712,
                             N_sim:int=10,
                             ):
    
    key = jrandom.key(seed)
    key, subkey = jrandom.split(key)
    
    eps = jrandom.normal(subkey, shape=(N_sim,))
    
    if manifold == "direction_only":
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            rho=3/2
            phi=jnp.pi/2+6*jnp.pi/10
            theta = lambda t,x,v: jnp.pi+e
            a=lambda t,x,v: 2
            b=lambda t,x,v: 2
            c1 = lambda t,x,v: -rho*jnp.cos(phi)
            c2 = lambda t,x,v: -rho*jnp.sin(phi)
            M1 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            a=lambda t,x,v: 1
            b=lambda t,x,v: 1
            c1=lambda t,x,v: 3/4
            c2=lambda t,x,v: 0
            theta=lambda t,x,v: 0
            M2 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([2.,8.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_only":
        
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = TimeOnly()
            
            rho=lambda t: -3/2
            phi=lambda t: 0
            theta = lambda t,x,v: jnp.pi/4+e
            a=lambda t,x,v: 7
            b=lambda t,x,v: a(t,x,v)/4
            c1 = lambda t,x,v: rho(t)*jnp.cos(phi(t))
            c2 = lambda t,x,v: rho(t)*jnp.sin(phi(t))
            M2 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([5*jnp.pi,0.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "poincarre":
        
        eps = jnp.clip(eps+0.5, 0.1, 0.9)
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:    
            M1 = PointcarreLeft(d=e)
            M2 = PointcarreRight(d=0.5)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
        
        k = 10.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([1.,1.], dtype=jnp.float32)
        zT = jnp.array([k,1.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")