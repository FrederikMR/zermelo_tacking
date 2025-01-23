#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
from jax import jit, lax

from geometry.manifolds import EllipticFinsler, PointcarreLeft, PointcarreRight, TimeOnly

#%% Load manifolds

def load_manifold(manifold:str="direction_only", 
                  ):
    
    if manifold == "direction_only":
        
        rho=3/2
        phi=jnp.pi/2+6*jnp.pi/10
        theta = lambda t,x,v: jnp.pi
        a=lambda t,x,v: 2
        b=lambda t,x,v: 2
        c1 = lambda t,x,v: -rho*jnp.cos(phi)
        c2 = lambda t,x,v: -rho*jnp.sin(phi)
        Malpha = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        a=lambda t,x,v: 1
        b=lambda t,x,v: 1
        c1=lambda t,x,v: 3/4
        c2=lambda t,x,v: 0
        theta=lambda t,x,v: 0
        Mbeta = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        tack_metrics = [Malpha,Mbeta,Malpha,Mbeta,Malpha]
        reverse_tack_metrics = [Mbeta, Malpha, Mbeta, Malpha, Mbeta]
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([2.,8.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_only":

        Malpha = TimeOnly()
        
        rho=lambda t: -3/2
        phi=lambda t: 0
        theta = lambda t,x,v: jnp.pi/4
        a=lambda t,x,v: 7
        b=lambda t,x,v: a(t,x,v)/4
        c1 = lambda t,x,v: rho(t)*jnp.cos(phi(t))
        c2 = lambda t,x,v: rho(t)*jnp.sin(phi(t))
        Mbeta = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        tack_metrics = [Malpha,Mbeta,Malpha,Mbeta,Malpha]
        reverse_tack_metrics = [Mbeta, Malpha, Mbeta, Malpha, Mbeta]
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([5*jnp.pi,0.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "poincarre":
        
        Malpha = PointcarreLeft(d=0.5)
        Mbeta = PointcarreRight(d=0.5)
        
        tack_metrics = [Malpha,Mbeta,Malpha,Mbeta,Malpha]
        reverse_tack_metrics = [Mbeta, Malpha, Mbeta, Malpha, Mbeta]
        
        k = 10.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([1.,1.], dtype=jnp.float32)
        zT = jnp.array([k,1.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")