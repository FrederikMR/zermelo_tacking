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
from geometry.manifolds import ExpectedEllipticFinsler, ExpectedPointcarreLeft, ExpectedPointcarreRight

#%% Load manifolds

def load_stochastic_manifold(manifold:str="direction_only", 
                             seed:int=2712,
                             N_sim:int=10,
                             ):
    
    key = jrandom.key(seed)
    key, subkey = jrandom.split(key)
    
    if manifold == "direction_only":
        
        eps = jrandom.normal(subkey, shape=(N_sim,2,2))
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            rho=3/2
            phi=jnp.pi/2+6*jnp.pi/10
            theta = lambda t,x,v: jnp.pi+e[0][0]
            a=lambda t,x,v: 2+e[0][1]
            b=lambda t,x,v: 2
            c1 = lambda t,x,v: -rho*jnp.cos(phi)
            c2 = lambda t,x,v: -rho*jnp.sin(phi)
            M1 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            a=lambda t,x,v: 1+e[1][0]
            b=lambda t,x,v: 1
            c1=lambda t,x,v: 3/4
            c2=lambda t,x,v: 0
            theta=lambda t,x,v: 0+e[1][0]
            M2 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
            
        rho=3/2
        phi=jnp.pi/2+6*jnp.pi/10
        theta = lambda t,x,v: jnp.pi
        a=lambda t,x,v: 2
        b=lambda t,x,v: 2
        c1 = lambda t,x,v: -rho*jnp.cos(phi)
        c2 = lambda t,x,v: -rho*jnp.sin(phi)
        key, subkey = jrandom.split(key)
        Malpha_expected = ExpectedEllipticFinsler(subkey, N_sim=100, c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        a=lambda t,x,v: 1
        b=lambda t,x,v: 1
        c1=lambda t,x,v: 3/4
        c2=lambda t,x,v: 0
        theta=lambda t,x,v: 0
        key, subkey = jrandom.split(key)
        Mbeta_expected = ExpectedEllipticFinsler(subkey, N_sim=100, c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([2.,8.], dtype=jnp.float32)
        
        return t0, z0, zT, Malpha_expected, Mbeta_expected, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_only":
        
        eps = jrandom.normal(subkey, shape=(N_sim,2))
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = TimeOnly()
            
            rho=lambda t: -3/2
            phi=lambda t: 0
            theta = lambda t,x,v: jnp.pi/4+e[0]
            a=lambda t,x,v: 7+e[1]
            b=lambda t,x,v: a(t,x,v)/4
            c1 = lambda t,x,v: rho(t)*jnp.cos(phi(t))
            c2 = lambda t,x,v: rho(t)*jnp.sin(phi(t))
            M2 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
          
            Malpha.append(M1)
            Mbeta.append(M2) 
        
        Malpha_expected = TimeOnly()
        
        rho=lambda t: -3/2
        phi=lambda t: 0
        theta = lambda t,x,v: jnp.pi/4
        a=lambda t,x,v: 7
        b=lambda t,x,v: a(t,x,v)/4
        c1 = lambda t,x,v: rho(t)*jnp.cos(phi(t))
        c2 = lambda t,x,v: rho(t)*jnp.sin(phi(t))
        key, subkey = jrandom.split(key)
        Mbeta_expected = ExpectedEllipticFinsler(subkey, N_sim=100, c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([5*jnp.pi,0.], dtype=jnp.float32)
        
        return t0, z0, zT, Malpha_expected, Mbeta_expected, tack_metrics, reverse_tack_metrics
    
    elif manifold == "poincarre":
        
        eps = jrandom.uniform(subkey, shape=(N_sim,2), minval=0.4, maxval=0.6)
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = PointcarreLeft(d=e[0])
            M2 = PointcarreRight(d=e[1])
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
            
         
        key, subkey = jrandom.split(key)
        Malpha_expected = ExpectedPointcarreLeft(subkey, N_sim=100)
        key, subkey = jrandom.split(key)
        Mbeta_expected = ExpectedPointcarreRight(subkey, N_sim=100)
        
        k = 10.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([1.,1.], dtype=jnp.float32)
        zT = jnp.array([k,1.], dtype=jnp.float32)
        
        return t0, z0, zT, Malpha_expected, Mbeta_expected, tack_metrics, reverse_tack_metrics
    
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")