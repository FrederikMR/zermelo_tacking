#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp

from geometry.manifolds import EllipticFinsler, PointcarreLeft, PointcarreRight, TimeOnly
from geometry.manifolds import PointcarreNorthLeft, PointcarreNorthRight

#%% Load manifolds

def load_manifold(manifold:str="direction_only", 
                  alpha:float=1.0,
                  ):
    
    if manifold == "direction_only":
        c1_m1=lambda t,x,v: -1.5*jnp.cos(jnp.pi/2+6*jnp.pi/10)
        c2_m1=lambda t,x,v: -1.5*jnp.sin(jnp.pi/2+6*jnp.pi/10)
        a_m1=lambda t,x,v: 2
        b_m1=lambda t,x,v: 2
        theta_m1=lambda t,x,v: jnp.pi
        
        c1_m2=lambda t,x,v: 3/4
        c2_m2=lambda t,x,v: 0
        a_m2=lambda t,x,v: 1
        b_m2=lambda t,x,v: 1
        theta_m2=lambda t,x,v: 0
        
        Malpha = EllipticFinsler(c1=c1_m1,
                                 c2=c2_m1, 
                                 a=a_m1,
                                 b=b_m1,
                                 theta=theta_m1,
                                 )
        Mbeta = EllipticFinsler(c1=c1_m2,
                                c2=c2_m2, 
                                a=a_m2,
                                b=b_m2,
                                theta=theta_m2,
                                )
        
        tack_metrics = [Malpha,Mbeta,Malpha,Mbeta,Malpha]
        reverse_tack_metrics = [Mbeta, Malpha, Mbeta, Malpha, Mbeta]
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([2.,8.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_position":
        c1_m1=lambda t,x,v: 0.5 #0.5*jnp.arctan(x[1])
        c2_m1=lambda t,x,v: 0.5 #0.5*jnp.arctan(x[1])
        a_m1=lambda t,x,v: 1.+t+(x[0]**2)+(x[1]**2) #jnp.arctan(x[1])*(2.+jnp.sin(t))
        b_m1=lambda t,x,v: 1.+t+(x[0]**2)+(x[1]**2) #jnp.arctan(x[1])*(2.+jnp.sin(t))
        theta_m1=lambda t,x,v: 0.0 #t/3
        
        c1_m2=lambda t,x,v: 0.5 #0.5*jnp.arctan(x[1])
        c2_m2=lambda t,x,v: -0.5 #-0.5*jnp.arctan(x[1])
        a_m2=lambda t,x,v: 1.+t+(x[0]**2)+(x[1]**2) #jnp.arctan(x[1])*(2.+jnp.sin(t))
        b_m2=lambda t,x,v: 1.+t+(x[0]**2)+(x[1]**2) #jnp.arctan(x[1])*(2.+jnp.sin(t))
        theta_m2=lambda t,x,v: 0.0 #t/3
        
        Malpha = EllipticFinsler(c1=c1_m1,
                                 c2=c2_m1, 
                                 a=a_m1,
                                 b=b_m1,
                                 theta=theta_m1,
                                 )
        Mbeta = EllipticFinsler(c1=c1_m2,
                                c2=c2_m2, 
                                a=a_m2,
                                b=b_m2,
                                theta=theta_m2,
                                )
        
        tack_metrics = [Malpha,Mbeta]
        reverse_tack_metrics = [Mbeta, Malpha]
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([10.,0.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_position2":
        c1_m1=lambda t,x,v: 0.5 #0.5*jnp.arctan(x[1])
        c2_m1=lambda t,x,v: 0.5 #0.5*jnp.arctan(x[1])
        a_m1=lambda t,x,v: 1.+t+(x[0]**2)+(x[1]**2) #jnp.arctan(x[1])*(2.+jnp.sin(t))
        b_m1=lambda t,x,v: 1.+t+(x[0]**2)+(x[1]**2) #jnp.arctan(x[1])*(2.+jnp.sin(t))
        theta_m1=lambda t,x,v: 0.0 #t/3
        
        c1_m2=lambda t,x,v: 0.5 #0.5*jnp.arctan(x[1])
        c2_m2=lambda t,x,v: -0.5 #-0.5*jnp.arctan(x[1])
        a_m2=lambda t,x,v: 1.+t+(x[0]**2)+(x[1]**2) #jnp.arctan(x[1])*(2.+jnp.sin(t))
        b_m2=lambda t,x,v: 1.+t+(x[0]**2)+(x[1]**2) #jnp.arctan(x[1])*(2.+jnp.sin(t))
        theta_m2=lambda t,x,v: 0.0 #t/3
        
        Malpha = EllipticFinsler(c1=c1_m1,
                                 c2=c2_m1, 
                                 a=a_m1,
                                 b=b_m1,
                                 theta=theta_m1,
                                 )
        Mbeta = EllipticFinsler(c1=c1_m2,
                                c2=c2_m2, 
                                a=a_m2,
                                b=b_m2,
                                theta=theta_m2,
                                )
        
        tack_metrics = [Malpha,Mbeta]
        reverse_tack_metrics = [Mbeta, Malpha]
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([10.,2.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_only":
        
        c1_m2=lambda t,x,v: -1.5*jnp.cos(0.)
        c2_m2=lambda t,x,v: -1.5*jnp.sin(0.) 
        a_m2=lambda t,x,v: 7
        b_m2=lambda t,x,v: 7./4
        theta_m2=lambda t,x,v: jnp.pi/4

        Malpha = TimeOnly()
        Mbeta = EllipticFinsler(c1=c1_m2,
                                c2=c2_m2, 
                                a=a_m2,
                                b=b_m2,
                                theta=theta_m2,
                                )
        
        tack_metrics = [Malpha,Mbeta]
        reverse_tack_metrics = [Mbeta, Malpha]
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([5*jnp.pi,0.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "poincarre":
        
        Malpha = PointcarreLeft(d=0.5, alpha=alpha)
        Mbeta = PointcarreRight(d=0.5, alpha=alpha)
        
        tack_metrics = [Malpha,Mbeta,Malpha,Mbeta,Malpha]
        reverse_tack_metrics = [Mbeta, Malpha, Mbeta, Malpha, Mbeta]
        
        k = 20.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,1.], dtype=jnp.float32)
        zT = jnp.array([k,1.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "poincarre_north":
        
        Malpha = PointcarreNorthLeft()
        Mbeta = PointcarreNorthRight()
        
        tack_metrics = [Malpha,Mbeta,Malpha,Mbeta,Malpha]
        reverse_tack_metrics = [Mbeta, Malpha, Mbeta, Malpha, Mbeta]
        
        k = 40.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,1.], dtype=jnp.float32)
        zT = jnp.array([k,5.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    elif manifold == "poincarre_north_a":
        
        Malpha = PointcarreNorthLeft()
        Mbeta = PointcarreNorthRight()
        
        tack_metrics = [Malpha,Mbeta,Malpha,Mbeta,Malpha]
        reverse_tack_metrics = [Mbeta, Malpha, Mbeta, Malpha, Mbeta]
        
        a = 5
        k = 20.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([a,5.], dtype=jnp.float32)
        zT = jnp.array([k-a,5.], dtype=jnp.float32)
        
        return t0, z0, zT, tack_metrics, reverse_tack_metrics
    
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
