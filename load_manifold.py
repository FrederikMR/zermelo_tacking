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

import pandas as pd

from geometry.manifolds import EllipticFinsler, PointcarreLeft, PointcarreRight, TimeOnly
from geometry.manifolds import ExpectedEllipticFinsler, ExpectedPointcarreLeft, ExpectedPointcarreRight

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
        b=lambda t,x,v: 7./4
        c1 = lambda t,x,v: rho(t)*jnp.cos(phi(t))
        c2 = lambda t,x,v: rho(t)*jnp.sin(phi(t))
        Mbeta = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        tack_metrics = [Malpha,Mbeta]
        reverse_tack_metrics = [Mbeta, Malpha]
        
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
        
#%% Load Manifolds

def load_stochastic_manifold(manifold:str="direction_only", 
                             seed:int=2712,
                             N_sim:int=10,
                             ):
    
    key = jrandom.key(seed)
    key, subkey = jrandom.split(key)
    
    if manifold == "direction_only":
        
        
        eps = jrandom.uniform(subkey, shape=(N_sim,2), minval=-0.5, maxval=0.5)
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            rho=3/2
            phi=jnp.pi/2+6*jnp.pi/10
            theta = lambda t,x,v: jnp.pi
            a=lambda t,x,v: 2
            b=lambda t,x,v: 2
            c1 = lambda t,x,v: -rho*jnp.cos(phi+e[0])
            c2 = lambda t,x,v: -rho*jnp.sin(phi+e[0])
            M1 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            a=lambda t,x,v: 1
            b=lambda t,x,v: 1
            c1=lambda t,x,v: 3/4*jnp.cos(e[1])
            c2=lambda t,x,v: 0*jnp.sin(e[1])
            theta=lambda t,x,v: 0
            M2 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
            
        key, subkey = jrandom.split(key)
        eps = jrandom.uniform(subkey, shape=(100,2), minval=-0.5, maxval=0.5)
            
        rho=3/2
        phi=jnp.pi/2+6*jnp.pi/10
        theta = lambda t,x,v,eps: jnp.pi
        a=lambda t,x,v,eps: 2
        b=lambda t,x,v,eps: 2
        c1 = lambda t,x,v,eps: -rho*jnp.cos(phi+eps)
        c2 = lambda t,x,v,eps: -rho*jnp.sin(phi+eps)
        Malpha_expected = ExpectedEllipticFinsler(subkey, eps[:,0], c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        a=lambda t,x,v,eps: 1
        b=lambda t,x,v,eps: 1
        c1=lambda t,x,v,eps: 3/4*jnp.cos(eps)
        c2=lambda t,x,v,eps: 0*jnp.sin(eps)
        theta=lambda t,x,v,eps: 0
        Mbeta_expected = ExpectedEllipticFinsler(subkey, eps[:,1], c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([2.,8.], dtype=jnp.float32)
        
        return t0, z0, zT, Malpha_expected, Mbeta_expected, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_only":
        
        eps = jrandom.uniform(subkey, shape=(N_sim,), minval=-0.5, maxval=0.5)
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = TimeOnly()
            
            rho=lambda t: -3/2
            phi=lambda t: 0
            theta = lambda t,x,v: jnp.pi/4
            a=lambda t,x,v: 7
            b=lambda t,x,v: 7./4
            c1 = lambda t,x,v: rho(t)*jnp.cos(phi(t)+e)
            c2 = lambda t,x,v: rho(t)*jnp.sin(phi(t)+e)
            M2 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
          
            Malpha.append(M1)
            Mbeta.append(M2) 
        
        key, subkey = jrandom.split(key)
        eps = jrandom.uniform(subkey, shape=(100,), minval=-0.5, maxval=0.5)
        
        Malpha_expected = TimeOnly()
        
        rho=lambda t: -3/2
        phi=lambda t: 0
        theta = lambda t,x,v,eps: jnp.pi/4
        a=lambda t,x,v,eps: 7
        b=lambda t,x,v,eps: a(t,x,v,eps)/4
        c1 = lambda t,x,v,eps: rho(t)*jnp.cos(phi(t)+eps)
        c2 = lambda t,x,v,eps: rho(t)*jnp.sin(phi(t)+eps)
        Mbeta_expected = ExpectedEllipticFinsler(subkey, eps, c1=c1,c2=c2, a=a,b=b,theta=theta)
        
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
        eps = jrandom.uniform(subkey, shape=(100,2), minval=0.4, maxval=0.6)
        Malpha_expected = ExpectedPointcarreLeft(subkey, eps[:,0])
        Mbeta_expected = ExpectedPointcarreRight(subkey, eps[:,1])
        
        k = 10.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([1.,1.], dtype=jnp.float32)
        zT = jnp.array([k,1.], dtype=jnp.float32)
        
        return t0, z0, zT, Malpha_expected, Mbeta_expected, tack_metrics, reverse_tack_metrics
    
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
        
#%% Load manifolds

def load_albatross_data(manifold:str = "poincarre",
                        file_path:str = '../../../../Data/albatross/tracking_data.xls', 
                        N_sim:int=10,
                        seed:int=2712,
                        ):
    
    albatross_data = pd.read_excel(file_path)
    bird_idx = [0,25,50]
    data_idx = {bird_idx[0]: [[67,90], [149, 195], [315, 335]],
                bird_idx[1]: [[30, 40], [115, 140], [140, 160]],
                bird_idx[2]: [[15, 35], [92, 108], [325, 370]]}
    
    
    track_id = albatross_data["TRACKID"].unique()
    time_data = []
    w1 = []
    w2 = []
    x1 = []
    x2 = []
    for id_val in track_id:
        dummy_data = albatross_data[albatross_data['TRACKID']==id_val].reset_index()
        t = pd.to_datetime(dummy_data['YMDHMS'], format='%Y%m%d%H%M%S')
        t = t-t.loc[0]
        time_data.append(jnp.array(t.dt.total_seconds().to_numpy()).squeeze()/3600)
        w1.append(jnp.array([dummy_data['WND_SPD_MS_5'].to_numpy()]).squeeze()*jnp.cos(dummy_data['WND_DIR'].to_numpy()/(2*jnp.pi)))
        w2.append(jnp.array([dummy_data['WND_SPD_MS_5'].to_numpy()]).squeeze()*jnp.sin(dummy_data['WND_DIR'].to_numpy()/(2*jnp.pi)))
        x1.append(jnp.array([dummy_data['LATITUDE'].to_numpy()]).squeeze())
        x2.append(jnp.array([dummy_data['LONGITUDE'].to_numpy()]).squeeze())
        
    x_data = [jnp.vstack((y1,y2)).T for y1,y2 in zip(x1,x2)]
    w_data = [jnp.vstack((y1,y2)).T for y1,y2 in zip(w1,w2)]
    
    t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
    z0 = [x_data[b_idx][idx_val[0]] for b_idx,vals in data_idx.items() for idx_val in vals]
    zT = [x_data[b_idx][idx_val[1]] for b_idx,vals in data_idx.items() for idx_val in vals]
    
    key = jrandom.key(seed)
    key, subkey = jrandom.split(key)
    
    if manifold == "direction_only":
        
        v_min = 0.0
        v_max = 20.0
        v_mean= v_max/2
        v_slope = 0.25

        frac_fun = lambda v: v_min/v_max+1.0/(1+jnp.exp(-v_slope*(jnp.linalg.norm(v)-v_mean)))

        eps = jrandom.uniform(subkey, shape=(N_sim,2), minval=-0.5, maxval=0.5)
        
        Malpha_stoch = []
        Mbeta_stoch = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            a=lambda t,x,v: jnp.linalg.norm(v)
            b=lambda t,x,v: jnp.linalg.norm(v)
            c1=lambda t,x,v: frac_fun(v)*jnp.linalg.norm(v)
            c2=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)*jnp.sqrt((1-frac_fun(v)**2))
            theta = lambda t,x,v: (jnp.pi/2-jnp.arctan(v[1]/v[0]))+e[0]#-jnp.pi/4
            M1 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            a=lambda t,x,v: jnp.linalg.norm(v)
            b=lambda t,x,v: jnp.linalg.norm(v)
            c1=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)
            c2=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)*jnp.sqrt((1-frac_fun(v)**2))
            theta = lambda t,x,v: (jnp.pi/2-jnp.arctan(v[1]/v[0]))+e[1]#-jnp.pi/4
            M2 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            Malpha_stoch.append(M1)
            Mbeta_stoch.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])

        a=lambda t,x,v: jnp.linalg.norm(v)
        b=lambda t,x,v: jnp.linalg.norm(v)
        c1=lambda t,x,v: frac_fun(v)*jnp.linalg.norm(v)
        c2=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)*jnp.sqrt((1-frac_fun(v)**2))
        theta = lambda t,x,v: (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        Malpha = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        a=lambda t,x,v: jnp.linalg.norm(v)
        b=lambda t,x,v: jnp.linalg.norm(v)
        c1=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)
        c2=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)*jnp.sqrt((1-frac_fun(v)**2))
        theta = lambda t,x,v: (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        Mbeta = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        return t0, z0, zT, Malpha, Mbeta, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_only":
        
        eps = jrandom.uniform(subkey, shape=(N_sim,), minval=-0.5, maxval=0.5)
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = TimeOnly()
            
            a=lambda t,x,v: jnp.linalg.norm(v)
            b=lambda t,x,v: jnp.linalg.norm(v)
            c1=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)
            c2=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)*jnp.sqrt((1-frac_fun(v)**2))
            theta = lambda t,x,v: (jnp.pi/2-jnp.arctan(v[1]/v[0]))+e#-jnp.pi/4
            M2 = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
          
            Malpha.append(M1)
            Mbeta.append(M2) 
        
        
        key, subkey = jrandom.split(key)
        eps = jrandom.uniform(subkey, shape=(100,), minval=-0.5, maxval=0.5)
        
        Malpha = TimeOnly()
        
        a=lambda t,x,v: jnp.linalg.norm(v)
        b=lambda t,x,v: jnp.linalg.norm(v)
        c1=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)
        c2=lambda t,x,v: -frac_fun(v)*jnp.linalg.norm(v)*jnp.sqrt((1-frac_fun(v)**2))
        theta = lambda t,x,v: (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        Mbeta = EllipticFinsler(c1=c1,c2=c2, a=a,b=b,theta=theta)
        
        return t0, z0, zT, Malpha, Mbeta, tack_metrics, reverse_tack_metrics
    
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

        Malpha = PointcarreLeft(d=0.5)
        Mbeta = PointcarreRight(d=0.5)
        
        return t0, z0, zT, Malpha, Mbeta, tack_metrics, reverse_tack_metrics
    
    return