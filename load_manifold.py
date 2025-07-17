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

import pandas as pd

from typing import Tuple

from geometry.manifolds import EllipticFinsler, PointcarreLeft, PointcarreRight, TimeOnly
from geometry.manifolds import ExpectedEllipticFinsler, ExpectedPointcarreLeft, ExpectedPointcarreRight
from geometry.manifolds import StochasticEllipticFinsler
from geometry.manifolds import LeftWind, RightWind, StochasticLeftWind, StochasticRightWind
from geometry.manifolds import ExpectedLeftWind, ExpectedRightWind
from geometry.manifolds import PointcarreNorthLeft, PointcarreNorthRight, ExpectedPointcarreNorthLeft, ExpectedPointcarreNorthRight

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
        z0 = jnp.array([0.,1.], dtype=jnp.float32)
        zT = jnp.array([20.,1.], dtype=jnp.float32)
        
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
        
        k = 20.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,1.], dtype=jnp.float32)
        zT = jnp.array([k,1.], dtype=jnp.float32)
        
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
        
#%% Load Manifolds

def load_stochastic_manifold(manifold:str="direction_only", 
                             seed:int=2712,
                             N_sim:int=10,
                             alpha:float=1.0,
                             ):
    
    key = jrandom.key(seed)
    key, subkey = jrandom.split(key)
    
    if manifold == "direction_only":
        
        c1_m1=lambda t,x,v,eps: -1.5*jnp.cos(jnp.pi/2+6*jnp.pi/10)
        c2_m1=lambda t,x,v,eps: -1.5*jnp.sin(jnp.pi/2+6*jnp.pi/10) 
        a_m1=lambda t,x,v,eps: 2
        b_m1=lambda t,x,v,eps: 2
        theta_m1=lambda t,x,v,eps: eps+jnp.pi
        
        c1_m2=lambda t,x,v,eps: 3/4
        c2_m2=lambda t,x,v,eps: 0
        a_m2=lambda t,x,v,eps: 1
        b_m2=lambda t,x,v,eps: 1
        theta_m2=lambda t,x,v,eps: eps+0
        
        sigma = 1.0
        
        eps = sigma*jrandom.normal(subkey, shape=(N_sim,2))
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = StochasticEllipticFinsler(eps=e[0],
                                           c1=c1_m1,
                                           c2=c2_m1, 
                                           a=a_m1,
                                           b=b_m1,
                                           theta=theta_m1,
                                           )
            M2 = StochasticEllipticFinsler(eps=e[1],
                                           c1=c1_m2,
                                           c2=c2_m2, 
                                           a=a_m2,
                                           b=b_m2,
                                           theta=theta_m2,
                                           )
            
            Malpha.append(M1)
            Mbeta.append(M2)
        
        tack_metrics = [(m1, m2) for m1, m2 in zip(Malpha, Mbeta)]
        reverse_tack_metrics = [(m2, m1) for m1, m2 in zip(Malpha, Mbeta)]
        
        key, subkey = jrandom.split(key)
        eps = sigma*jrandom.normal(subkey, shape=(100,2))
        
        Malpha_expected = ExpectedEllipticFinsler(eps[:,0], 
                                                  c1=c1_m1,
                                                  c2=c2_m1, 
                                                  a=a_m1,
                                                  b=b_m1,
                                                  theta=theta_m1,
                                                  )
        Mbeta_expected = ExpectedEllipticFinsler(eps[:,1], 
                                                 c1=c1_m2,
                                                 c2=c2_m2, 
                                                 a=a_m2,
                                                 b=b_m2,
                                                 theta=theta_m2,
                                                 )
        
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([0.,0.], dtype=jnp.float32)
        zT = jnp.array([2.,8.], dtype=jnp.float32)
        
        return t0, z0, zT, Malpha_expected, Mbeta_expected, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_only":
        
        c1_m2=lambda t,x,v,eps: -1.5
        c2_m2=lambda t,x,v,eps: 0.
        a_m2=lambda t,x,v,eps: 7.
        b_m2=lambda t,x,v,eps: 7./4
        theta_m2=lambda t,x,v,eps: eps+jnp.pi/4
        
        sigma = 1.0
        
        eps = sigma*jrandom.normal(subkey, shape=(N_sim,))
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = TimeOnly()
            M2 = StochasticEllipticFinsler(eps=e,
                                           c1=c1_m2,
                                           c2=c2_m2, 
                                           a=a_m2,
                                           b=b_m2,
                                           theta=theta_m2,
                                           )
          
            Malpha.append(M1)
            Mbeta.append(M2) 
            
        tack_metrics = [(m1, m2) for m1, m2 in zip(Malpha, Mbeta)]
        reverse_tack_metrics = [(m2, m1) for m1, m2 in zip(Malpha, Mbeta)]
        
        key, subkey = jrandom.split(key)
        eps = sigma*jrandom.normal(subkey, shape=(100,))
        
        Malpha_expected = TimeOnly()
        Mbeta_expected = ExpectedEllipticFinsler(eps, 
                                                 c1=c1_m2,
                                                 c2=c2_m2, 
                                                 a=a_m2,
                                                 b=b_m2,
                                                 theta=theta_m2,
                                                 )
        
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
            M1 = PointcarreLeft(d=e[0], alpha=alpha)
            M2 = PointcarreRight(d=e[1], alpha=alpha)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
        tack_metrics = [(m1, m2) for m1, m2 in zip(Malpha, Mbeta)]
        reverse_tack_metrics = [(m2, m1) for m1, m2 in zip(Malpha, Mbeta)]

        key, subkey = jrandom.split(key)
        eps = jrandom.uniform(subkey, shape=(100,2), minval=0.4, maxval=0.6)
        Malpha_expected = ExpectedPointcarreLeft(subkey, eps[:,0], alpha=alpha)
        Mbeta_expected = ExpectedPointcarreRight(subkey, eps[:,1], alpha=alpha)
        
        k = 10.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([1.,1.], dtype=jnp.float32)
        zT = jnp.array([k,1.], dtype=jnp.float32)
        
        return t0, z0, zT, Malpha_expected, Mbeta_expected, tack_metrics, reverse_tack_metrics
    elif manifold == "poincarre_north":
        
        eps = jrandom.uniform(subkey, shape=(N_sim,2), minval=1.0, maxval=3.0)
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = PointcarreNorthLeft()
            M2 = PointcarreNorthRight()
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
        tack_metrics = [(m1, m2) for m1, m2 in zip(Malpha, Mbeta)]
        reverse_tack_metrics = [(m2, m1) for m1, m2 in zip(Malpha, Mbeta)]

        key, subkey = jrandom.split(key)
        eps = jrandom.uniform(subkey, shape=(100,2), minval=1.0, maxval=3.0)
        Malpha_expected = ExpectedPointcarreNorthLeft(subkey, eps[:,0])
        Mbeta_expected = ExpectedPointcarreNorthRight(subkey, eps[:,1])
        
        k = 10.
        t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
        z0 = jnp.array([1.,1.], dtype=jnp.float32)
        zT = jnp.array([k,1.], dtype=jnp.float32)
        
        return t0, z0, zT, Malpha_expected, Mbeta_expected, tack_metrics, reverse_tack_metrics
    
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
        
#%% Load Albatross data

def load_albatross_data(file_path:str = '../../../../Data/albatross/tracking_data.xls', 
                        )->Tuple:
    
    albatross_data = pd.read_excel(file_path)
    bird_idx = [0,25,50]
    data_idx = {bird_idx[0]: [[67,90], [149, 195], [315, 335]],
                bird_idx[1]: [[30, 40], [115, 140], [140, 160]],
                bird_idx[2]: [[15, 35], [92, 108], [325, 370]],
                }
    
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
    
    return x_data, time_data, x1, x2, w1, w2, w_data, data_idx

#%% Load Albatross Metrics

def load_albatross_metrics(manifold:str = "poincarre",
                           file_path:str = '../../../../Data/albatross/tracking_data.xls', 
                           N_sim:int=5,
                           seed:int=2712,
                           idx_birds:int = 0,
                           idx_data:int = 0,
                           alpha:float=1.0,
                           ):
    
    x_data, time_data, x1, x2, w1, w2, w_data, data_idx = load_albatross_data(file_path)
    
    bird_idx = list(data_idx.keys())
    bird_type = bird_idx[idx_birds]
    start_idx = data_idx[bird_idx[idx_birds]][idx_data][0]
    end_idx = data_idx[bird_idx[idx_birds]][idx_data][1]
    
    t0 = jnp.zeros(1, dtype=jnp.float32).squeeze()
    z0 = x_data[bird_type][start_idx]
    zT = x_data[bird_type][end_idx]

    w_val = w_data[bird_type][start_idx]
    
    key = jrandom.key(seed)
    key, subkey = jrandom.split(key)
    
    v_min = 0.0
    v_max = 20.0
    v_mean= v_max/2
    v_slope = 0.25
    
    if manifold == "direction_only":
        
        sigma = 1.0

        eps = sigma*jrandom.normal(subkey, shape=(N_sim,2))
        
        Malpha_stoch = []
        Mbeta_stoch = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = StochasticLeftWind(w=w_val,
                                    eps=e[0],
                                    v_min = v_min,
                                    v_max = v_max,
                                    v_mean = v_mean,
                                    v_slope = v_slope,
                                    )

            M2 = StochasticRightWind(w=w_val,
                                     eps=e[1],
                                     v_min = v_min,
                                     v_max = v_max,
                                     v_mean = v_mean,
                                     v_slope = v_slope,
                                     )
            
            Malpha_stoch.append(M1)
            Mbeta_stoch.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])

        Malpha = LeftWind(w=w_val,
                          v_min = v_min,
                          v_max = v_max,
                          v_mean = v_mean,
                          v_slope = v_slope,
                          )

        Mbeta = RightWind(w=w_val,
                          v_min = v_min,
                          v_max = v_max,
                          v_mean = v_mean,
                          v_slope = v_slope,
                          )
        
        eps = sigma*jrandom.normal(subkey, shape=(100,2))
        
        MEalpha = ExpectedLeftWind(w=w_val,
                                   eps=eps[:,0],
                                   v_min = v_min,
                                   v_max = v_max,
                                   v_mean = v_mean,
                                   v_slope = v_slope,
                                   )

        MEbeta = ExpectedRightWind(w=w_val,
                                   eps=eps[:,1],
                                   v_min = v_min,
                                   v_max = v_max,
                                   v_mean = v_mean,
                                   v_slope = v_slope,
                                   )
        
        
        return t0, z0, zT, Malpha, Mbeta, MEalpha, MEbeta, tack_metrics, reverse_tack_metrics
    
    elif manifold == "time_only":
        
        sigma = 1.0
        
        eps = sigma*jrandom.normal(subkey, shape=(N_sim,))
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = TimeOnly()
            M2 = StochasticRightWind(w=w_val,
                                     eps=e,
                                     v_min = v_min,
                                     v_max = v_max,
                                     v_mean = v_mean,
                                     v_slope = v_slope,
                                     )
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])
          
            Malpha.append(M1)
            Mbeta.append(M2) 
        
        
        key, subkey = jrandom.split(key)
        eps = sigma*jrandom.normal(subkey, shape=(100,))
        
        MEalpha = TimeOnly()
        MEbeta = ExpectedRightWind(w=w_val,
                                   eps=eps,
                                   v_min = v_min,
                                   v_max = v_max,
                                   v_mean = v_mean,
                                   v_slope = v_slope,
                                   )
        
        Malpha = TimeOnly()
        Mbeta = RightWind(w=w_val,
                          v_min = v_min,
                          v_max = v_max,
                          v_mean = v_mean,
                          v_slope = v_slope,
                          )
        
        return t0, z0, zT, Malpha, Mbeta, MEalpha, MEbeta, tack_metrics, reverse_tack_metrics
    
    elif manifold == "poincarre":
        
        eps = jrandom.uniform(subkey, shape=(N_sim,2), minval=0.4, maxval=0.6)
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = PointcarreLeft(d=e[0], alpha=alpha)
            M2 = PointcarreRight(d=e[1], alpha=alpha)
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])

        Malpha = PointcarreLeft(d=0.5, alpha=alpha)
        Mbeta = PointcarreRight(d=0.5, alpha=alpha)
        
        eps = jrandom.uniform(subkey, shape=(100,2), minval=0.4, maxval=0.6)
        MEalpha = ExpectedPointcarreLeft(subkey, eps[:,0], alpha=alpha)
        MEbeta = ExpectedPointcarreRight(subkey, eps[:,1], alpha=alpha)
        
        return t0, z0, zT, Malpha, Mbeta, MEalpha, MEbeta, tack_metrics, reverse_tack_metrics
    
    elif manifold == "poincarre_north":
        
        eps = jrandom.uniform(subkey, shape=(N_sim,2), minval=1.0, maxval=3.0)
        
        Malpha = []
        Mbeta = []
        tack_metrics = []
        reverse_tack_metrics = []
        for e in eps:
            M1 = PointcarreNorthLeft()
            M2 = PointcarreNorthRight()
            
            Malpha.append(M1)
            Mbeta.append(M2)
            
            tack_metrics.append([M1, M2])
            reverse_tack_metrics.append([M2, M1])

        Malpha = PointcarreNorthLeft()
        Mbeta = PointcarreNorthRight()
        
        eps = jrandom.uniform(subkey, shape=(100,2), minval=1.0, maxval=3.0)
        MEalpha = ExpectedPointcarreNorthLeft(subkey, eps[:,0])
        MEbeta = ExpectedPointcarreNorthRight(subkey, eps[:,1])
        
        return t0, z0, zT, Malpha, Mbeta, MEalpha, MEbeta, tack_metrics, reverse_tack_metrics
    
    return
