#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:57:46 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JaxMan.initialize import *
from JaxMan.manifold.riemannian.RiemannianMetric import RiemannianMetric
from JaxMan.manifold.riemannian.RiemannianCurvature import RiemannianCurvature
from JaxMan.manifold.riemannian.RiemannianDistance import RiemannianDistance

from JaxMan.integration.JaxIntegration import rectangle, trapezoidal, simpson
from JaxMan.integration.JaxODE import euler, heun, rk4

#%% Riemannian Manifold Class

class RiemannianManifold(object):
    
    def __init__(self,
                 G:Callable[[jnp.ndarray], jnp.ndarray],
                 dim:int,
                 GeodesicMethod:str = "Energy",
                 IntegrationMethod:str = "simpson",
                 ODEMethod:str = "rk4",
                 optimizer:str = "BFGS",
                 dt:float=1e-2,
                 tol:float=1e-6,
                 maxiter:int = 1000):
    
        self.G = G
        self.dim = dim
        self.GeodesicMethod=GeodesicMethod
        self.optimizer = optimizer
        self.dt = dt
        self.tol = tol
        self.maxiter = maxiter
        
        if IntegrationMethod == "rectangle":
            self.IntegrationMethod = rectangle
        elif IntegrationMethod == "trapezoidal":
            self.IntegrationMethod = trapezoidal
        elif IntegrationMethod == "simpson":
            self.IntegrationMethod = simpson
        else:
            raise ValueError('Unsupported integration method!')
            
        if ODEMethod == "euler":
            self.ODEMethod = euler
        elif ODEMethod == "heun":
            self.ODEMethod = heun
        elif ODEMethod == "rk4":
            self.ODEMethod = rk4
        else:
            raise ValueError('Unsupported ODE method!')
        
        
        
        RiemannianMetric(self)
        RiemannianCurvature(self)
        RiemannianDistance(self)
        
    def __str__(self):
        
        return "Riemannian Manifold of dimension {}".format(self.dim)
    
    def ChartUpdate(self):
        
        return
    