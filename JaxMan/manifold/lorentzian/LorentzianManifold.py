#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:25:40 2023

@author: fmry
"""

#%% Sources

#%% Modules

from JaxMan.initialize import *
from JaxMan.manifold.lorentzian.LorentzianMetric import LorentzianMetric
from JaxMan.manifold.lorentzian.LorentzianCurvature import LorentzianCurvature
from JaxMan.manifold.lorentzian.LorentzianDistance import LorentzianDistance

from JaxMan.integration.JaxIntegration import rectangle, trapezoidal, simpson
from JaxMan.integration.JaxODE import euler, heun, rk4

#%% Lorentzian Manifold

class LorentzianManifold(object):
    
    def __init__(self,
                 h:Callable[[float, jnp.ndarray], jnp.ndarray],
                 dim:int,
                 GeodesicMethod:str = "Energy",
                 IntegrationMethod:str = "simpson",
                 ODEMethod:str = "rk4",
                 optimizer:str = "BFGS",
                 dt:float=1e-2,
                 tol:float=1e-3,
                 maxiter:int = 1000):
        
        self.G = lambda t,x: jnp.block([[jnp.ones(1), jnp.zeros((1,dim))],
                                        [jnp.zeros((dim,1)), -h(t,x)]])
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
        
        LorentzianMetric(self)
        LorentzianCurvature(self)
        LorentzianDistance(self)
        
    def __str__(self):
        
        return "Lorentzian Manifold of dimension {}".format(self.dim)
    
    
    
    
    