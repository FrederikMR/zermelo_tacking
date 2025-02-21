#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:53:14 2024

@author: fmry
"""

#%% Modules

from geometry.setup import *

from geometry.manifolds import LorentzFinslerManifold

#%% GEORCE Estimation of Tack Points and Geodesics

class ConstantTacking(ABC):
    def __init__(self,
                 Malpha:LorentzFinslerManifold,
                 Mbeta:LorentzFinslerManifold,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 )->None:
        
        self.Malpha = Malpha
        self.Mbeta = Mbeta
        
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Tacking Computation Object using ADAM and GEORCE Optimizers"
    
    def geodesic(self,
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        return (zT-z0)*jnp.linspace(0.0,1.0,self.T,endpoint=False,dtype=z0.dtype)[1:].reshape(-1,1)+z0

    def travel_time(self,
                    p:Array,
                    )->Array:
        
        return jnp.abs(self.Malpha.F(self.t0, jnp.zeros(self.dim),p-self.z0))+jnp.abs(self.Mbeta.F(self.t0, jnp.zeros(self.dim),self.zT-p))
    
    def obj_fun(self,
                p:Array,
                )->Array:
        
        return self.Malpha.F(self.t0, jnp.zeros(self.dim),p-self.z0)+self.Mbeta.F(self.t0, jnp.zeros(self.dim),self.zT-p)
    
    def __call__(self, 
                 t0:Array,
                 z0:Array,
                 zT:Array,
                 n_tacks:int=1,
                 )->Array:
        
        self.t0 = t0
        self.z0 = z0
        self.zT = zT
        self.dim = len(z0)
        
        p = z0+(zT-z0)*0.5
        
        res = jminimize(fun=self.obj_fun,
                        x0=p.reshape(-1),
                        method="BFGS",
                        tol=self.tol,
                        options={'maxiter':self.max_iter},
                        )
        
        #t = res.fun
        p = res.x.reshape(-1)
        zt = jnp.vstack((self.z0,
                         self.geodesic(self.z0, p),
                         self.geodesic(p, self.zT),
                         self.zT,
                         ))

        t = self.travel_time(p)
        grad = res.jac
        idx = res.nit
        
        return t.reshape(1), zt, grad, idx
    