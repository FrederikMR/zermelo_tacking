#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.manifolds import FinslerManifold

#%% Gradient Descent Estimation of Geodesics

class BFGSOptimization(ABC):
    def __init__(self,
                 M1:FinslerManifold,
                 M2:FinslerManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 )->None:
            
        if init_fun is None:
            self.init_fun = self.init_default
            
        self.M1 = M1
        self.M2 = M2
        self.T = T
        
        self.save_zt = []
        
        self.dim = None
        self.z0 = None
        self.zT = None
        self.idx = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using BFGS"
    
    def init_default(self, 
                     z0:Array, 
                     zT:Array,
                     idx:int,
                     T:int,
                     )->Array:
        
        z_mid = jnp.array([-jnp.cos(jnp.pi/4), jnp.sin(jnp.pi/4)])
        t1 = jnp.linspace(0.0,1.0,idx,endpoint=False, dtype=z0.dtype)[1:].reshape(-1,1)
        t2 = jnp.linspace(0.0,1.0,T-idx,endpoint=False, dtype=z0.dtype)[1:].reshape(-1,1)

        if len(t1)==0:
            curve1 = z0
        else:
            curve1 = z0+(z_mid-z0)*t1
        if len(t2)==0:
            curve2=zT
        else:
            curve2 = curve1[-1]+(zT-z_mid)*t2
        
        return jnp.vstack((curve1.reshape(-1,len(z0)), curve2.reshape(-1,len(z0))))
        
        #return z0+(zT-z0)*jnp.linspace(0.0,1.0,T,endpoint=False, dtype=z0.dtype)[1:].reshape(-1,1)
    
    def energy(self, 
               zt:Array, 
               )->Array:
        
        zt = zt.reshape(-1, self.dim)
        
        zt1 = zt[:self.idx]
        zt2 = zt[self.idx:]

        term1 = zt1[0]-self.z0
        val1 = self.M1.F(self.z0, term1)**2
        
        term2 = zt1[1:]-zt1[:-1]
        val2 = vmap(lambda x,v: self.M1.F(x,v)**2)(zt1[:-1], term2)
        
        term3 = zt2[0]-zt1[-1]
        val3 = self.M1.F(zt1[-1], term3)**2
        
        term4 = zt2[1:]-zt2[:-1]
        val4 = vmap(lambda x,v: self.M2.F(x,v)**2)(zt2[:-1], term4)
        
        term5 = self.zT-zt2[-1]
        val5 = self.M2.F(zt2[-1], term5)**2

        return val1+jnp.sum(val2)+val3+jnp.sum(val4)+val5
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(lambda z: self.energy(z))(zt)
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 idx:int,
                 )->Array:
        
        self.idx = idx
        self.dim = len(z0)
        zt = self.init_fun(z0,zT,idx,self.T)
        
        self.z0 = z0
        self.zT = zT
        
        res = jminimize(self.energy,
                        x0 = zt.reshape(-1),
                        method="BFGS",
                        )
        
        return res.x.reshape(-1,self.dim)
    
    
    