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
from geometry.geodesics import GEORCE
from geometry.geodesics import JAXOptimization

#%% Single Point Tacking Optimization

class SingleTackingOptimization(ABC):
    def __init__(self,
                 M1:FinslerManifold,
                 M2:FinslerManifold,
                 Geodesic1:object=None,
                 Geodesic2:object=None,
                 )->None:
        
        self.M1 = M1
        self.M2 = M2
        
        if Geodesic1 is None:
            Geo1 = GEORCE(M1)
            #Geo1 = JAXOptimization(M1, lr_rate=0.01, max_iter=1000)
            self.Geodesic1 = lambda z0,zT: Geo1(z0,zT,"for")[0][-1]
        else:
            self.Geodesic1 = Geodesic1
            
        if Geodesic2 is None:
            Geo2 = GEORCE(M2)
            #Geo2 = JAXOptimization(M2, lr_rate=0.01, max_iter=1000)
            self.Geodesic2 = lambda z0,zT: Geo2(z0,zT,"for")[0][-1]
        else:
            self.Geodesic2 = Geodesic2
        
        self.z0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Tacking Point Optimization with two Finsler metrics"
    
    def length(self, 
               z_tack:Array,
               )->Array:
        
        zt1 = self.Geodesic1(self.z0, z_tack)
        zt2 = self.Geodesic2(z_tack, self.zT)
        
        return self.M1.length(zt1)+self.M2.length(zt2)
    
    def Dlength(self,
                z_tack:Array,
                )->Array:
        
        zt1 = self.Geodesic1(self.z0, z_tack)
        zt2 = self.Geodesic2(z_tack, self.zT)
        
        return grad(self.length)(z_tack)
    
    def __call__(self,
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        self.z0 = z0
        self.zT = zT
        
        z_tack = z0+(zT-z0)*0.5
        
        #z_tack = jnp.array([-5.0,5.0])
        
        res = jminimize(self.length,
                        x0 = z_tack,
                        method="BFGS",
                        )
        
        z_tack = res.x
        zt1 = self.Geodesic1(self.z0, z_tack)
        zt2 = self.Geodesic2(z_tack, self.zT)
        
        return zt1, zt2, z_tack
        

#%% Single Point Tacking Optimization

class SingleTackingOptimization2(ABC):
    def __init__(self,
                 M1:FinslerManifold,
                 M2:FinslerManifold,
                 Geodesic:object,
                 )->None:
        
        self.M1 = M1
        self.M2 = M2
        self.Geodesic = Geodesic

        self.method = "exact"
        
        self.z0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Tacking Point Optimization with two Finsler metrics"
    
    def length(self, 
               zt:Array, 
               idx:int,
               )->Array:
        
        T = len(zt)+1
        dt = 1.0/T
        
        zt1 = zt[:idx]
        zt2 = zt[idx:]

        term1 = (zt1[0]-self.z0)*T
        val1 = self.M1.F(self.z0, term1)
        
        term2 = (zt1[1:]-zt1[:-1])*T
        val2 = vmap(lambda x,v: self.M1.F(x,v))(zt1[:-1], term2).squeeze()
        
        term3 = (zt2[0]-zt1[-1])*T
        val3 = self.M1.F(zt1[-1], term3)
        
        term4 = (zt2[1:]-zt2[:-1])*T
        val4 = vmap(lambda x,v: self.M2.F(x,v))(zt2[:-1], term4).squeeze()
        
        term5 = (self.zT-zt2[-1])*T
        val5 = self.M2.F(zt2[-1], term5)
        
        gamma = jnp.concatenate((val1.reshape(1),val2.reshape(-1),val3.reshape(1),val4.reshape(-1),val5.reshape(1)),axis=0)

        return jnp.trapz(gamma, dx=dt)
    
    def heuristic_estimate(self,
                           z0:Array,
                           zT:Array,
                           )->Array:
        
        idx = jnp.arange(0,self.T,1)
        chi = 0.1
        
        return
    
    def exact_estimate(self, 
                       z0:Array,
                       zT:Array,
                       )->Array:
        
        idx = lax.iota(jnp.int32, self.Geodesic.T)#jnp.arange(0,self.T,1)
        
        length = []
        zt = []
        for i in range(2,self.Geodesic.T-2):
            z = self.Geodesic(z0,zT,i)
            l = self.length(z,i)
            zt.append(z)
            length.append(l)
            print(f"Tacking point {i+1}\n\t-Length={l}")
        zt = jnp.stack(zt)
        length = jnp.stack(length)
        
        #zt = vmap(lambda i: self.geodesic(z0,zT,i))(idx)
        #length = vmap(lambda z,i:self.length(z, i))(zt,idx)
        
        #t = zt[:,0]
        
        tacking_point = jnp.argmin(length)#jnp.argmin(t)
        
        return zt[tacking_point], tacking_point, length[tacking_point]
    
    def __call__(self,
                 z0:Array,
                 zT:Array,
                 )->Array:
        
        self.z0 = z0
        self.zT = zT
        
        if self.method == "exact":
            zt, tacking_point, length = self.exact_estimate(z0, zT)        
        
        return zt, tacking_point, length