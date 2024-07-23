#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.finsler.manifolds import FinslerManifold
from geometry.finsler.geodesics import GEORCE

#%% Gradient Descent Estimation of Geodesics

class ScipyOptimization(ABC):
    def __init__(self,
                 M:List[FinslerManifold],
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 )->None:

        self.M = M
        self.Geodesic = lambda z0, zT, M: z0+(zT-z0)*jnp.linspace(0,1,100,endpoint=True).reshape(-1,1)#GEORCE(M, max_iter=100, T=T)(z0,zT,"for")[0][-1]
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        self.save_zt = []
        
        self.dim = None
        self.z0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Scipy Optimizers"
    
    def init_length(self,
                    zt:Array,
                    zT:Array,
                    M:FinslerManifold,
                    *args
                    )->Array:
        
        return M.length(jnp.vstack((self.z0,zt,zT)))
    
    def end_length(self, 
                   zt:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:
        
        return M.length(jnp.vstack((zt,self.zT)))
    
    def mid_length(self, 
                   zt:Array,
                   zT:Array,
                   M:FinslerManifold,
                   *args
                   )->Array:
        
        return M.length(jnp.vstack((zt,zT)))
    
    def length(self, 
               zt:Array, 
               *args
               )->Array:
            
        z0 = zt[0]
        zT = zt[-1]
        zt = zt[1:-1].reshape(self.n_curves, -1, self.dim)
        
        zt_first = zt[0]
        zT_first = zt[1][0]
        zt_end = zt[-1]
        M0 = self.M[0]
        MT = self.M[-1]

        l1 = self.init_length(zt_first, zT_first, M0)
        lT = self.end_length(zt_end, MT)
        if self.N_tacks>1:
            l_tacks = jnp.sum(jnp.stack([self.mid_length(zt[i],zt[i+1][0],self.M[i]) for i in range(1,len(ztacks)+1)]))
        else:
            l_tacks = 0.0
            
        return l1+lT+l_tacks
    
    def obj_fun(self, 
                ztack_point:Array, 
                *args
                )->Array:
        
        ztack_point = ztack_point.reshape(-1, self.dim)

        l1 = self.M[0].length(self.Geodesic(self.z0, ztack_point[0], self.M[0]))
        lT = self.M[self.N_tacks].length(self.Geodesic(ztack_point[-1], self.zT, self.M[self.N_tacks]))
        
        if self.N_tacks>1:
            Mtacks = self.M[1:-1]
            ztacks = ztack_point[0:-1]
            l_tacks = jnp.stack([Mtacks[i].length(self.Geodesic(ztack_point[i], 
                                                                ztack_point[i+1], 
                                                                Mtacks[i])) for i in range(len(ztack_point)-1)])
        else:
            l_tacks = jnp.zeros(1)
            
        return l1+lT+jnp.sum(l_tacks)
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 N_tacks:int = 1,
                 )->Array:
        
        if N_tacks is None:
            N_tacks = len(self.M)-1
            
        self.n_curves = N_tacks+1
        self.N_tacks = N_tacks
        self.dim = len(z0)
        
        tack_time = jnp.array([1.0/(N_tacks+1)]*N_tacks)
        ztack = z0+(zT-z0)*tack_time.reshape(-1,1)
        
        self.z0 = z0
        self.zT = zT
        
        
        res = jminimize(fun = self.obj_fun,
                        x0=ztack.reshape(-1), 
                        method="BFGS",
                        tol=self.tol,
                        options={'maxiter':self.max_iter}
                        )
        
        ztack_point = res.x.reshape(-1, self.dim)
        
        geo1 = jnp.vstack((self.z0, self.Geodesic(self.z0, ztack_point[0], self.M[0]), ztack_point[0]))
        geoT = jnp.vstack((ztack_point[-1], self.Geodesic(ztack_point[-1], self.zT, self.M[self.N_tacks]),self.zT))
        
        if self.N_tacks>1:
            Mtacks = self.M[1:-1]
            ztacks = ztack_point[0:-1]
            geot = jnp.stack([jnp.vstack((ztack_point[i],
                                          self.Geodesic(ztack_point[i], 
                                                      ztack_point[i+1], Mtacks[i]),
                                          ztack_point[i+1])) for i in range(len(ztack_point)-1)])
            zt = jnp.vstack((geo1,geot.reshape(-1,self.dim),geoT))
        else:
            zt = jnp.vstack((geo1, geoT))
        
        
        return zt    
    
    