#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:18:38 2025

@author: fmry
"""

#%% Modules

from geometry.setup import *

####################

from .manifold import LorentzFinslerManifold

#%% Elliptic Finsler

class PointcarreLeft(LorentzFinslerManifold):
    def __init__(self,
                 d:float=0.5,
                 )->None:
        
        self.d = d

        super().__init__(F=self.F_metric)
        
        return
        
    def F_metric(self, t, z, dz):
        
        u, v = z[0], z[1]
        x,y = dz[0], dz[1]
        
        d = self.d
        d2 = d**2
        
        return (-jnp.sqrt((x**2)+(y**2)-((x-y)**2)*d2)+(x+y)*d)/(2*d2*v-v)
    
#%% Elliptic Finsler

class PointcarreRight(LorentzFinslerManifold):
    def __init__(self,
                 d:float=0.5,
                 )->None:
        
        self.d = d

        super().__init__(F=self.F_metric)
        
        return
        
    def F_metric(self, t, z, dz):
        
        u, v = z[0], z[1]
        x,y = dz[0], dz[1]
        
        d = self.d
        d2 = d**2
        
        return (-jnp.sqrt((x**2)+(y**2)-((x+y)**2)*d2)+(x-y)*d)/(2*d2*v-v)
    
#%% Elliptic Finsler

class ExpectedPointcarreLeft(LorentzFinslerManifold):
    def __init__(self,
                 subkey,
                 eps:Array,
                 )->None:
        
        self.eps = eps

        super().__init__(F=self.F_metric)
        
        return
        
    def F_metric(self, t, z, dz):
        
        return jnp.mean(vmap(self.F_sample, in_axes=(None,None,None,0))(t,z,dz,self.eps), axis=0)
    
    def F_sample(self, t, z, dz, eps):
        
        u, v = z[0], z[1]
        x,y = dz[0], dz[1]
        
        d = eps
        d2 = d**2
        
        return (-jnp.sqrt((x**2)+(y**2)-((x-y)**2)*d2)+(x+y)*d)/(2*d2*v-v)
    
#%% Elliptic Finsler

class ExpectedPointcarreRight(LorentzFinslerManifold):
    def __init__(self,
                 subkey,
                 eps:Array,
                 )->None:
        
        self.eps = eps

        super().__init__(F=self.F_metric)
        
        return
        
    def F_metric(self, t, z, dz):
        
        return jnp.mean(vmap(self.F_sample, in_axes=(None,None,None,0))(t,z,dz,self.eps), axis=0)
        
    def F_sample(self, t, z, dz, eps):
        
        u, v = z[0], z[1]
        x,y = dz[0], dz[1]
        
        d = eps
        d2 = d**2
        
        return (-jnp.sqrt((x**2)+(y**2)-((x+y)**2)*d2)+(x-y)*d)/(2*d2*v-v)