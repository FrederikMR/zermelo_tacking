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

class PointcarreNorthLeft(LorentzFinslerManifold):
    def __init__(self,
                 )->None:

        super().__init__(F=self.F_metric)
        
        return
        
    def F_metric(self, t, z, dz):
        
        u, v = z[0], z[1]
        x,y = dz[0], dz[1]
        
        return 2.*(0.5*jnp.sqrt((3*(x**2))+2*x*y+3*(y**2))-0.5*x-0.5*y)/(10*jnp.arctan(v))
    
#%% Elliptic Finsler

class PointcarreNorthRight(LorentzFinslerManifold):
    def __init__(self,
                 )->None:

        super().__init__(F=self.F_metric)
        
        return
        
    def F_metric(self, t, z, dz):
        
        u, v = z[0], z[1]
        x,y = dz[0], dz[1]
        
        return 2.*(0.5*jnp.sqrt((3*(x**2))-2.*x*y+3*(y**2))-0.5*x+0.5*y)/(10*jnp.arctan(v))
    
#%% Elliptic Finsler

class ExpectedPointcarreNorthLeft(LorentzFinslerManifold):
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
        
        return 2.*(0.5*jnp.sqrt((3*(x**2))+2*x*y+3*(y**2))-0.5*x-0.5*y)/(3*jnp.arctan(v))
    
#%% Elliptic Finsler

class ExpectedPointcarreNorthRight(LorentzFinslerManifold):
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
        
        return 2.*(0.5*jnp.sqrt((3*(x**2))-2.*x*y+3*(y**2))-0.5*x+0.5*y)/(3*jnp.arctan(v))
