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

class EllipticFinsler(LorentzFinslerManifold):
    def __init__(self,
                 c1:Callable=lambda t,x,v: 1.0, 
                 c2:Callable=lambda t,x,v: 1.0,
                 a:Callable=lambda t,x,v: 1.0,
                 b:Callable=lambda t,x,v: 1.0,
                 theta:Callable=lambda t,x,v: jnp.pi/4,
                 )->None:
        
        self.c1 = c1
        self.c2 = c2
        self.a = a
        self.b = b
        self.theta = theta
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
        
    def F_metric(self, t, x, v):
        
        c1 = self.c1(t,x,v)
        c2 = self.c2(t,x,v)
        a = self.a(t,x,v)
        b = self.b(t,x,v)
        theta = self.theta(t,x,v)
        
        x,y = v[0], v[1]
        
        a2 = a**2
        b2 = b**2
        a4 = a**4
        b4 = b**4
        c12 = c1**2
        c22 = c2**2

        f = (-a2*c2*(x*jnp.sin(theta)+y*jnp.cos(theta))-\
             b2*c1*(x*jnp.cos(theta)-y*jnp.sin(theta))+\
             (a4*b2*(x*jnp.sin(theta)+y*jnp.cos(theta))**2+\
              a2*b4*(x*jnp.cos(theta)-y*jnp.sin(theta))**2-\
              a2*b2*c12*(x*jnp.sin(theta)+y*jnp.cos(theta))**2+\
              2*a2*b2*c1*c2*(x*jnp.cos(theta)-y*jnp.sin(theta))*(x*jnp.sin(theta)+y*jnp.cos(theta))-\
              a2*b2*c22*(x*jnp.cos(theta)-y*jnp.sin(theta))**2)**(1/2))/(a2*b2-a2*c22-b2*c12)
        
        return f