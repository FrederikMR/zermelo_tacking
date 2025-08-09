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

class TimeOnly(LorentzFinslerManifold):
    def __init__(self,
                 )->None:
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
        
    def F_metric(self, t, x, v):
        
        x,y = v[0], v[1]
        
        sin_t = jnp.sin(t)
        cos_t_pi4 = jnp.cos(t * jnp.pi / 4)
        sin_t_pi4 = jnp.sin(t * jnp.pi / 4)
    
        a = (2 + sin_t / 2)         # Common factor A
        b = (3/2 + (3 * sin_t) / 8) # Common factor B
        a2 = a**2
        b2 = b**2
        a4 = a**4
        b4 = b**4
    
        # Numerator
        term1 = a2 * sin_t_pi4 * y
        term2 = b2 * cos_t_pi4 * x
    
        sqrt_inner = (
            a4 * b2 * y**2 +
            a2 * b4 * x**2 -
            a2 * b2 * cos_t_pi4**2 * y**2 +
            2 * a2 * b2 * cos_t_pi4 * sin_t_pi4 * x * y -
            a2 * b2 * sin_t_pi4**2 * x**2
        )
    
        numerator = term1 + term2 + jnp.sqrt(sqrt_inner)
    
        # Denominator
        denominator = (
            a2 * b2 -
            a2 * sin_t_pi4**2 -
            b2 * cos_t_pi4**2
        )
    
        return numerator / denominator
        
        #Pi = jnp.pi
        #sin, cos = jnp.sin, jnp.cos
    
        #f = ((1.0*(2+1/2*sin(t))**2)*sin(1/4*t*jnp.pi)*y+1.0*((3/2+3/8*sin(t))**2)*cos(1/4*t*jnp.pi)*x \
        #     +(((2+1/2*sin(t))**4)*((3/2+3/8*sin(t))**2)*(y**2)+((2+1/2*sin(t))**2)*((3/2+3/8*sin(t))**4)*(x**2) \
        #       -1.00*((2+1/2*sin(t))**2)*((3/2+3/8*sin(t))**2)*(cos(1/4*t*Pi)**2)*(y**2) \
        #       +2.00*((2+1/2*sin(t))**2)*((3/2+3/8*sin(t))**2)*cos(1/4*t*Pi)*sin(1/4*t*Pi)*x*y \
        #       -1.00*(2+1/2*(sin(t))**2)*((3/2+3/8*sin(t))**2)*(sin(1/4*t*Pi)**2)*(x**2))**(1/2)) \
        #/(((2+1/2*sin(t))**2)*((3/2+3/8*sin(t))**2)-1.00*((2+1/2*sin(t))**2)*(sin(1/4*t*Pi)**2) \
        #  -1.00*((3/2+3/8*sin(t))**2)*(cos(1/4*t*Pi)**2))
    
        #return f