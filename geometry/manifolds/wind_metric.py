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

class RightWind(LorentzFinslerManifold):
    def __init__(self,
                 w:Array,
                 v_min:float,
                 v_max:float,
                 v_mean:float,
                 v_slope:float,
                 )->None:
        
        self.w = w
        self.v_min = v_min
        self.v_max = v_max
        self.v_mean = v_mean
        self.v_slope = v_slope
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
    
    def __str__(self)->str:
        return "Elliptic Finsler Metric"
        
    def frac_fun(self, v_norm):
        
        return self.v_min/self.v_max+1.0/(1+jnp.exp(-self.v_slope*(v_norm-self.v_mean)))
        
    def metric(self, v):
        
        v_norm = jnp.linalg.norm(v)
        
        frac = self.frac_fun(v_norm)
        
        #v /= 111111
        #v_norm = jnp.linalg.norm(v)
        
        a=v_norm
        b=v_norm
        c1=-frac*v_norm
        c2=-frac*v_norm*jnp.sqrt((1-frac**2))
        theta = (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        
        return a,b,c1,c2,theta
        
    def F_metric(self, t, x, v):

        a, b, c1, c2, theta = self.metric(self.w)
        
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
        
        return f.squeeze()
    
#%% Left Wind
    
class LeftWind(LorentzFinslerManifold):
    def __init__(self,
                 w:Array,
                 v_min:float,
                 v_max:float,
                 v_mean:float,
                 v_slope:float,
                 )->None:
        
        self.w = w
        self.v_min = v_min
        self.v_max = v_max
        self.v_mean = v_mean
        self.v_slope = v_slope
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
    
    def __str__(self)->str:
        return "Elliptic Finsler Metric"
        
    def frac_fun(self, v_norm):
        
        return self.v_min/self.v_max+1.0/(1+jnp.exp(-self.v_slope*(v_norm-self.v_mean)))
        
    def metric(self, v):
        
        v_norm = jnp.linalg.norm(v)
        
        frac = self.frac_fun(v_norm)
        
        #v /= 111111
        #v_norm = jnp.linalg.norm(v)
        
        a=v_norm
        b=v_norm
        c1=frac*v_norm
        c2=-frac*v_norm*jnp.sqrt((1-frac**2))
        theta = (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        
        return a,b,c1,c2,theta
        
    def F_metric(self, t, x, v):

        a,b,c1,c2,theta = self.metric(self.w)
        
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
        
        return f.squeeze()

#%% Elliptic Finsler

class StochasticRightWind(LorentzFinslerManifold):
    def __init__(self,
                 w:Array,
                 eps:Array,
                 v_min:float,
                 v_max:float,
                 v_mean:float,
                 v_slope:float,
                 )->None:
        
        self.w = w
        self.v_min = v_min
        self.v_max = v_max
        self.v_mean = v_mean
        self.v_slope = v_slope
        self.eps = eps
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
    
    def __str__(self)->str:
        return "Elliptic Finsler Metric"
        
    def frac_fun(self, v_norm):
        
        return self.v_min/self.v_max+1.0/(1+jnp.exp(-self.v_slope*(v_norm-self.v_mean)))
        
    def metric(self, v):
        
        v_norm = jnp.linalg.norm(v)
        
        frac = self.frac_fun(v_norm)
        
        #v /= 111111
        #v_norm = jnp.linalg.norm(v)
        
        a=v_norm
        b=v_norm
        c1=-frac*v_norm
        c2=-frac*v_norm*jnp.sqrt((1-frac**2))
        theta = (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        
        return a,b,c1,c2,theta
        
    def F_metric(self, t, x, v):

        a, b, c1, c2, theta = self.metric(self.w)
        theta += self.eps
        
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
        
        return f.squeeze()
    
#%% Left Wind
    
class StochasticLeftWind(LorentzFinslerManifold):
    def __init__(self,
                 w:Array,
                 eps:Array,
                 v_min:float,
                 v_max:float,
                 v_mean:float,
                 v_slope:float,
                 )->None:
        
        self.w = w
        self.v_min = v_min
        self.v_max = v_max
        self.v_mean = v_mean
        self.v_slope = v_slope
        self.eps = eps
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
    
    def __str__(self)->str:
        return "Elliptic Finsler Metric"
        
    def frac_fun(self, v_norm):
        
        return self.v_min/self.v_max+1.0/(1+jnp.exp(-self.v_slope*(v_norm-self.v_mean)))
        
    def metric(self, v):
        
        v_norm = jnp.linalg.norm(v)
        
        frac = self.frac_fun(v_norm)
        
        #v /= 111111
        #v_norm = jnp.linalg.norm(v)
        
        a=v_norm
        b=v_norm
        c1=frac*v_norm
        c2=-frac*v_norm*jnp.sqrt((1-frac**2))
        theta = (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        
        return a,b,c1,c2,theta
        
    def F_metric(self, t, x, v):
        
        a,b,c1,c2,theta = self.metric(self.w)
        theta += self.eps
        
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
        
        return f.squeeze()
    
#%% Expected Elliptic Finsler

class ExpectedLeftWind(LorentzFinslerManifold):
    def __init__(self,
                 w:Array,
                 eps:Array,
                 v_min:float,
                 v_max:float,
                 v_mean:float,
                 v_slope:float,
                 )->None:
        
        self.w = w
        self.v_min = v_min
        self.v_max = v_max
        self.v_mean = v_mean
        self.v_slope = v_slope
        self.eps = eps
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
    
    def __str__(self)->str:
        return "Elliptic Finsler Metric"
        
    def frac_fun(self, v_norm):
        
        return self.v_min/self.v_max+1.0/(1+jnp.exp(-self.v_slope*(v_norm-self.v_mean)))
        
    def metric(self, v):
        
        v_norm = jnp.linalg.norm(v)
        
        frac = self.frac_fun(v_norm)
        
        #v /= 111111
        #v_norm = jnp.linalg.norm(v)
        
        a=v_norm
        b=v_norm
        c1=frac*v_norm
        c2=-frac*v_norm*jnp.sqrt((1-frac**2))
        theta = (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        
        return a,b,c1,c2,theta
    
    def F_metric(self, t, x, v):
        
        return jnp.mean(vmap(self.F_sample, in_axes=(None,None,None,0))(t,x,v,self.eps), axis=0)
        
    def F_sample(self, t, x, v, eps):
        
        a,b,c1,c2,theta = self.metric(self.w)
        theta += eps
        
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
        
        return f.squeeze()
    
#%% Expected Elliptic Finsler

class ExpectedRightWind(LorentzFinslerManifold):
    def __init__(self,
                 w:Array,
                 eps:Array,
                 v_min:float,
                 v_max:float,
                 v_mean:float,
                 v_slope:float,
                 )->None:
        
        self.w = w
        self.v_min = v_min
        self.v_max = v_max
        self.v_mean = v_mean
        self.v_slope = v_slope
        self.eps = eps
        
        self.dim = 2
        self.emb_dim = None

        super().__init__(F=self.F_metric)
        
        return
    
    def __str__(self)->str:
        return "Elliptic Finsler Metric"
        
    def frac_fun(self, v_norm):
        
        return self.v_min/self.v_max+1.0/(1+jnp.exp(-self.v_slope*(v_norm-self.v_mean)))
        
    def metric(self, v):
        
        v_norm = jnp.linalg.norm(v)
        
        frac = self.frac_fun(v_norm)
        
        #v /= 111111
        #v_norm = jnp.linalg.norm(v)
        
        a=v_norm
        b=v_norm
        c1=-frac*v_norm
        c2=-frac*v_norm*jnp.sqrt((1-frac**2))
        theta = (jnp.pi/2-jnp.arctan(v[1]/v[0]))#-jnp.pi/4
        
        return a,b,c1,c2,theta
    
    def F_metric(self, t, x, v):
        
        return jnp.mean(vmap(self.F_sample, in_axes=(None,None,None,0))(t,x,v,self.eps), axis=0)
        
    def F_sample(self, t, x, v, eps):

        a, b, c1, c2, theta = self.metric(self.w)
        theta += eps
        
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
        
        return f.squeeze()