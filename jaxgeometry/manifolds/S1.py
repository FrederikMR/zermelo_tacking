## This file is part of Jax Geometry
#
# Copyright (C) 2021, Stefan Sommer (sommer@di.ku.dk)
# https://bitbucket.org/stefansommer/jaxgeometry
#
# Jax Geometry is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Jax Geometry is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Jax Geometry. If not, see <http://www.gnu.org/licenses/>.
#

#%% Sources

#%% Modules

from jaxgeometry.setup import *
import jaxgeometry.manifolds.riemannian as riemannian

#%% Circle

class S1(riemannian.EmbeddedManifold):
    """ 2d Ellipsoid """
    
    def __str__(self):
        
        return "Circle S1"
    
    def __init__(self, 
                 angle_shift:float = jnp.pi,
                 use_spherical_coords = False):
        
        self.use_spherical_coords = use_spherical_coords
        self.angle_shift = angle_shift
        
        self.F_spherical = lambda x: jnp.array([jnp.cos(x[0]), jnp.sin(x[0])]).reshape(2)
        self.F_spherical_inv = lambda x: (jnp.arctan2(x[1][1],x[1][0]) % self.angle_shift).reshape(1)
        
        self.F_steographic_inv = lambda x: (x[1][0]/(1-x[1][1])).reshape(1)
        self.F_steographic = lambda x: (jnp.array([2*x[0], x[0]**2-1])/(x[0]**2+1)).reshape(2)
        
        do_chart_update_spherical = lambda x: x[0] > self.angle_shift-.1 
        do_chart_update_steographic = lambda x: x[0] > .1 # look for a new chart if true
        
        if use_spherical_coords:
            F = self.F_spherical
            invF = self.F_spherical_inv
            self.do_chart_update = do_chart_update_spherical
            self.centered_chart = self.centered_chart_spherical
            self.chart = lambda : jnp.array([1.0, 0.0])
        else:
            F = self.F_steographic
            invF = self.F_steographic_inv
            self.do_chart_update = do_chart_update_steographic
            self.centered_chart = self.centered_chart_steographic
            self.chart = lambda : jnp.array([0.0, -1.0])

        riemannian.EmbeddedManifold.__init__(self,F,1,2,invF=invF)

        # action of matrix group on elements
        self.act = lambda g,x: jnp.tensordot(g,x,(1,0))
        self.acts = lambda g,x: jnp.tensordot(g,x,(2,0))
        
        riemannian.metric(self)
        riemannian.curvature(self)
        riemannian.geodesic(self)
        riemannian.Log(self)
        riemannian.parallel_transport(self)
        
        if use_spherical_coords:
            self.Log = self.StdLogSpherical
        else:
            self.Log = self.StdLogSteographic

        #Heat kernels
        self.hk = jit(lambda x,y,t: hk(self, x,y, t))
        self.log_hk = jit(lambda x,y,t: log_hk(self, x, y, t))
        self.gradx_log_hk = jit(lambda x,y,t: gradx_log_hk(self, x, y, t))
        self.grady_log_hk = jit(lambda x,y,t: grady_log_hk(self, x, y, t))
        #self.ggrady_log_hk = jit(lambda x,y,t: -jnp.eye(self.dim)/t)
        self.gradt_log_hk = lambda x,y,t: gradt_log_hk(self, x, y, t)
    
    def centered_chart_spherical(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return stop_gradient(self.F(x))
        else:
            return x % self.angle_shift # already in embedding space

    def centered_chart_steographic(self,x):
        """ return centered coordinate chart """
        if type(x) == type(()): # coordinate tuple
            return stop_gradient(self.F(x))
        else:
            return x # already in embedding space

    def StdLogSpherical(self, x,y):

        return (y[0]-x[0])%self.angle_shift
    
    def StdLogEmb(self, x,y):
        proj = lambda x,y: jnp.dot(x,y)*x
        Fx = self.F(x)
        v = y-proj(Fx,y)
        theta = jnp.arccos(jnp.dot(Fx,y))
        normv = jnp.linalg.norm(v,2)
        w = cond(normv >= 1e-5,
                         lambda _: theta/normv*v,
                         lambda _: jnp.zeros_like(v),
                         None)
    
    def StdLogSteographic(self, x,y):
        Fx = self.F(x)
        return jnp.dot(self.invJF((Fx,x[1])),self.StdLogEmb(x,y))

#%% Heat Kernel

def hk(M:object, x:jnp.ndarray,y:jnp.ndarray,t:float,N_terms=20)->float:
    
    def step(carry:float, k:int)->Tuple[float, None]:
        
        carry += jnp.exp(-0.5*(2*jnp.pi*k+x1-y1)**2/t)
        
        return carry, None
    
    x1 = x[0]#jnp.arctan2(x[1][1],x[1][0]) % (2*jnp.pi)
    y1 = y[0]#jnp.arctan2(y[1][1],y[1][0]) % (2*jnp.pi)
    
    const = 1/jnp.sqrt(2*jnp.pi*t)
   
    val, _ = scan(step, init=jnp.zeros(1), xs=jnp.arange(-N_terms+1,N_terms,1)) 
   
    return val*const

def log_hk(M:object, x:jnp.ndarray,y:jnp.ndarray,t:float)->float:
    
    return jnp.log(hk(x,y,t))

def gradx_log_hk(M:object, x:jnp.ndarray,y:jnp.ndarray,t:float, N_terms=20)->Tuple[jnp.ndarray, jnp.ndarray]:
    
    def get_coords(Fx:jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        chart = M.centered_chart(Fx)
        return (M.invF((Fx,chart)),chart)

    def to_TMchart(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:
        
        x = invF_spherical(Fx)
        invJFx = JinvF_spherical((x,Fx))
        
        return jnp.dot(invJFx.reshape(-1,1),v)

    def to_TMx(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:

        x = get_coords(Fx)

        return jnp.dot(M.invJF((Fx,x[1])),v)
    
    def step(carry:float, k:int)->Tuple[float, None]:
        
        term1 = 2*jnp.pi*k+x1-y1
        
        carry -= jnp.exp(-0.5*(term1**2)*tinv)*term1*tinv
        
        return carry, None
    
    F_spherical = lambda x: jnp.array([jnp.cos(x[0]), jnp.sin(x[0])])
    invF_spherical = lambda x: jnp.arctan2(x[1][1]/x[1][0]) % (2*jnp.pi)
    Jf_spherical = jacfwdx(F_spherical)

    const = 1/jnp.sqrt(2*jnp.pi*t)
        
    x1 = x[0]#jnp.arctan2(x[1][1]/x[1][0]) % (2*jnp.pi)
    y1 = y[0]#jnp.arctan2(y[1][1]/y[1][0]) % (2*jnp.pi)
    tinv = 1/t
   
    val, _ = scan(step, init=jnp.zeros(1), xs=jnp.arange(0,N_terms,1)) 
    grad = val*const/hk(M,x,y,t)
    
    grad_chart = to_TMchart(x, grad)
    grad_x = to_TMx(x[1], grad_chart)
   
    return grad#grad_x, grad_chart

def grady_log_hk(M:object, x:jnp.ndarray, y:jnp.ndarray, t:float, N_terms=20) -> Tuple[jnp.ndarray, jnp.ndarray]:
    
    def get_coords(Fx:jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

        chart = M.centered_chart(Fx)
        return (M.invF((Fx,chart)),chart)

    def to_TMchart(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:
        
        x = invF_spherical(Fx)
        invJFx = JinvF_spherical((x,Fx))
        
        return jnp.dot(invJFx.reshape(-1,1),v)

    def to_TMx(Fx:jnp.ndarray,v:jnp.ndarray) -> jnp.ndarray:

        x = get_coords(Fx)

        return jnp.dot(M.invJF((Fx,x[1])),v)
    
    def step(carry:float, k:int)->Tuple[float,None]:
        
        term1 = 2*jnp.pi*k+x1-y1
        
        carry += jnp.exp(-0.5*(term1**2)*tinv)*term1*tinv
        
        return carry, None
    
    F_spherical = lambda x: jnp.array([jnp.cos(x[0]), jnp.sin(x[0])])
    invF_spherical = lambda x: jnp.arctan2(x[1][1],x[1][0]) % (2*jnp.pi)
    Jf_spherical = jacfwdx(F_spherical)
    JinvF_spherical = jacfwdx(invF_spherical)

    const = 1/jnp.sqrt(2*jnp.pi*t)
    
    x1 = x[0]#jnp.arctan2(x[1][1],x[1][0]) % jnp.pi
    y1 = y[0]#jnp.arctan2(y[1][1],y[1][0]) % jnp.pi
    tinv = 1/t
    
    val, _ = scan(step, init=jnp.zeros(1), xs=jnp.arange(-N_terms+1,N_terms,1)) 
    grad = val*const/hk(M,x,y,t)
   
    grad_chart = to_TMchart(y, grad)
    grad_x = to_TMx(y[1], grad_chart)
   
    return grad#grad_x, grad_chart

def gradt_log_hk(M:object, x:jnp.ndarray, y:jnp.ndarray, t:float, N_terms=20)->float:
    
    def step1(carry:float, k:int)->Tuple[float,None]:
        
        term1 = 0.5*(2*jnp.pi*k+x1-y1)**2
        
        carry += jnp.exp(-term1/t)*term1/(t**2)
        
        return carry, None
    
    def step2(carry:float, k:int)->Tuple[float,None]:
        
        carry += jnp.exp(-(0.5*(2*jnp.pi*k+x1-y1)**2)/t)
        
        return carry, None
    
    x1 = x[0]#jnp.arctan2(x[1][1],x[1][0]) % (2*jnp.pi)
    y1 = y[0]#jnp.arctan2(y[1][1],y[1][0]) % (2*jnp.pi)
        
    const1 = 1/jnp.sqrt(2*jnp.pi*t)
    const2 = -1/(2*jnp.sqrt(jnp.pi)*(t)**(3/2))
   
    val1, _ = scan(step1, init=jnp.zeros(1), xs=jnp.arange(-N_terms+1,N_terms,1)) 
    val1 *= const1
    
    val2, _ = scan(step2, init=jnp.zeros(1), xs=jnp.arange(-N_terms+1,N_terms,1)) 
    val2 *= const2
   
    return (val1+val2)/hk(M,x,y,t)