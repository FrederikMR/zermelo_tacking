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

#%% Logaritmic Map

def initialize(M:object,
               f=None,
               method='BFGS'
               )->None:
    """ numerical Riemannian Logarithm map """

    def loss(x:Tuple[ndarray, ndarray],
             v:ndarray,
             y:Tuple[ndarray, ndarray]
             )->float:
        
        (x1,chart1) = f(x,v)
        y_chart1 = M.update_coords(y,chart1)
        
        return 1./M.dim*jnp.sum(jnp.square(x1 - y_chart1[0]))
    
    def shoot(x:Tuple[ndarray, ndarray],
              y:Tuple[ndarray, ndarray],
              v0:ndarray=None
              )->Tuple[ndarray, ndarray]:

        if v0 is None:
            v0 = jnp.zeros(M.dim)

        res = minimize(lambda w: (loss(x,w,y),dloss(x,w,y)), v0, method=method, jac=True, options={'disp': False, 'maxiter': 100})

        return (res.x,res.fun)
    
    def dist(x:Tuple[ndarray, ndarray],
             y:Tuple[ndarray, ndarray]
             )->float:
        
        v = M.Log(x,y)
        
        curve = M.geodesic(x,v[0],dts(T,n_steps))
        
        dt = jnp.diff(curve[0])
        val = vmap(lambda v: M.norm(x,v))(curve[1][:,1])
        
        return jnp.sum(0.5*dt*(val[1:]+val[:-1])) #trapezoidal rule

    if f is None:
        print("using M.Exp for Logarithm")
        f = M.Exp

    dloss = grad(loss,1)

    M.Log = shoot
    M.dist = dist
    
    return
