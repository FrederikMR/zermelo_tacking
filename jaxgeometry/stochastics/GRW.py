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

#%% Geodesic Random Walk

def initialize(M:object,
               b_fun:Callable[[ndarray, ndarray], ndarray] = None,
               sigma_fun:Callable[[ndarray, ndarray], ndarray] = None
               )->None:
    
    def GRW(x:Tuple[ndarray, ndarray],
            dt:ndarray,
            dW:ndarray,
            )->Tuple[ndarray, ndarray, ndarray]:
        
        def walk(c:Tuple[ndarray, ndarray, ndarray],
                 y:Tuple[ndarray, ndarray]
                 )->Tuple[ndarray, ndarray, ndarray, float]:
            
            x,chart = c
            t,dt,dW = y
            
            v = b_fun(t,(x,chart))*dt+jnp.tensordot(sigma_fun(t,(x,chart)),dW,(1,0))
            
            x_new = M.Exp((x,chart), v, T=1.0)
            
            out = x_new
            
            return out, out
            
        t = jnp.cumsum(dt)
        
        return (t, *scan(walk, init=x, xs=(t,dt, dW))[1])
    
    def product_GRW(x:Tuple[ndarray, ndarray],
            dt:ndarray,
            dW:ndarray,
            )->Tuple[ndarray, ndarray, ndarray]:
        
        def walk(c:Tuple[ndarray, ndarray, ndarray],
                 y:Tuple[ndarray, ndarray]
                 )->Tuple[ndarray, ndarray, ndarray, float]:
            
            x,chart = c
            t,dt,dW = y
            
            (xs, charts) = vmap(lambda x,chart, dW: \
                                M.Exp((x,chart), b_fun(t,(x,chart))*dt+jnp.tensordot(sigma_fun(t,(x,chart)),dW,(1,0)), T=1.0)
                                )(x,chart,dW)
            
            return (xs,charts), (xs,charts)
            
        t = jnp.cumsum(dt)
        
        (xs, chartss) = scan(walk, init=(x[0], x[1]), xs=(t,dt, dW))[1]
        
        return (t, xs, chartss)
    
    if b_fun is None:
        b_fun = lambda t,x: jnp.zeros(M.dim)
    if sigma_fun is None:
        sigma_fun = lambda t,x: jnp.eye(M.dim)
                    
    M.GRW = jit(lambda x, t, dt, dW: GRW((x,chart), dt, dW))
    M.product_GRW = jit(lambda x, dt, dW: product_GRW((x[0],x[1]), dt, dW))

    
    return