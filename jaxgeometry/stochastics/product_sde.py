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

#%% Product Diffusion Process

def initialize(M:object,
               sde,
               chart_update:Callable,
               integrator:Callable=integrator_ito):
    """ product diffusions """

    def sde_product(c,
                    y
                    ):
        t,x,chart,*cy = c
        dt,dW = y
        
        (det,sto,X,*dcy) = vmap(lambda x,chart,dW,*_cy: sde((t,x,chart,*_cy),(dt,dW)),0)(x,chart,dW,*cy)

        return (det,sto,X,*dcy)

    chart_update_product = vmap(chart_update)

    product = jit(lambda x,dts,dWs,*cy: integrate_sde(sde_product,integrator,chart_update_product,x[0],x[1],dts,dWs,*cy))

    return (product,sde_product,chart_update_product)

# for initializing parameters
def tile(x:ndarray,N:int):
    
    try:
        return jnp.tile(x,(N,)+(1,)*x.ndim)
    except AttributeError:
        try:
            return jnp.tile(x,N)
        except TypeError:
            return tuple([tile(y,N) for y in x])