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

#%% MPP Landmarks Log

###############################################################
# Most probable paths for landmarks via development - BVP     # 
###############################################################
def initialize(M:object,
               method:str='BFGS'
               )->None:

    def loss(x:ndarray,
             lambd:ndarray,
             y:ndarray,
             qps:ndarray,
             _dts:ndarray
             )->ndarray:
        
        (_,xs,_,charts) = M.MPP_landmarks(x,lambd,qps,_dts)
        (x1,chart1) = (xs[-1],charts[-1])
        y_chart1 = M.update_coords(y,chart1)
        
        return 1./M.dim*jnp.sum(jnp.square(x1 - y_chart1[0]))

    def shoot(x:ndarray,
              y:ndarray,
              qps:ndarray,
              _dts:ndarray,
              lambd0:ndarray=None
              )->Tuple[ndarray, ndarray]:        

        if lambd0 is None:
            lambd0 = jnp.zeros(M.dim)

        res = minimize(jax.value_and_grad(lambda w: loss(x,w,y,qps,_dts)), lambd0, method=method, jac=True, options={'disp': False, 'maxiter': 100})

        return (res.x,res.fun)

    M.Log_MPP_landmarks = shoot
    
    return