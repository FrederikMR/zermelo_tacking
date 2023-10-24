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

#%% Code

class Manifold(object):
    """ Base Riemannian manifold class """
    
    def __init__(self)->None:
        self.dim = None        
        if not hasattr(self, 'do_chart_update'):
            self.do_chart_update = None # set to relevant function if what updates are desired
            
        return
            
    def __str__(self)->str:
        return "abstract Riemannian manifold"
            
    def chart(self)->ndarray:
        """ return default or specified coordinate chart. This method will generally be overriding by inheriting classes """
        # default value 
        return jnp.zeros(1) 
    
    def centered_chart(self,coords):
        """ return centered coordinate chart. Must be implemented by inheriting classes 
        Generally wish to stop gradient computations through the chart choice
        """
        return stop_gradient(jnp.zeros(1))
    
    def coords(self,coords:ndarray=None,chart:ndarray=None)->Tuple[ndarray, ndarray]:
        """ return coordinate representation of point in manifold """
        if coords is None:
            coords = jnp.zeros(self.dim)
        if chart is None:
            chart = self.chart()

        return (jnp.array(coords),chart)
    
    def update_coords(self,coords:ndarray,new_chart:ndarray)->ndarray:
        """ change between charts """
        
        assert(False) #not implemented here
    
    def update_vector(self,coords:ndarray,new_coords:ndarray,new_chart:ndarray,v:ndarray)->ndarray:
        """ change tangent vector between charts """
        
        assert(False) # not implemented here
        
    def update_covector(self,coords:ndarray,new_coords:ndarray,new_chart:ndarray,p:ndarray)->ndarray:
        """ change cotangent vector between charts """
        assert(False) # not implemented here
        
    def newfig(self)->None:
        """ open new plot for manifold """
        
        return
    
class EmbeddedManifold(Manifold):
    """ Embedded Riemannian manifold in Euclidean Space base class """
    
    def __init__(self,F:Callable[[ndarray], ndarray]=None,
                 dim:int=None,
                 emb_dim:int=None,
                 invF:Callable[[ndarray], ndarray]=None)->None:
        Manifold.__init__(self)
        self.dim = dim
        self.emb_dim = emb_dim

        # embedding map and its inverse
        if F is not None:
            self.F = F
            self.invF = invF
            self.JF = jacfwdx(self.F)
            self.invJF = jacfwdx(self.invF)
            
            @jit
            def g(x:Tuple[ndarray, ndarray])->ndarray:
                
                JF = self.JF(x)
                
                return jnp.tensordot(JF,JF,(0,0))

            # metric matrix
            self.g = g
            
        return
    
    def __str__(self)->str:
        return "Riemannian manifold of dimension %d embedded in R^%d" % (self.dim,self.emb_dim)
    
    def update_coords(self,coords:ndarray,new_chart:ndarray)->Tuple[ndarray, ndarray]:
        """ change between charts """
        return (self.invF((self.F(coords),new_chart)),new_chart)

    def update_vector(self,coords:ndarray,new_coords:ndarray,new_chart:ndarray,v:ndarray)->ndarray:
        """ change tangent vector between charts """
        return jnp.tensordot(self.invJF((self.F((new_coords,new_chart)),new_chart)),jnp.tensordot(self.JF(coords),v,(1,0)),(1,0))

    def update_covector(self,coords:ndarray,new_coords:ndarray,new_chart:ndarray,p:ndarray)->ndarray:
        """ change cotangent vector between charts """
        return jnp.tensordot(self.JF((new_coords,new_chart)).T,jnp.tensordot(self.invJF((self.F(coords),coords[1])).T,p,(1,0)),(1,0))
    
    def plot_path(self, xs:Tuple[ndarray, ndarray], 
                  vs:ndarray=None, 
                  v_steps:int=None, 
                  i0:int=0, 
                  color:str='b', 
                  color_intensity:float=1., 
                  linewidth:float=1., 
                  s:int=15.,
                  prevx:Tuple[ndarray, ndarray]=None, 
                  last:bool=True) -> None:
    
        if vs is not None and v_steps is not None:
            v_steps = np.arange(0,n_steps)
    
        xs = list(xs)
        N = len(xs)
        prevx = None
        for i,x in enumerate(xs):
            xx = x[0] if type(x) is tuple else x
            if xx.shape[0] > self.dim and (self.emb_dim == None or xx.shape[0] != self.emb_dim): # attached vectors to display
                v = xx[self.dim:].reshape((self.dim,-1))
                x = (xx[0:self.dim],x[1]) if type(x) is tuple else xx[0:self.dim]
            elif vs is not None:
                v = vs[i]
            else:
                v = None
            self.plotx(x, v=v,
                       v_steps=v_steps,i=i,
                       color=color,
                       color_intensity=color_intensity if i==0 or i==N-1 else .7,
                       linewidth=linewidth,
                       s=s,
                       prevx=prevx,
                       last=i==(N-1))
            prevx = x 
            
        return

    # plot x. x can be either in coordinates or in R^3
    def plotx(self, x:Tuple[ndarray, ndarray], 
              u:ndarray=None, 
              v:ndarray=None, 
              v_steps:int=None, 
              i:int=0, 
              color:str='b',               
              color_intensity:float=1., 
              linewidth:float=1., 
              s:float=15., 
              prevx:Tuple[ndarray, ndarray]=None, 
              last:bool=True)->None:
    
        assert(type(x) == type(()) or x.shape[0] == self.emb_dim)
    
        if v is not None and v_steps is None:
            v_steps = np.arange(0,n_steps)        
    
        if type(x) == type(()): # map to manifold
            Fx = self.F(x)
            chart = x[1]
        else: # get coordinates
            Fx = x
            chart = self.centered_chart(Fx)
            x = (self.invF((Fx,chart)),chart)

        if prevx is not None:
            if type(prevx) == type(()): # map to manifold
                Fprevx = self.F(prevx)
            else:
                Fprevx = prevx
                prevx = (self.invF((Fprevx,chart)),chart)
    
        ax = plt.gca()
        if prevx is None or last:
            ax.scatter(Fx[0],Fx[1],Fx[2],color=color,s=s)
        if prevx is not None:
            xx = np.stack((Fprevx,Fx))
            ax.plot(xx[:,0],xx[:,1],xx[:,2],linewidth=linewidth,color=color)
    
        if u is not None:
            Fu = np.dot(self.JF(x), u)
            ax.quiver(Fx[0], Fx[1], Fx[2], Fu[0], Fu[1], Fu[2],
                      pivot='tail',
                      arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                      color='black')
    
        if v is not None:
            if i in v_steps:
                if not v.shape[0] == self.emb_dim:
                    v = np.dot(self.JF(x), v)
                ax.quiver(Fx[0], Fx[1], Fx[2], v[0], v[1], v[2],
                          pivot='tail',
                          arrow_length_ratio = 0.15, linewidths=linewidth, length=0.5,
                        color='black')
        
        return
    