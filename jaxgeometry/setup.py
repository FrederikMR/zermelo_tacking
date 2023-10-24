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

#Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#JAX
from jax.numpy import ndarray
import jax.numpy as jnp
from jax.lax import stop_gradient, scan, cond, linalg
from jax import vmap, grad, jacfwd, jacrev, random, jit, value_and_grad, Array, device_get, \
    tree_leaves, tree_map, tree_flatten, tree_unflatten

#JAX Optimization
from jax.example_libraries import optimizers

#JAX scipy
import jax.scipy as jscipy

#Scipy
from scipy.linalg import fractional_matrix_power, logm

#haiku
import haiku as hk

#optax
import optax

#tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

#numpy
import numpy as np

#scipy
from scipy.optimize import minimize,fmin_bfgs,fmin_cg, approx_fprime

#sklearn
from sklearn.decomposition import PCA

#os
import os

#pickle
import pickle

#Plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#functools
from functools import partial

#typing
from typing import Callable, NamedTuple, Tuple

#JAXGeometry
from jaxgeometry.params import *
from jaxgeometry.autodiff import *
from jaxgeometry.integration import *
from jaxgeometry.vector_operations import *


