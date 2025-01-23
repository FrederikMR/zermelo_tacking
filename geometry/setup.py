#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:46:44 2024

@author: fmry
"""

#%% Modules

from jax import Array
from jax import vmap, grad, jacfwd, jacrev, value_and_grad
from jax import lax

import jax.numpy as jnp

import jax 
jax.config.update("jax_enable_x64", True)

#jac scipy
import jax.scipy as jscipy
from jax.scipy.optimize import minimize as jminimize

#JAX Optimization
from jax.example_libraries import optimizers

#scipy
from scipy.optimize import minimize

from abc import ABC
from typing import Callable, Tuple, Dict, List