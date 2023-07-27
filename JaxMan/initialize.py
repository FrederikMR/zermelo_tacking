#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 09:58:24 2023

@author: fmry
"""

#%% Sources

#%% Loading Modules

#JAX
from jax import jacfwd, jit, grad, vmap, lax
from jax.scipy.optimize import minimize
import jax.numpy as jnp

#functools
from functools import partial

#typing
from typing import Callable