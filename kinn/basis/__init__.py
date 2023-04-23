# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__author__ = {'Gabriel S. Gusmao' : 'gusmaogabriels@gmail.com'}
__version__ = '1.0'

import jax.numpy as jnp
from numpy.random import choice
import numpy as onp
from jax import grad, jit, vmap, pmap, jacobian, jacfwd, jacrev, hessian, random
import jaxlib
from functools import partial
from jax.lax import Precision
from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers
from jax.config import config
from jax.tree_util import tree_map
config.update("jax_debug_nans", True)
config.update('jax_enable_x64', True)
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt
import itertools
from matplotlib import animation, cm
from IPython.display import HTML
from IPython.display import display, Image
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
from scipy.interpolate import griddata, interp1d, BSpline, CubicSpline
from scipy.integrate import solve_ivp, quadrature, romberg, simps
plt.style.use('seaborn-white')

from .model import *
from .nnx import *
from .mle import *
#from .mle_noncentered import *
from .mle_pfr import *

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

left  = 0.05  # the left side of the subplots of the figure
right = 0.925    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.85      # the top of the subplots of the figure
wspace = 0.225   # the amount of width reserved for blank space between subplots
hspace = 0.25   # the amount of height reserved for white space between subplots
