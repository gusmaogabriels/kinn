"""kinn.basis -- Kinetics-Informed Neural Networks.

Core:
  model  -- stoichiometry, SVD decomposition, rate law (M @ r), PSSH
  nnx    -- neural network (MLP, structured nn_npt with SVD projection)
  mle    -- MLE optimizer with OAS covariance reweighting

kinn provides the trajectory parameterization in SVD coordinates and the
mass-action physics residual (dx/dt - M @ r). Plotting is optional.
"""
__author__ = {'Gabriel S. Gusmao': 'gusmaogabriels@gmail.com'}
__version__ = '1.0'

# JAX core (required by submodules via `from . import jnp, jit, ...`)
import jax
import jax.numpy as jnp
from numpy.random import choice
import numpy as onp
np = onp
from jax import grad, jit, vmap, pmap, jacobian, jacfwd, jacrev, hessian, random
from functools import partial
from jax.lax import Precision
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

jax.config.update('jax_enable_x64', True)
config = jax.config

try:
    from jax.example_libraries import optimizers
except ImportError:
    optimizers = None

# Stdlib + scipy (used by mle.py internals)
import time
import itertools
from scipy.interpolate import interp1d
from scipy.interpolate import griddata, BSpline, CubicSpline
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

try:
    from IPython.display import clear_output
except ImportError:
    clear_output = lambda wait=False: None

try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps

# Sub-modules
from .model import model, pssh
from .nnx import nn
from .mle import nn_npt, opt, cov_oas


def __getattr__(name):
    # Keep the original notebook imports available without importing plotting
    # libraries in local optimization or command-line processes.
    if name in {'plt', 'animation', 'cm', 'HTML', 'display', 'Image',
                'FormatStrFormatter', 'MaxNLocator', 'make_axes_locatable',
                'SMALL_SIZE', 'MEDIUM_SIZE', 'BIGGER_SIZE',
                'left', 'right', 'bottom', 'top', 'wspace', 'hspace'}:
        from . import plot_setup
        return getattr(plot_setup, name)
    raise AttributeError(name)
