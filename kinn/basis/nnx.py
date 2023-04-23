#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Fri Aug 02 17:45:50 2019
#
# @author:gusmaogabriels@gmail.com // gusmaogabriels@gatech.edu
#
# """

# ## Neural Network Class
#
# Generic Fully Connected using JAX.

# -*- coding: utf-8 -*-

# #### Necessary Libraries

# + {"active": ""}
# from __future__ import print_function, division, absolute_import
# import jax.numpy as np
# from numpy.random import choice
# import numpy as onp
# from jax import grad, jit, vmap, jacobian, jacfwd, jacrev
# from jax import random
# from jax.scipy.special import logsumexp
# from jax.experimental import optimizers
# from jax.config import config
# from jax.tree_util import tree_map
# config.update("jax_debug_nans", True)
# config.update('jax_enable_x64', True)
# import time
# from IPython.display import clear_output
# from matplotlib import pyplot as plt
# import itertools
# from matplotlib import animation
# from IPython.display import HTML
# from IPython.display import display, Image
# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.ticker import MaxNLocator
# plt.style.use('seaborn-white')
# -

from . import random, jnp, jit, vmap, jacfwd, pmap, Precision
from jax.nn import sigmoid


# #### Set of helper functions based on [JAX tutorials](https://colab.research.google.com/github/google/jax/blob/master/notebooks/neural_network_with_tfds_data.ipynb) ***(Google Colabs)***

# *Dense Neural Network: random initialization of layers weights and biases.*

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Network and model initialization functions

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k, scale) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# Universal activation function to be used. (JIT precompiled)

@jit
def act_fun(x):
    #return jnp.maximum(0, x)
    #return jnp.nan_to_num(x / (1.0 + jnp.exp(-x)))
    #return x / (1.0 + jnp.exp(-x))
    return jnp.tanh(x)
    #return 0.5*jnp.tanh(x) + 0.5*x / (1.0 + jnp.exp(-x))
    #return(x)

# Fully-connected NN evaluator for *state* variables. (JIT precompiled)

@jit
def state(params, t):
    # per-example state
    activations = t
    for i, (w, b) in enumerate(params[:-1]):
        outputs = jnp.dot(w, activations) + b
        activations = act_fun(outputs)
    
    final_w, final_b = params[-1]
    y = (jnp.dot(final_w, activations) + final_b)
    #y = y / y.sum()
    return y


@jit
def normtrig(x):
    n = len(y)
    y = jnp.sin(x)**2.
    z = jnp.prod((jnp.ones([n]*2)-jnp.triu(jnp.ones([n]*2))).dot(jnp.diag(y))-jnp.diag(2.-y)+jnp.triu(jnp.ones([n]*2)),axis=1)
    return jnp.concatenate((z,jnp.prod(y)))    


# Derivatives of state variables with respect to inputs using autograd `jacfwd` for tall Jacobians. (JIT precompiled)

class nn_(object):
    
    def __init__(self, layers_sizes, act_fun, nn_scale=0.01, init=True, pmap=False):
        self.layers_sizes = layers_sizes
        if init: self.init(nn_scale, act_fun)
        self._jit_compile()
            
    def _jit_compile(self):
        if pmap :
            self.batched_state = jit(pmap(self.state, in_axes=(None,0)))
        else:
            self.batched_state = jit(vmap(self.state, in_axes=(None,0)))
        #self.diff_state    = jit(lambda params,t:self._diff_state(self.batched_state,params,t,order,argnum=None))
        self.diff_state    = self._diff_state(self.batched_state,1)
    
    def init(self, nn_scale, act_fun):
        self.nn_scale     = nn_scale
        self.act_fun      = act_fun
        self.__randkey__ = random.PRNGKey(0)
        self.params = [init_network_params(layer_sizes, self.__randkey__, self.nn_scale) for\
                       layer_sizes in self.layers_sizes]# initialize parameters 
    
    def _fun(self,x,t,argnum):
        i = jnp.arange(len(t))
        x[i,:,i,:argnum]
        
    def _diff_state(self,fun,order,argnum=None): # ,params,t
        #return (jacobian(batched_state,argnums=1)(params,t)[i,:,i,0])
        if order > 0:
            return self._diff_state(jacfwd(fun),order-1,argnum)
        else:
            return jit(lambda params, t: jnp.nan_to_num(self._fun(jacfwd(lambda t : fun(params,t))(t)),t,argnum))
    
    def _state(self, params, t):
        # per-example state
        activations = t
        for i, (w, b) in enumerate(params[:-1]):
            outputs = jnp.dot(w, activations) + b
            activations = self.act_fun[i](outputs)

        final_w, final_b = params[-1]
        y = (jnp.dot(final_w, activations) + final_b)
        #y = y / y.sum()
        return self.constraints(y,t)
    
    def state(self, params, t):
        return self._state(params,t)
    
    def constraints(self,x,t):
        return x
                
    #def diff_state(self,nn_params,t):
    #    i = jnp.arange(len(t))
    #    return jnp.nan_to_num(jacfwd(lambda t : self.batched_state(nn_params,t))(t)[i,:,i,0])
    
    def __call__(self,t,nn):
        # nn used to train multiple nn's simultaneously
        return self.batched_state(self.params[nn],t)


class nn(object):
    
    def __init__(self, layers_sizes, act_fun, nn_scale=0.01, init=True, usepmap=False):
        self.layers_sizes = layers_sizes
        self.usepmap = usepmap
        if init: self.init(nn_scale, act_fun)
        
    def _jit_compile(self):
        if self.usepmap:
            self.batched_state = jit(pmap(self.state, in_axes=(None,0)))
        else:
            self.batched_state = jit(vmap(self.state, in_axes=(None,0)))
        self.diff_state    = jit(lambda params,t:vmap(jacfwd(self.state,argnums=(1)),in_axes=(None,0))(params,t))
        self.d_pars     = jit(lambda params,t:vmap(jacfwd(self.state,argnums=(2)),in_axes=(None,0))(params,t))
        self.d_state_pars     = jit(lambda params,t:vmap(jacfwd(jacfwd(self.state,argnums=(1)),argnums=(2)),in_axes=(None,0))(params,t))
        self.diff_state2   = jit(lambda params,t:vmap(jacfwd(jacfwd(self.state,argnums=(1)),argnums=(1)),in_axes=(None,0))(params,t))
    
    def init(self, nn_scale, act_fun):
        self.nn_scale     = nn_scale
        self.act_fun      = act_fun
        self._init_params()
        self._jit_compile()
        
    def _init_params(self):
        self.__randkey__ = random.PRNGKey(0)
        self.set_params([init_network_params(layer_sizes, self.__randkey__, self.nn_scale) for\
                       layer_sizes in self.layers_sizes])
    
    def set_params(self,params):
        self.params = params
        
    def _diff_state(self,batched_state,params,t):
        i = jnp.arange(len(t))
        #return (jacobian(batched_state,argnums=1)(params,t)[i,:,i,0])
        return jnp.nan_to_num(jacfwd(lambda t : batched_state(params,t))(t)[i,:,i,:])
    
    def _state(self, params, t):
        # per-example state
        activations = t
        for i, (w, b) in enumerate(params[:-1]):
            outputs = jnp.dot(w, activations,precision=Precision.HIGHEST) + b
            activations = self.act_fun[i](outputs)

        final_w, final_b = params[-1]
        y = (jnp.dot(final_w, activations,precision=Precision.HIGHEST) + final_b)
        #y = y / y.sum()
        return y
    
    def state(self, params, t):
        return self.constraints(self._state(params,t),t)
    
    def constraints(self,x,t):
        return x
                
    #def diff_state(self,nn_params,t):
    #    i = jnp.arange(len(t))
    #    return jnp.nan_to_num(jacfwd(lambda t : self.batched_state(nn_params,t))(t)[i,:,i,0])
    
    def __call__(self,t,nn):
        # nn used to train multiple nn's simultaneously
        return self.batched_state(self.params[nn],t)

class nn_bvp(nn):

    # expected normalized independent variable t \in [0,1]
    def set_boundaries(self,x0,xf):
        self.x0 = x0
        self.xf = xf

    @jit
    def state(self,params, t):
        # per-example stateions
        activations = t#*fac
        for i,(w, b) in enumerate(params[0][:-1]):
            outputs = jnp.dot(w, activations,precision=Precision.HIGHEST) + b
            activations = self.act_fun[i](outputs)
        
        final_w, final_b = params[0][-1]
        z = (jnp.dot(final_w, activations,precision=Precision.HIGHEST) + final_b)
        
        t_ = t#jnp.sin(t*jnp.pi/2)
        kactivations = t#*fac
        for i,(w, b) in enumerate(params[1][:-1]):
            koutputs = jnp.dot(w, kactivations,precision=Precision.HIGHEST) + b
            kactivations = self.act_fun[i](koutputs)
        
        final_w, final_b = params[1][-1]
        k = jnp.exp(jnp.dot(final_w, kactivations,precision=Precision.HIGHEST) + final_b)
        
        alpha = ((jnp.tanh(k*t)-jnp.tanh(k*(t-1)))+jnp.tanh(-k)).mean()
        
        kactivations_ = t#*fac
        for i,(w, b) in enumerate(params[2][:-1]):
            koutputs = jnp.dot(w, kactivations_,precision=Precision.HIGHEST) + b
            kactivations_ = self.act_fun[i](koutputs)
        
        final_w, final_b = params[2][-1]
        #k_ = jnp.exp(jnp.dot(final_w, kactivations_,precision=Precision.HIGHEST) + final_b)
        k_ = jnp.dot(final_w, kactivations_,precision=Precision.HIGHEST) + final_b
        
        beta = (1.-jnp.exp(-k_*t))/(1.-jnp.exp(-k_))#((jnp.tanh(k_*t))).mean()#beta = ((jnp.tanh(k_*t)-jnp.tanh(k_*(t-1)))+jnp.tanh(-k_)).mean()
        #alpha = t*(1.-t)
        
        n = len(z+1)
        l = [1-1./(n+1)]
        for i in range(n-1):
            l += [2-1./l[-1]]
        offset=jnp.array(l)
        offset=jnp.log(offset/(1-offset))
        
    #     k = jnp.exp(params[1])
    #     k_ = jnp.exp(params[1])
    #     alpha = (jnp.tanh(k*t)-jnp.tanh(k*(t-1)))+jnp.tanh(-k)
    #     beta  = (jnp.tanh(k_*t)-jnp.tanh(k_*(t-1)))+jnp.tanh(-k_)
        #xs = normtrig(z,t,offset)#
        xs = ((normtrig(z,t,offset))*alpha)+(1-alpha)*((self.xf*beta+self.x0*(1-beta)))
        return xs

class nn_positive(nn):

    @jit
    def state_csi(self,params,t):
        # per-example stateions
        activations = t#*fac
        activations0 = t*0
        for i,(w,b) in enumerate(params[0][:-1]):
            outputs = jnp.dot(jnp.exp(w), activations,precision=Precision.HIGHEST)+sigmoid(b)
            outputs0 = jnp.dot(jnp.exp(w), activations0,precision=Precision.HIGHEST)+sigmoid(b)
            activations = self.act_fun[i](outputs)
            activations0 = self.act_fun[i](outputs0)
        
        final_w,b = params[0][-1]
        dz  = (jnp.dot(jnp.exp(final_w), activations,precision=Precision.HIGHEST))+sigmoid(b)
        dz0 = (jnp.dot(jnp.exp(final_w), activations0,precision=Precision.HIGHEST))+sigmoid(b)
        csi = dz-dz0
        return csi