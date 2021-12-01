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

from . import random, jnp, jit, vmap, jacfwd, pmap, Precision


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


# Derivatives of state variables with respect to ijnputs using autograd `jacfwd` for tall Jacobians. (JIT precompiled)

class nn(object):
    
    def __init__(self, layers_sizes, act_fun, nn_scale=0.01, init=True, usepmap=False):
        self.layers_sizes = layers_sizes
        self.usepmap = usepmap
        if init: self.init(nn_scale, act_fun)
        
    def _rejit(self):
        if self.usepmap:
            self.batched_state = jit(pmap(self.state, in_axes=(None,0)))
        else:
            self.batched_state = jit(vmap(self.state, in_axes=(None,0)))
        self.diff_state    = jit(lambda params,t:vmap(jacfwd(self.state,argnums=(1)),in_axes=(None,0))(params,t))
        self.diff_state2   = jit(lambda params,t:vmap(jacfwd(jacfwd(self.state,argnums=(1)),argnums=(1)),in_axes=(None,0))(params,t))
    
    def init(self, nn_scale, act_fun):
        self.nn_scale     = nn_scale
        self.act_fun      = act_fun
        self._init_params()
        self._rejit()
        
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
