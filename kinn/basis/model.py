#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Fri Aug 02 17:45:50 2019
#
# @author:gusmaogabriels@gmail.com // gusmaogabriels@gatech.edu
#
# """

# ## Kinetic Model Class

# -*- coding: utf-8 -*-

# #### Necessary Libraries

from . import random, jnp, jit, vmap, np, grad, jacfwd, hessian, pmap, Precision


# *Own modification of random_layer_params to generate model parameters.*

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_model_params(m, key, scale=1e-2):
    #w_key, b_key = random.split(key)
    #print(tuple(scale * random.normal(key, (m, 1))))
    return (scale * random.normal(key, (m,)))#, scale * random.normal(b_key, (n,))


def init_model_params(size, key, scale):
    key = random.split(key,2)[-1] 
    return [random_model_params(s, key, scale) for s in size]


def mdiff_state(batched_state,params,t):
        i = jnp.arange(len(t[0]))
        #return (jacobian(batched_state,argnums=1)(params,t)[i,:,i,0])
        return [jnp.nan_to_num(_[i,:,i,:]) for s,_ in enumerate(jacfwd(lambda t : batched_state(params,t))(t))]
        #return jnp.nan_to_num(jacfwd(lambda t : batched_state(params,t))(t)[i,:,i,:])


def mdiff_params(batched_state,params,t):
        i = jnp.arange(len(t[0]))
        #return (jacobian(batched_state,argnums=1)(params,t)[i,:,i,0])
        #return [jnp.nan_to_num(_[i,:,i,:]) for s,_ in enumerate(jacfwd(lambda params : batched_state(params,t))(params))]
        return [jnp.nan_to_num(_) for s,_ in enumerate(jacfwd(lambda params : batched_state(params,t))(params))]
        #return jnp.nan_to_num(jacfwd(lambda params : batched_state(params,t))(params))
        #return jnp.nan_to_num(jacfwd(lambda params : batched_state(params,t))(params)[i,:,i,:])


class model(object):
   
    def __init__(self, stoich, model_scale=0.0001, usepmap=False):
        self.prec  = prec = 1e-12
        self.__randkey__ = random.PRNGKey(0)
        self.M = jnp.array(stoich)
        u, s, vt  = jnp.linalg.svd(self.M)
        self.S    = jnp.diag(s[s>prec])
        self.Sinv = jnp.diag(jnp.reciprocal(s[s>prec]))
        self.U   = u[:,jnp.arange(len(s))[s>prec]]
        self.Ur  = u[:,jnp.arange(len(s))[s>prec]]
        self.Un  = jnp.hstack((u[:,jnp.arange(len(s))[s<=prec]],u[:,len(s):]))
        self.Pr  = self.Ur.dot(self.Ur.T)
        self.Pn  = self.Un.dot(self.Un.T)
        #self.U_  = u[:,jnp.arange(len(s))[s<=prec]]
        self.Vt  = vt[jnp.arange(len(s))[s>prec],:]
        #self.Vt_ = vt[jnp.arange(len(s))[s<=prec],:]
        
        self.omega = [[_[1] for _ in p] for p in [list(filter(lambda x : x[0]<0,zip(c,range(len(c))))) for c in self.M.T]]
        self.params = init_model_params([self.M.shape[1]], self.__randkey__, model_scale)
        if usepmap:
            #self.batched_eval = jit(pmap(self.single_eval,in_axes=(None,0)))
            self.batched_eval = jit(pmap(self.single_eval,in_axes=(None,0)))
            self.batched_r    = jit(pmap(self.r,in_axes=(None,0)))
        else:
            self.batched_eval = jit(vmap(self.single_eval,in_axes=(None,0)))
            self.batched_r    = jit(vmap(self.r,in_axes=(None,0)))
        self.diff_eval   = jit(lambda params, t : mdiff_state(self.batched_eval,params,t))
        self.diff_r      = jit(lambda params, t : mdiff_state(self.batched_r,params,t))
        self.diff_params = jit(lambda params, t : mdiff_params(self.batched_eval,params,t))

    def f(self,batch):
        t, x = batch
        return jnp.array([jnp.prod(jnp.array([jnp.power(x[s],-self.M[s,_]) for s in self.omega[_]])) for _ in range(self.M.shape[1])])
    
    def k(self, m_params, batch):
        #'''
        #logk, = m_params
        #return jnp.exp(logk)
        #''' 
        k, = m_params
        return k
    
    def r(self, m_params, batch):
        t, x = batch
        return self.k(m_params, batch)*self.f(batch)
    
    def single_eval(self, m_params, batch):
        return jnp.dot(self.M,self.r(m_params,batch),precision=Precision.HIGHEST)
    
    def __call__(self,batch):
        return self.batched_eval(self.params,batch)

