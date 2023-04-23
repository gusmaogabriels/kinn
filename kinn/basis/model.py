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

from . import random, jnp, jit, vmap, onp, grad, jacfwd, hessian, pmap, Precision
from jax.lax import while_loop
from jax import jacrev, jacobian


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


def get_svd(M,threshold=1e-8):
    u,s,vt = jnp.linalg.svd(M)
    s = jnp.concatenate((s,jnp.zeros(u.shape[1]-len(s))))
    sv = jnp.concatenate((s,jnp.zeros(jnp.max(jnp.array([0,vt.T.shape[1]-len(s)])))))
    Un = u[:,s<threshold]
    Ur = u[:,s>=threshold]
    S = jnp.diag((s[:Ur.shape[1]]))
    Sinv = jnp.diag(jnp.reciprocal(s[s>=threshold]))
    if vt.T.shape[1]>len(sv):
        Vr = vt.T[:,sv>=threshold]
        Vn = vt.T[:,sv<threshold]
    else:
        Vr = vt.T
        Vn = jnp.empty([1,1])
    return {'Ur':Ur,'Un':Un,'S':S,'Sinv':Sinv,'Vr':Vr,'Vn':Vn}

class model(object):
   
    def __init__(self, stoich, nobs=None, model_scale=0.0001, usepmap=False):
        self.nobs = nobs
        self.prec  = prec = 1e-12
        self.__randkey__ = random.PRNGKey(0)
        self.M = jnp.array(stoich)

        for k,v in get_svd(self.M).items():    
            setattr(self,k,v)

        for _ in ['r','n']:
            setattr(self,'U'+_+'o',getattr(self,'U'+_)[:self.nobs,:])
            setattr(self,'U'+_+'l',getattr(self,'U'+_)[self.nobs:,:])

        self.Pr  = self.Ur.dot(self.Ur.T)
        self.Pn  = self.Un.dot(self.Un.T)

        u,s,vt = jnp.linalg.svd(stoich[self.nobs:,:])      
        self.Vrl = vt.T[:,:jnp.sum(s>self.prec)]
        self.Vnl = vt.T[:,jnp.sum(s>self.prec):]
        u,s,vt = jnp.linalg.svd(stoich[self.nobs:,:])      
        self.Vro = vt.T[:,:jnp.sum(s>self.prec)]
        self.Vno = vt.T[:,jnp.sum(s>self.prec):]       

        self.Mpssh = self.M[:self.nobs,:].dot(self.Vnl)
        for k,v in get_svd(self.Mpssh).items():    
            setattr(self,k+'_pssh',v)

                
        for _ in ['Uro','Url','Uno','Unl']:
            u, s, vt  = jnp.linalg.svd(getattr(self,_).T)
            setattr(self,_+'r',u[:,:len(s)][:,s>=prec])
            setattr(self,_+'n',jnp.hstack((u[:,:len(s)][:,s<prec],u[:,len(s):])))
            e, p  = jnp.linalg.eigh(getattr(self,_).dot(getattr(self,_).T))
            setattr(self,_+'r_',p[:,:len(e)][:,e>=prec])
            setattr(self,_+'n_',jnp.hstack((p[:,:len(e)][:,e<prec],p[:,len(e):])))        

        self.theta_ref = 298.15*8.316455e-3

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
        if len(batch) == 2:
            x, theta = batch
        else:
            x, = batch
        return jnp.array([jnp.prod(jnp.array([jnp.power(x[s],-self.M[s,_]) for s in self.omega[_]])) for _ in range(self.M.shape[1])])
    
    def k(self, m_params, batch):
        logk, = m_params
        return jnp.exp(logk)
    
    def r(self, m_params, batch):
        return self.k(m_params, batch)*self.f(batch)
    
    def single_eval(self, m_params, batch):
        return jnp.dot(self.M,self.r(m_params,batch),precision=Precision.HIGHEST)
    
    def __call__(self,batch):
        return self.batched_eval(self.params,batch)


class power_law(model):
    
    def r(self, m_params, batch):
        x, rtheta = batch
        y = x
        return (jnp.log(y+self.prec).dot(jnp.exp(m_params['w'].T))+(-jnp.exp(m_params['b'])/rtheta-(jnp.exp(m_params['c'])-jnp.exp(-m_params['c']))/self.theta_ref))
        #return self.k(m_params, batch)*self.f(batch)

from functools import partial
class pssh(model):

    #PSSH to the surface

    def __init__(self, stoich, nobs=None, model_scale=0.0001, usepmap=False):
        super(pssh,self).__init__(stoich, nobs, model_scale=0.0001, usepmap=False)
        self.dpssh = jacobian(lambda m_params, xg, xl, theta : self.pssh_res(m_params,xg,xl,theta),argnums=2)

    @partial(jit, static_argnums=(0))
    def pssh_res(self, m_params, xg, xl, theta):
        return self.Vrl.T.dot(self.r(m_params,[jnp.concatenate((xg,xl)),theta]))

    @partial(jit, static_argnums=(0))
    def upd_xl(self,m_params,xg,xl,theta):
        dpssh = self.dpssh(m_params,xg,xl,theta)
        dpssh = dpssh[:,:-1]-dpssh[:,-1:]
        dxl = jnp.linalg.pinv(dpssh).dot(-self.pssh_res(m_params, xg, xl, theta))
        xl0 = jnp.concatenate((xl[:-1]+dxl,xl[-1:]-jnp.sum(dxl)))
        return xl0*(xl0>0)/jnp.sum(xl0*(xl0>0))

    @partial(jit, static_argnums=(0))
    def pssh_xl(self,m_params,batch):
        xn = batch[0]
        theta = batch[-1]
        x = xn[:-1]
        xg = x[:self.nobs]
        xl0 = x[self.nobs:]
        xl = while_loop(jit(lambda xl : jnp.linalg.norm(self.pssh_res(m_params,xg,xl,theta))>self.prec),\
                   jit(lambda xl : self.upd_xl(m_params,xg,xl,theta)),
                   xl0)
        return xl

    @partial(jit, static_argnums=(0))
    def single_eval(self, m_params, batch):
        xn = batch[0]
        theta = batch[-1]
        x = xn[:-1]
        xg = x[:self.nobs]
        xl = self.pssh_xl(m_params,batch)        
        return jnp.dot(jnp.vstack((self.M,jnp.sum(self.M,axis=0).reshape(1,-1))),\
            self.r(m_params,[jnp.concatenate((xg,xl)),theta]),precision=Precision.HIGHEST)

pfr = pssh

# +
from jax.random import multivariate_normal as mvn

class model_stochastic(object):
    
    def __init__(self, stoich, model_scale=0.0001):
        self.__randkey__ = random.PRNGKey(0)
        self.M = jnp.array(stoich)
        self.params = model_scale * random.normal(self.__randkey__, (self.M.shape[0],self.M.shape[0]))
        self.batched_eval = jit(pmap(self.single_eval,in_axes=(None,0)))
        self._ = 0

    def single_eval(self, m_params, batch):
        t, theta = batch
        k, = m_params
        k = jnp.exp(k)
        r = jnp.dot(self.M,mvn(random.PRNGKey(self._),theta,self.M))
        self._ += 1
        return jnp.dot(self.M,r)
    
    def __call__(self,batch):
        return self.batched_eval(self.params,batch)
