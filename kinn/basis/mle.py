#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Fri Aug 02 17:45:50 2019
#
# @author:gusmaogabriels@gmail.com // gusmaogabriels@gatech.edu
#
# """

# ## Trainer Class

# -*- coding: utf-8 -*-

# #### Necessary Libraries

from readline import append_history_file
from . import jnp, random, jit, partial, itertools, clear_output, grad, jacfwd, hessian, pmap, Precision, partial
from jax import vmap, jacrev
from jax.tree_util import tree_map
from jax.nn import softplus, sigmoid, swish, relu
from jax.lax import Precision
from jax.tree_util import tree_flatten
from jax.flatten_util import ravel_pytree
import optax
from . import nn

from sklearn.covariance import MinCovDet
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from matplotlib import pyplot as plt
from jax import value_and_grad


import warnings
warnings.filterwarnings('ignore')

# Useful activation functions

tanh       = jit(lambda x : jnp.tanh(x))
gauss      = jit(lambda x : jnp.exp(-x**2))
sin        = jit(lambda x : jnp.sin(x))
cos        = jit(lambda x : jnp.cos(x))

class nn_c(nn):
    
    def __init__(self,*args,**kwargs):
        self.bc = kwargs.pop('bc')
        super().__init__(*args,**kwargs)
    
    def constraints(self,x,t):
        t0, x0 = self.bc
        return x0+(x-x0)*(t-t0)#*jnp.tanh((t-t0))

class nn_cn(nn):
    
    def constraints(self,x,t):
        
        #x_cov_ = jnp.sin(jnp.concatenate((jnp.tanh(x)*jnp.pi,jnp.array([0.]))))**2
        x_cov_ = jnp.sin(jnp.concatenate((x,jnp.array([0.]))))**2
        x_cov  = 1.-x_cov_
        for _ in range(len(x_cov)):
            x_cov *= jnp.concatenate((jnp.array([1.]*(_+1)),x_cov_[:-(_+1)]))
        return x_cov

class nn_cn_bc(nn_c):
    
    def __init__(self,*args,**kwargs):
        #self.bc = kwargs.pop('bc')
        super().__init__(*args,**kwargs)
        ang0 = []
        for _ in range(len(self.bc[1])-1):
            den = jnp.prod([jnp.sin(ang0[ix])**2. for ix in range(_)])
            if den != 0.:
                ang0 += [jnp.arcsin(jnp.sqrt(1.-self.bc[1][_]/den))]
            else:
                ang0 += [jnp.array(jnp.pi/2.)]
        #ang0 = jnp.clip(jnp.nan_to_num(jnp.arctanh((jnp.array(ang0))*(4./(jnp.pi))-1.)),a_min=-10.,a_max=10.)
        #ang0 = jnp.clip(jnp.nan_to_num(jnp.arctanh(jnp.array(ang0)*2./jnp.pi)),a_min=-30.,a_max=30.)
        self.bc_ang = [self.bc[0],jnp.array(ang0)]
    
    def constraints(self,x,t):
        t0, x0 = self.bc_ang
        #x      = x0+jnp.tanh(t-t0)*(x-x0)
        x      = x0+jnp.tanh((t-t0))*(x-x0)
        #x_cov_ = jnp.sin(jnp.concatenate(((1.+jnp.tanh(x))*jnp.pi/4.,jnp.array([0.]))))**2
        #x_cov_ = jnp.sin(jnp.concatenate((jnp.tanh(x)*jnp.pi/2.,jnp.array([0.]))))**2
        x_cov_ = jnp.sin(jnp.concatenate((x,jnp.array([0.]))))**2
        x_cov  = 1.-x_cov_
        for _ in range(len(x_cov)):
            x_cov *= jnp.concatenate((jnp.array([1.]*(_+1)),x_cov_[:-(_+1)]))
        #return x0*(1.-jnp.tanh(t-t0))+x_cov*jnp.tanh(t-t0)*jnp.sum(x0)
        return x_cov

class nn_npt(object):
    
    def __init__(self, nns, model, usepmap=False, bc=None, nobs=None, trig=False,\
                 damp=False, gain=1.,mode=None, nn_scale=0.001, out_n=[], rndecomp=True):
        self.nns = nns
        self.bc  = bc
        self.model = model
        self.rndecomp = rndecomp
        self.out_n   = out_n
        self.nobs    = nobs
        self.prec    = prec = 1e-12
        self.Ur      = self.model.Ur
        self.Un      = self.model.Un
        self.Pr      = self.Ur.dot(self.Ur.T) if len(self.Ur)>0 else jnp.eye(self.nns[0].layers_sizes[i][0][-1])
        self.Pn      = self.Un.dot(self.Un.T) if len(self.Un)>0 else jnp.eye(self.nns[0].layers_sizes[i][0][-1])
        self.Uro     = self.Ur[:self.nobs,:]
        self.Url     = self.Ur[self.nobs:,:]
        self.Uno     = self.Un[:self.nobs,:]
        self.Unl     = self.Un[self.nobs:,:]
        self.Unl_l1  = self.Unl.sum(axis=0)        
        self.Prl_inv = jnp.linalg.pinv(self.Url.T.dot(self.Url))
        self.Pnl_inv = jnp.linalg.pinv(self.Unl.T.dot(self.Unl))
        U,s,Vt       = jnp.linalg.svd(self.Uro.T)
        self.dimor   = sum(s>=prec)
        self.Prno    = jnp.hstack((U[:,:len(s)][:,s<prec],U[:,len(s):]))
        self.Prro    = U[:,:len(s)][:,s>=prec]
        U,s,Vt       = jnp.linalg.svd(self.Uno.T)
        self.dimon   = sum(s>=prec)
        #self.dimon   = self.Uno.shape[1]
        self.Pnno    = jnp.hstack((U[:,:len(s)][:,s<prec],U[:,len(s):]))
        self.Pnro    = U[:,s>=prec]
        U,s,Vt       = jnp.linalg.svd(self.Url.T.dot(self.Url))
        self.dimr    = sum(s<prec)+(U.shape[1]-len(s))
        self.Pnrl    = jnp.hstack((U[:,:len(s)][:,s<prec],U[:,len(s):]))
        self.Prrl    = U[:,s>=prec]
        U,s,Vt       = jnp.linalg.svd(self.Unl.T.dot(self.Unl))
        self.dimn    = sum(s<prec)+(U.shape[1]-len(s))
        #self.dimn    = self.Unl.shape[1]
        self.Pnnl    = jnp.hstack((U[:,:len(s)][:,s<prec],U[:,len(s):]))
        self.Prnl    = U[:,s>=prec]
        
        self.layers_sizes = [[_.copy() for _ in nn.layers_sizes] for nn in self.nns]
        self.trig    = trig
        self.damp    = damp
        self.gain    = gain
        self.mode    = mode
        self.null_size = [ls[0][-1] for ls in self.layers_sizes]
        for i in [0]:
            if len(self.Ur)>self.nobs:
                self.layers_sizes[i][0][-1] = self.layers_sizes[i][0][-1]-self.nobs+self.dimr
            else:
                self.layers_sizes[i][0][-1] = self.dimr
            self.nns[i].layers_sizes = [self.layers_sizes[i][0].copy()]
            self.nns[i]._init_params()
            self.nns[i]._jit_compile()
        self.params = list(zip(*[_.params for _ in self.nns]))
        self.flatfun = jit(lambda x : jnp.hstack([vmap(lambda k : k.flatten())(_) for _ in tree_flatten(x)[0]]))
        self.cnstr  = [self.flatfun]
        self.init(nn_scale)
        
        if usepmap :
            self.batched_state = jit(pmap(self.state, in_axes=(None,0,None)))
        else:
            self.batched_state = jit(vmap(self.state, in_axes=(None,0,None)))
        self.diff_state    = jit(lambda params,t,zn:vmap(jacfwd(self.state,argnums=(1)),in_axes=(None,0,None))(params,t,zn))
        self.diff_state2   = jit(lambda params,t,zn:vmap(jacfwd(jacfwd(self.state,argnums=(1)),argnums=(1)),in_axes=(None,0,None))(params,t,zn))
        self.d_pars        = jit(lambda params,t,zn:vmap(jacfwd(self.state,argnums=(0)),in_axes=(None,0,None))(params,t,zn))
        self.d_state_pars     = jit(lambda params,t,zn:vmap(jacfwd(jacfwd(self.state,argnums=(1)),argnums=(0)),in_axes=(None,0,None))(params,t,zn))
        # self.ddxdx             = vmap(jit(lambda params,t,zn:self.cnstr[0](jacfwd(jacfwd(self.state,argnums=(1)),argnums=(0))(params,t,zn)[0][0]).dot(\
        #                             jnp.linalg.pinv((self.cnstr[0](jacfwd(self.state,argnums=(0))(params,t,zn)[0][0]))))),in_axes=(None,0,None))
        self.ddxdx             = vmap(jit(lambda params,t,zn:self.Ur.dot(self.Ur.T.dot(self.cnstr[0](jacfwd(jacfwd(self.state,argnums=(1)),argnums=(0))(params,t,zn)[0][0]))).dot(\
                                    jnp.linalg.pinv(self.Ur.T.dot(self.cnstr[0](jacfwd(self.state,argnums=(0))(params,t,zn)[0][0])))).dot(self.Ur.T)),in_axes=(None,0,None))                                    
        self.ddxpdx            = vmap(jit(lambda params,t,zn:self.Ur.T.dot((self.cnstr[0](jacfwd(jacfwd(self.state,argnums=(1)),argnums=(0))(params,t,zn)[0][0])).dot(\
                                    (self.cnstr[0](jacfwd(self.state,argnums=(0))(params,t,zn)[0][0]).T))).dot(self.Ur)),in_axes=(None,0,None))
        
    def set_nullspace(self,zn):
        self.zn = zn

    def set_gamma(self,zon,zln,Ugn):
        self.zon = zon
        self.zln = zln
        self.Ugn = Ugn
        
    def init(self, nn_scale):
        self.nn_scale    = nn_scale
        self.__randkey__ = random.PRNGKey(0)
        if len(self.Un)>self.nobs:
            self.params += [[random.normal(random.PRNGKey(0), (len(self.Un)-self.nobs-1*self.trig+self.dimn,)) for ls in self.layers_sizes]]
        else:
            pass
 
    def set_params(self,params):
        for i, nn in enumerate(self.nns):
            nn.set_params([param[i] for param in params])
    
    @partial(jit,static_argnums=(0,))    
    def normtrig2(self,x,t):
        n = len(x)
        arg = jnp.reciprocal(jnp.arange(2,n+2))[::-1]
        y = jnp.sin(sigmoid(x)*jnp.pi/2.+0*jnp.pi/4.)**2.
        z = jnp.prod((jnp.ones([n]*2)-jnp.triu(jnp.ones([n]*2))).dot(jnp.diag(y),precision=Precision.HIGHEST)+jnp.diag(-y)+jnp.triu(jnp.ones([n]*2)),axis=1)
        return jnp.concatenate((z,jnp.array([jnp.prod(y)])))
          
    @partial(jit,static_argnums=(0,))
    def normtrig(self,x,t,offset):
        y = sigmoid(x+offset)
        _sin = jnp.concatenate((jnp.array([1.]),y))
        _cos = jnp.concatenate((1-y,jnp.array([1.])))    
        return jnp.cumprod(_sin)*_cos

    vnormtrig = vmap(normtrig,in_axes=(0,None,None))
        
    @partial(jit,static_argnums=(0,))
    def dnormtrig(self,x,t,xln,offset):
        return self.normtrig(x,t,offset)-xln
    
    def clipzr(self,sigzr,xnl):
        return self.Ur.dot(-self.Prl_inv.dot(self.Url.T.dot(xnl))+self.Pn.dot(sigmoid(sigzr)))#*self.Prl_inv.dot(self.Url.T.sum(axis=1)))
        
    def __call__(self,t,nn):
        return self.batched_state(self.params[nn],t)

    def state(self,params,t,zn):
        if self.mode == 'forward':
            t0, x0 = self.bc
        else:
            x0 = jnp.zeros(len(self.Ur))
        #dx    = jnp.concatenate([self.nns[i].state(params[i],t) for i in range(len(params))])-x0
        out_r  = self.nns[0].state(params[0][0],t)      
        #out_n  = params[0][0][-1][-1]
        if self.trig and self.nobs==0:
            dx = self.normtrig(out_r[self.nobs:],t)-x0
        elif self.trig and self.nobs>0:
            if self.rndecomp:
                out_n  = params[1][0]
                #out_n  = self.out_n
                #zn   = self.Pnro.dot(out_n[:self.dimon],precision=Precision.HIGHEST)+self.Unl.T.dot(self.normtrig(out_n[self.dimon:],t),precision=Precision.HIGHEST)

                if len(zn)>0:
                    pass
                else:
                    if hasattr(self,'zn'):
                        zn   = self.zn
                    elif hasattr(self,'zon'):
                        zn   = self.zon + self.zln + self.Ugn.dot(out_n[:self.Ugn.shape[1]])
                    else:
                        zn   = self.Pnnl.dot(out_n[:self.dimn],precision=Precision.HIGHEST)+self.Pnl_inv.dot(self.Unl.T).dot(self.normtrig(out_n[self.dimn:],t,0.),precision=Precision.HIGHEST)
                    
                dx_n = self.Un.dot(zn,precision=Precision.HIGHEST)
                dxl  = self.dnormtrig(out_r[self.dimr:],t,dx_n[self.nobs:],0.)
                zr   = self.Pnrl.dot(out_r[:self.dimr],precision=Precision.HIGHEST)+\
                        self.Prl_inv.dot(self.Url.T.dot(dxl,precision=Precision.HIGHEST),precision=Precision.HIGHEST)
                dx_r = self.Ur.dot(zr,precision=Precision.HIGHEST)
                dx = dx_r+dx_n
            else:
                dx = jnp.concatenate((out_r[:self.nobs],self.normtrig(out_r[self.nobs:],t,0.)))
        else: # homogenous
            if self.rndecomp:
                dx_n = self.Un.dot(self.out_n,precision=Precision.HIGHEST)
                dx_r = self.Ur.dot(out_r,precision=Precision.HIGHEST)
                dx   = dx_r+dx_n
            else:
                dx = out_r
        if self.mode == 'forward':
            if self.damp:
                xt = self.nns[1].state(params[0][1],t)
            else:
                xt = 0.
            #x = (dx_r+dx_n)*jnp.tanh((t-t0)*self.gain*jnp.exp(xt))+x0
            #x = dx_r*jnp.tanh((t-t0)*self.gain*jnp.exp(xt))+x0
            x = dx_r*jnp.tanh((t-t0)*self.gain*jnp.exp(xt))+x0
        elif self.mode == 'inverse':
            x = dx
        else:
             raise Exception('mode not implemented ({})'.format(self.mode))
        return x

#@jit
def r2_score(y,ypred):
    y_, ypred_   = [(_-_.mean(axis=0).reshape(1,-1))/jnp.where(_.std(axis=0).reshape(1,-1)==0,1.,\
                                                              _.std(axis=0).reshape(1,-1)) for _ in [y,ypred]]
    return jnp.mean(jnp.diag(jnp.dot(y_.T,ypred_)*jnp.reciprocal(len(y_)))**2)   

__metrics__ = {'r2' : r2_score,
              'MAE' : mean_absolute_error,
              'MSE' : mean_squared_error,
              'TotalVar': explained_variance_score}

@jit
def mrse(y,yhat):#,mode=0):
    return jnp.mean((y-yhat)**2,axis=1)

def delta(y,yhat):
    return y-yhat

@jit
def oas(XT):
    """
    https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/covariance/_shrunk_covariance.py#L434
    """

    n_features, n_samples = XT.shape

    emp_cov = jnp.cov(XT)
    mu = jnp.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = jnp.mean(emp_cov ** 2)
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

    #shrinkage = 1. if den == 0 else min(num / den, 1.)
    shrinkage = jnp.min(jnp.array([num / den, 1.]))*(den>0.)+1.*(den<=0.)
    shrunk_cov = (1. - shrinkage) * emp_cov + jnp.eye(n_features) * shrinkage * mu
    #shrunk_cov.flat[::n_features + 1] += shrinkage * mu

    #return shrunk_cov, shrinkage
    return emp_cov, shrinkage
    
@jit    
def _cov_oas(emp_cov,n_samples):
    
    
    n_features = emp_cov.shape[0]

    mu = jnp.trace(emp_cov) / n_features

    # formula from Chen et al.'s **implementation**
    alpha = jnp.mean(emp_cov ** 2)
    num = alpha + mu ** 2
    den = (n_samples + 1.) * (alpha - (mu ** 2) / n_features)

    #shrinkage = 1. if den == 0 else min(num / den, 1.)
    shrinkage = jnp.min(jnp.array([num / den, 1.]))*(den>0.)+1.*(den<=0.)
    shrunk_cov = (1. - shrinkage) * emp_cov + jnp.eye(n_features) * shrinkage * mu
    #shrunk_cov.flat[::n_features + 1] += shrinkage * mu

    #return shrunk_cov, shrinkage
    return emp_cov, shrinkage
    
cov_oas = vmap(_cov_oas,in_axes=(0,None))

class opt(object):
    
    def __init__(self, nn, model, num_iter=1e2, num_epochs=1e2, batch_size=0.99, split=0.99, tol=0., rtol = None, stol=1e-2, kltol=None,\
                 verbose=False, mode='forward', bc = None, historian = False, iter_data = [None], err_tags=[], nobs = None, inference=False, kl_inference=False):
        if not hasattr(nn,'__iter__'):
            nn = [nn] # making nn subclass"able"
        self.nn = nn
        self.model     = model
        self.mode      = mode
        self.bc        = bc
        self.inference = inference
        self.kl_inference = kl_inference
        if mode == 'forward':
            self.params = {'sm':[_.params for _ in self.nn],
                           'pm':[]}
            self.ndata = nobs
            scales = False
        elif mode == 'inverse':
            self.params = {'sm':[_.params for _ in self.nn],
                           'pm':[jnp.log(jnp.abs(self.model.params[0]))]}
            if isinstance(nobs,int) and nobs<self.model.M.shape[0]:
                scales = jnp.hstack((jnp.ones(self.model.M.shape[0]-nobs)[:,jnp.newaxis]))
            else:
                scales = None
            self.ndata = self.model.M.shape[0]
        else: 
            raise Exception("mode parameter must be either 'forward' or 'inverse'")
        
        self.num_iter    = int(num_iter)
        self.num_epochs  = int(num_epochs)
        self.cond        = 1e3
        self.tol         = tol
        self.rtol        = rtol if rtol else tol
        self.kltol       = self.rtol if not kltol else kltol 
        self.stol        = stol
        self.batch_size  = batch_size
        self.split       = split
        self.verbose     = verbose
        self.loss        = jit(self._loss)
        self.step        = jit(self._step)
        self.scales      = scales
        self.nobs        = nobs
        self.best_params = {'error':jnp.inf,'params':[]}
        self.errors      = [0.]
        self.epoch       = 0
        self.min_error   = jnp.inf
        self.omegas      = {'pm':{'r':[],'n':[]},'interp':{'r':[],'n':[]}}
        self.sigmas      = {'pm':[],'interp':[],'mpars':[]}
        self.mus         = {'pm':[],'interp':[],'mpars':[]}
        self.lcte        = {'pm':[],'interp':[]}
        self.lkhd        = {'pm':[],'interp':[]}
        self.prec        = prec = self.model.prec
        self.Ur          = self.model.Ur
        self.Uro         = self.model.Ur[:self.nobs,:]
        self.Url         = self.model.Ur[self.nobs:,:]
        self.Un          = self.model.Un
        self.Uno         = self.model.Un[:self.nobs,:]
        self.Unl         = self.model.Un[self.nobs:,:]
                
        for _ in ['Uro','Url','Uno','Unl']:
            u, s, vt  = jnp.linalg.svd(getattr(self,_).T)
            setattr(self,_+'r',u[:,:len(s)][:,s>=prec])
            setattr(self,_+'n',jnp.hstack((u[:,:len(s)][:,s<prec],u[:,len(s):])))
            e, p  = jnp.linalg.eigh(getattr(self,_).dot(getattr(self,_).T))
            setattr(self,_+'r_',p[:,:len(e)][:,e>=prec])
            setattr(self,_+'n_',jnp.hstack((p[:,:len(e)][:,e<prec],p[:,len(e):])))
        
       
        self.Pro         = self.Uro.dot(jnp.eye(self.Ur.shape[1])+self.Url.T.dot(self.Url)).dot(self.Uro.T)
        self.Prl         = (jnp.eye(self.Url.shape[0])+self.Url.dot(self.Url.T)).dot(self.Url.dot(self.Uro.T))
        self.Pno         = self.Uno.dot(jnp.eye(self.Un.shape[1])+self.Unl.T.dot(self.Unl)).dot(self.Uno.T)
        self.Pnl         = (jnp.eye(self.Unl.shape[0])+self.Unl.dot(self.Unl.T)).dot(self.Unl.dot(self.Uno.T))

        
        self.proj_rol    = jnp.vstack((self.Pro,self.Prl))        
        for i,f,_ in [[0,self.nobs,'o'],[self.nobs,self.model.M.shape[0],'l']]:
            u, s, vt          = jnp.linalg.svd(self.Ur[jnp.arange(i,f),:])
            setattr(self,'B'+_+'r',u[:,:len(s)][:,s>=prec])
            setattr(self,'B'+_+'n',jnp.hstack((u[:,:len(s)][:,s<prec],u[:,len(s):]))) 

        self.proj_nol    = jnp.vstack((self.Pno,self.Pnl))
                    
        if self.mode == 'inverse' and self.inference:
            self.proj_r = self.Bor
            self.proj_n = self.Bon                
        else:
            self.proj_r = self.Ur
            self.proj_n = self.Un
                     
        self.projs             = dict() 
        self.__historian__     = historian
        self.iter_data         = iter_data
        self.__hasoptimizer__  = False
        self.__isinitialized__ = False
        self.err_tags          = err_tags
        self.scaleopt_kwargs   = {'tol':1e-15,'method':'bfgs','options':{'gtol': 1e-15, 'disp': True, 'maxiter': 1e4, 'maxfeva': 1e5}}
        self.scaleopt_res      = None
        self.save_tags         =  ['params','iter_data','best_params','error','errors','epoch','err_tags',\
                                  'nobs','alpha','omegas','sigmas','inference', 'model.params','scales',\
                                  'scaleopt_res','ndata','lcte', 'projs']#'scale','pm_dx',

    def _loss(self, params, omegas, batch, zn):
        return jnp.array([_.mean() for _ in self.err_fun(self.res_fun(params,batch, zn)[0], omegas)]).mean()#+\
            #[jnp.linalg.norm(self.nn[i].ddxpdx(params['sm'][i],batch[i][0],zn[i])) for i in range(len(self.nn))]).mean()
    
    def set_optimizer(self, opt_dict):
        self.opt_dict = opt_dict
        self.__hasoptimizer__ = True

    def _proj_grads(self,params, omegas, batch, zn, subproj):
        grads = self.grads(params, omegas, batch, zn)
        grads['pm'][0] = subproj.dot(subproj.T).dot(grads['pm'][0])
        return grads
            
    def _step(self, opt_state, params, omegas, batch, zn, subproj):
        updates, opt_state = self.optimizer.update(self.proj_grads(params, omegas, batch, zn, subproj), opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    def send_parameters(self):
        if  self.mode == 'inverse':
            self.model.params = [jnp.exp(self.params['pm'][0])]
            for i, _ in enumerate(self.params['sm']):
                self.nn[i].set_params(_)
        else:
            for i, _ in enumerate(self.params['sm']):
                self.nn[i].set_params(_)
                
    def reinit(self,nn=False):
        if nn:
            self.nn.init()
            self.params['sm'] = [_.params for _ in self.nn[0]]
        self.opt_state = self._opt_init(self.params)
        self.best_params = {'error':jnp.inf,'params':[]}
    
    def jit_compile(self):
        # 're'JIT in case static memory maps have changed
        self.cinv_v         = jit(vmap(self._cinv,in_axes=(0)))
        self.vorthproj   = jit(vmap(self._orthproj,in_axes=(0)))
        self.vcolinproj  = jit(vmap(self._colinproj,in_axes=(0)))
        self.div_vn      = jit(vmap(self._div,in_axes=(0,None)))
        self.div_vv      = jit(vmap(self._div,in_axes=(0,0)))
        self.comm         = jit(lambda x,y : x.dot(y)-y.dot(x))
        self.comm_v       = jit(vmap(lambda x,y : self.comm(x,y), in_axes=(0)))
        self.quad           = jit(lambda x, H : x.dot(H,precision=Precision.HIGHEST).dot(x.T,precision=Precision.HIGHEST))
        self.quad_vn        = jit(vmap(lambda g, sigma : self.quad(g,sigma),in_axes=(0,None)))
        self.quad_vv        = jit(vmap(lambda g, sigma : self.quad(g,sigma),in_axes=(0,0)))
        self.quad_nv        = jit(vmap(lambda g, sigma : self.quad(g,sigma),in_axes=(None,0)))
        self.proj           = jit(vmap(lambda U, x : jnp.dot(U,x,precision=Precision.HIGHEST),in_axes=(None,0)))
        self.proj_vn        = jit(vmap(lambda U, x : jnp.dot(U,x,precision=Precision.HIGHEST),in_axes=(0,None)))
        self.proj_nv        = jit(vmap(lambda U, x : jnp.dot(x,U,precision=Precision.HIGHEST),in_axes=(None,0)))
        self.proj_vv        = jit(vmap(lambda U, x : jnp.dot(x,U,precision=Precision.HIGHEST),in_axes=(0,0)))
        self.loss     = jit(self._loss) 
        f             = lambda params, omegas, batch, zn : jnp.nan_to_num(self.loss(params, omegas, batch, zn))
        self.grads    = jit(grad(f, argnums=0))
        self.mperr    = vmap(jit(lambda errs_model, errs_data, pm_dx, pm_dp : self._subsolve(pm_dp,-jnp.dot(pm_dx,errs_data,\
                                        precision=Precision.HIGHEST)-errs_model)),in_axes=(0,0,0,0))
        self.avgqinv  = jit(lambda x : vmap(lambda x : x.T.dot(x,precision=Precision.HIGHEST))(x).mean(axis=0))
        self.avgproj  = jit(lambda f,x : vmap(lambda f, x : f.T.dot(x,precision=Precision.HIGHEST))(f,x).mean(axis=0))
        self.vsubinv  = jit(vmap(lambda x : self._subinv(x),in_axes=(0)))
        self.fb       = jit(vmap(lambda gsl,edx,gso,exo,pm_dp,dp: gsl.T.dot(edx-jnp.dot(gso,exo,precision=Precision.HIGHEST)-jnp.dot(pm_dp,dp,precision=Precision.HIGHEST),precision=Precision.HIGHEST)\
                                  ,in_axes=(0,0,0,0,0,None)))
        self.fiv      = jit(vmap(lambda Ainv,b : Ainv.dot(b-(jnp.sum(Ainv.dot(b,precision=Precision.HIGHEST)))/(jnp.sum(Ainv)),precision=Precision.HIGHEST)))
        self.flogdet  = jit(vmap(lambda x : jnp.sum(jnp.log(jnp.abs(jnp.linalg.eigvalsh(x))))))

        jit_dict = {'_get_stats':{'static_argnums':(1)},\
                    '_get_sigmas_full':{'static_argnums':(1)},\
                    '_upd_projs':{'static_argnums':(4)},\
                    '_hessian_fun':{'static_argnums':(3,4,5)},\
                    '_lkhd_fun':{'static_argnums':(3,4,5)},\
                    '_smp_khd_fun':{'static_argnums':(8,9,10)}}

        for _ in list(filter(lambda x : x.startswith('_') and \
            not x.startswith('__') and \
            callable(getattr(self,x)) and
            x not in jit_dict.keys(),self.__dir__())):
                setattr(self,_[1:],jit(getattr(self,_)))            
        for _ in jit_dict.keys():
            setattr(self,_[1:],jit(getattr(self,_),**jit_dict[_]))
        
    def initialize(self):
        if self.__hasoptimizer__:
            self.optimizer = optax.multi_transform({k:self.opt_dict[k]['optimizer'](**self.opt_dict[k]['kwargs']) for k in self.opt_dict.keys()},
                param_labels = {k:k for k in self.opt_dict.keys()})
            self.opt_state     = self.optimizer.init(self.params)
            self.jit_compile()
        else:
            raise Exception('Optimizer not defined.')

    def get_scales(self,datas):       
  
        fa = jit(lambda dy : jnp.diag(dy).dot(self.Unl.dot(self.Unl.T)).dot(jnp.diag(dy)))
        fb = jit(lambda dx, dy, gammar : jnp.diag(dy).dot(self.Unl.dot(self.Uno.T.dot(dx)+self.Unl.T.dot(jnp.diag(dy)).dot(gammar))))

        yls = [datas[_][1][:,self.nobs:] for _ in range(len(datas))]
        xos = [datas[_][1][:,:self.nobs] for _ in range(len(datas))]
        d = jnp.vstack(yls)
        A_ = d.T.dot(d)*jnp.reciprocal(len(d))+jnp.cov(d.T)
        e, P = jnp.linalg.eigh(A_)
        b_ = d.sum(axis=0)*jnp.reciprocal(len(d))
        esum = jnp.cumsum(e)/jnp.sum(e)
        self.sUr = Ur = P[:,esum>self.stol]
        self.sUn = Un = P[:,esum<=self.stol]
        gammar = Ur.dot(jnp.diag(jnp.reciprocal(e[esum>self.stol])).dot(Ur.T)).dot(b_)
        A = Un.T.dot(jnp.array([jnp.array([jnp.array([fa(dy)  for dy in yl[j+1:,:]-yl[j,:]]).sum(axis=0) \
                                    for j in range(len(yl)-1)]).sum(axis=0) for yl in yls]).mean(axis=0)).dot(Un)
        b = Un.T.dot(jnp.array([jnp.array([jnp.array([fb(dx,dy,gammar) for dx,dy in zip(xo[j+1:,:]-xo[j,:],yl[j+1:,:]-yl[j,:])]).sum(axis=0) \
                                                     for j in range(len(yl)-1)]).sum(axis=0) for xo,yl in zip(xos,yls)]).mean(axis=0))
        gamman = -Un.dot(jnp.linalg.pinv(A).dot(b))
        self.scales =  gamman+gammar
        self.res    =  jnp.concatenate([((yl*(self.scales))**2).sum(axis=1) for yl in yls]).mean()
        self.zn = [(jnp.hstack((datas[_][1][:,:self.nobs],\
                                         datas[_][1][:,self.nobs:]*self.scales)).dot(self.nn[_].Un)).mean(axis=0) for _ in range(len(datas))]
        # for _ in range(len(datas)):
            # self.nn[_].set_nullspace((jnp.hstack((datas[_][1][:,:self.nobs],\
            #                              datas[_][1][:,self.nobs:]*self.scales)).dot(self.nn[_].Un)).mean(axis=0))
            
            # zon = (datas[_][1][:,:self.nobs]).dot(self.nn[_].Uno).mean(axis=0)
            # zln = (datas[_][1][:,self.nobs:]*self.scales).dot(self.nn[_].Unl).mean(axis=0)
            # Ugn = (Un.T.dot(jnp.array([jnp.diag(datas[_][1][i,self.nobs:]).dot(self.nn[_].Unl) for i in range(datas[_][1].shape[0])]).mean(axis=0))).T
            # self.nn[_].set_gamma(zon,zln,Ugn)                                    
        return self.scales
            
    def _getattr(self,obj,tag):
        if len(tag) == 1:
            return getattr(obj,tag[0])
        else:
            return self._getattr(getattr(obj,tag[0]),tag[1:])
            
    def _setattr(self,obj,tag,value):
        if len(tag) == 1:
            setattr(obj,tag[0],value)
        else:
            self._setattr(getattr(obj,tag[0]),tag[1:],value)
        
    def dump(self,filename):
        jnp.savez(filename,\
                 data={tag:self._getattr(self,tag.split('.')) for tag in self.save_tags})
    
    def load(self,filename):
        objs = jnp.load(filename, allow_pickle=True)['data'].tolist()
        for tag in self.save_tags:
            try:
                self._setattr(self,tag.split('.'),objs[tag])
            except:
                print('Load error: missing attribute {}'.format(tag))
        if self.__hasoptimizer__:
            self.initialize()
        self.send_parameters()
        
    def get_state(self,state):
        iter_data = self.iter_data[state]
        if not hasattr(self,'idxs0'):
            self.idxs0 = None
        if not hasattr(self,'proj_r'):
            if self.mode == 'inverse' and not self.scale:
                self.proj_r = self.Bor
                self.proj_n = self.Bon                
            else:
                self.proj_r = self.Ur
                self.proj_n = self.Un
        self.iter_data += [[self.epoch,[self.params.copy()],[self.omegas],[self.errors.copy()],[repr(self.optimizer),self.opt_dict]]]
        self.epoch      = iter_data[0]
        self.params     = iter_data[1][0]
        self.errors     = iter_data[3][0]
        self.lkhd       = iter_data[3][1]
        self.iter_data += [[self.epoch,[self.params.copy()],[self.omegas],[self.errors.copy()],[repr(self.optimizer),self.opt_dict]]]
        
    @partial(jit, static_argnums=(0))
    def _subinv(self,A):      
        return self.cinv(A.T.dot(A,precision=Precision.HIGHEST))
    
    @partial(jit, static_argnums=(0))
    def _subsolve(self,A,y):
        x    = self._subinv(A).dot(A.T.dot(y,precision=Precision.HIGHEST),precision=Precision.HIGHEST)
        return x
        
    @partial(jit, static_argnums=(0))
    def _cinv(self,cov):
        return jnp.linalg.pinv(cov,rcond=None)

    @partial(jit,static_argnums=(0))   
    def _kl_div(self,c1,c2):
        # Kullback–Leibler for two zero-mean multivariate gaussian distributions
        cf = 50.
        e1, P1 = jnp.linalg.eigh(c1)
        e2, P2 = jnp.linalg.eigh(c2)
        tol = jnp.max(jnp.array([0.]+[(e[-1]-cf*e[0])/(cf-1.) for e in [e1,e2]]))
        e1 += tol
        e2 += tol
        tr = jnp.trace(P2.dot(jnp.diag(jnp.reciprocal(e2))).dot(P2.T).dot(P1).dot(jnp.diag(e1)).dot(P1.T))
        lg = jnp.sum(jnp.log(e2))-jnp.sum(jnp.log(e1))
        kb = (0.5)*(tr+lg-len(e1))
        return kb*(kb>0.)

    @partial(jit,static_argnums=(0))   
    def _js_div(self,c1,c2):
        # Jensen-Shannon for two zero-mean multivariate gaussian distributions
        cf = 50.
        e1, P1 = jnp.linalg.eigh(c1)
        e2, P2 = jnp.linalg.eigh(c2)
        em, Pm = jnp.linalg.eigh((c1+c2)/2.)
        tol = jnp.max(jnp.array([0.]+[(e[-1]-cf*e[0])/(cf-1.) for e in [e1,e2]]))
        e1 += tol
        e2 += tol
        em += tol
        lg = jnp.sum(jnp.log(em))-0.5*(jnp.sum(jnp.log(e1))+jnp.sum(jnp.log(e2)))
        kb  = 0.5*(lg)
        return kb#*(kb>0.)

    def _div(self,c1,c2):
        return self._js_div(c1,c2)

    def _get_stats(self,errs,center=True):
        if center:
            sigmas = tree_map(jit(lambda x : _cov_oas(jnp.cov(x,rowvar=False),x.shape[0])[0]),errs)
            sigmas['cross'] = tree_map(jit(lambda x, y : (x-x.mean(axis=0)).T.dot(y-y.mean(axis=0))/len(x)),errs['interp'],errs['pm']['interp'])
            sigmas['full'] = tree_map(jit(lambda x, y : jnp.cov(x.T,y.T)),errs['interp'],errs['pm']['interp'])
        else:
            sigmas = tree_map(jit(lambda x : x.T.dot(x)/len(x)),errs)
            sigmas['cross'] = tree_map(jit(lambda x, y : x.T.dot(y)/len(x)),errs['interp'],errs['pm']['interp'])
            sigmas['full'] = tree_map(jit(lambda x, y : jnp.dot(jnp.hstack((x,y)).T,jnp.hstack((x,y)))/len(x)),errs['interp'],errs['pm']['interp'])
        mus    = tree_map(jit(lambda x : jnp.mean(x,axis=0)),errs)
        vars    = tree_map(jit(lambda x : jnp.mean(x**2,axis=0)),errs)
        mus['full']    = tree_map(jit(lambda x, y : jnp.concatenate((jnp.mean(x,axis=0),jnp.mean(y,axis=0)))),errs['interp'],errs['pm']['pm'])
        return sigmas, mus, vars
    
    def _get_sigmas_full(self,errs,center=True):
        if center:
            return tree_map(lambda x, y : jnp.cov(x.T,y.T),errs['interp'],errs['mpars'])
        else:
            return tree_map(lambda x, y : jnp.dot(jnp.hstack((x,y)).T,jnp.hstack((x,y))),errs['interp'],errs['mpars'])

    def _orthproj(self,m):
        rcond = 10.*jnp.max(jnp.array(m.shape))*jnp.finfo(jnp.float64).eps
        u,s,_=jnp.linalg.svd(m)
        u = u[:len(s),:len(s)].dot(jnp.diag(s>rcond))
        return u.dot(u.T)

    def _colinproj(self,m):
        return jnp.eye(m.shape[0])-self._orthproj(m)

    def _get_pm_stats(self,errs,projs):

        errs_ = [self.mperr(edx,ex,pm_dx-ddx_dx,pm_dp) for edx,ex,pm_dx,pm_dp,ddx_dx in zip(errs['pm']['interp'],\
                                                                                            errs['interp'],\
                                                                                            projs['pm']['xs']['interp'],\
                                                                                            projs['pm']['mpars']['interp'],\
                                                                                            projs['dxs']['xs']['interp'])]
        sigmas_ = tree_map(jit(lambda x : _cov_oas(jnp.cov(x,rowvar=False),x.shape[0])[0]),errs_)
        mus_    = tree_map(jit(lambda x : jnp.mean(x,axis=0)),errs_)
        vars_    = tree_map(jit(lambda x : jnp.mean(x**2,axis=0)),errs_)
        return errs_, sigmas_, mus_, vars_
       
    def _upd_projs(self,params,xs_sm,tss,zn,reperror=False):
        ts, tsm = tss
        projs = {'pm':dict(),'dxs':dict()}
        projs['pm']['xs'] = {'pm':self.pred_pm_dx(params,xs_sm['pm']), 'interp':self.pred_pm_dx(params,xs_sm['interp'])}
        projs['pm']['mpars'] = {'pm':self.pred_pm_dp(params,xs_sm['pm']), 'interp':self.pred_pm_dp(params,xs_sm['interp'])}
        if reperror==True:
            projs['dxs']['xs'] = {'pm':self.pred_ddx_dx(params,tsm,zn),'interp':self.pred_ddx_dx(params,ts,zn)}
        else:
            projs['dxs']['xs'] = tree_map(lambda x : x*0, projs['pm']['xs'])
        return projs

    def _upd_omegas(self,xs_sm,sigmas,projs,mus,vars):
               
        omegas      = {'pm':{'r':[],'n':[]},'interp':{'r':[],'n':[]}}
        lcte        = {'pm':[],'interp':[]}
        if self.mode == 'forward':
            omegas['pm']['r'] = [self._cinv(self.quad(self.Ur.T,sigma)) for sigma in sigmas['pm']['pm']]
            omegas['pm']['n'] = [self._cinv(self.quad(self.Un.T,sigma)) for sigma in sigmas['pm']['pm']]
            lcte['pm'] = []
        elif self.mode == 'inverse':
            pm_dx = projs['pm']['xs']['pm']
            pm_dp = projs['pm']['mpars']['pm']
            ddx_dx = projs['dxs']['xs']['pm']
            # omegas['pm']['r'] = [self.cinv_v(cov_oas(self.quad_nv(self.Ur.T,\
            #                             self.quad_vn(pm_dx[i],sigmas['interp'][i]\
            #                                 +jnp.diag(jnp.abs(mus['interp'][i])))\
            #                             +self.quad_vn(pm_dp[i],sigmas['mpars'][i]\
            #                                 +jnp.diag(jnp.abs(mus['mpars'][i])))\
            #                             + jnp.diag(jnp.abs(mus['pm']['pm'][i]))\
            #                             + jnp.diag(jnp.abs(mus['interp'][i]))),len(pm_dp[i]))[0]) for i in range(len(sigmas['interp']))]  
            # omegas['pm']['r'] = [self.cinv_v(cov_oas(self.quad_nv(self.Ur.T,self.quad_vn(vmap(lambda x,y : jnp.hstack((x,y)), in_axes=0)(pm_dx[i],pm_dp[i]),\
            #             sigmas['full'][i]+jnp.diag(jnp.abs(jnp.concatenate((mus['interp'][i],mus['mpars'][i])))))+jnp.diag(jnp.abs(mus['pm']['pm'][i]))),len(pm_dp[i]))[0]) for i in range(len(sigmas['interp']))]
            # omegas['pm']['r'] = [self.cinv_v(cov_oas(self.quad_nv(self.Ur.T,self.quad_vn(vmap(lambda x,y : jnp.hstack((x,y)), in_axes=0)(-pm_dx[i]+ddx_dx[i],-pm_dp[i]),\
            #             sigmas['full'][i]+jnp.diag(jnp.abs(jnp.concatenate((mus['interp'][i],mus['mpars'][i])))))+0*sigmas['pm']['pm'][i]+0*(jnp.diag(jnp.abs(mus['interp'][i])+jnp.abs(mus['pm']['pm'][i])))),len(pm_dp[i]))[0]) for i in range(len(sigmas['interp']))]
            omegas['pm']['r'] = [self.cinv_v(cov_oas(self.quad_nv(self.Ur.T,self.quad_vn(-pm_dx[i]+ddx_dx[i],sigmas['interp'][i]+0*jnp.diag(jnp.abs(vars['interp'][i])))\
                    +self.quad_vn(-pm_dp[i],sigmas['mpars'][i]+0*jnp.diag(jnp.abs(vars['mpars'][i])))+\
                        jnp.diag(jnp.abs(mus['interp'][i]))),len(pm_dp[i]))[0]) for i in range(len(sigmas['interp']))]
            omegas['pm']['n'] = [self._cinv(self.quad(self.Un.T,sigma)) for sigma in sigmas['pm']['pm']]        
            # lcte['pm'] = [(self.Ur.shape[1]*jnp.log(2*jnp.pi)/2.+jnp.mean(self.flogdet(cov_oas(self.quad_nv(self.Ur.T,self.quad_vn(pm_dx[i],sigmas['interp'][i]+jnp.diag(jnp.abs(mus['interp'][i])))+\
            #                     self.quad_vn(pm_dp[i],sigmas['mpars'][i]+jnp.diag(jnp.abs(mus['mpars'][i])))),len(pm_dp[i]))[0]))/2.)/len(sigmas['interp']) for i in range(len(sigmas['interp']))]
            lcte['pm'] = [(self.Ur.shape[1]*jnp.log(2*jnp.pi)/2.+(self.flogdet(cov_oas(self.quad_nv(self.Ur.T,self.quad_vn(pm_dx[i],sigmas['interp'][i]+jnp.diag(jnp.abs(mus['interp'][i])))+\
                                self.quad_vn(pm_dp[i],sigmas['mpars'][i]+jnp.diag(jnp.abs(mus['mpars'][i])))),len(pm_dp[i]))[0]))/2.) for i in range(len(sigmas['interp']))]
            if self.inference:
                _sigmas = [_cov_oas(sigma[:self.nobs,:self.nobs]+jnp.diag(jnp.abs(mus[:self.nobs]))+jnp.diag(jnp.abs(dmus[:self.nobs]))\
                                     - sigma[self.nobs:,:self.nobs].T.dot(self.cinv(sigma[self.nobs:,self.nobs:]+jnp.diag(jnp.abs(mus[self.nobs:]))+jnp.diag(jnp.abs(dmus[self.nobs:]))),\
                                    precision=Precision.HIGHEST).dot(sigma[self.nobs:,:self.nobs],precision=Precision.HIGHEST),xs_sm['interp'][i].shape[0])[0] for i,(sigma, mus, dmus) in enumerate(zip(sigmas['interp'],mus['interp'],mus['pm']['pm']))]
                omegas['interp']['r'] = [self._cinv(cov_oas(jnp.array([self.quad(self.proj_r.T,sigma)]),sigma.shape[0])[0])[0] for sigma in _sigmas]
                omegas['interp']['n'] = [self._cinv(cov_oas(jnp.array([self.quad(self.proj_n.T,sigma)]),sigma.shape[0])[0])[0] for sigma in _sigmas]
                # lcte['interp'] = [self.proj_r.shape[1]*jnp.log(2*jnp.pi)/2.+\
                #                         jnp.mean(self.flogdet(cov_oas(jnp.array([self.quad(self.proj_r.T,sigma)]),sigma.shape[0])[0]))/2.\
                #                              for sigma in _sigmas]
                lcte['interp'] = [self.proj_r.shape[1]*jnp.log(2*jnp.pi)/2.+\
                                        (self.flogdet(cov_oas(jnp.array([self.quad(self.proj_r.T,sigma)]),sigma.shape[0])[0]))/2.\
                                             for sigma in _sigmas]
            else:
                omegas['interp']['r'] = [self._cinv(cov_oas(jnp.array([self.quad(self.proj_r.T,sigma+\
                                                jnp.diag(jnp.abs(mu)))]),sigma.shape[0])[0])[0]\
                                                    for sigma,mu,dmu in zip(sigmas['interp'],mus['interp'],mus['pm']['pm'])]
                omegas['interp']['n'] = [self._cinv(cov_oas(jnp.array([self.quad(self.proj_n.T,sigma+\
                                                jnp.diag(jnp.abs(mu)))]),sigma.shape[0])[0])[0] \
                                                    for sigma,mu,dmu in zip(sigmas['interp'],mus['interp'],mus['pm']['pm'])]
                # lcte['interp'] = [self.proj_r.shape[1]*jnp.log(2*jnp.pi)/2.+\
                #                         jnp.mean(self.flogdet(cov_oas(jnp.array([self.quad(self.proj_r.T,sigma)]),sigma.shape[0])[0]))/2. for\
                #                              sigma,mu in zip(sigmas['interp'],mus['interp'])]
                lcte['interp'] = [self.proj_r.shape[1]*jnp.log(2*jnp.pi)/2.+\
                                        (self.flogdet(cov_oas(jnp.array([self.quad(self.proj_r.T,sigma)]),sigma.shape[0])[0]))/2. for\
                                             sigma,mu in zip(sigmas['interp'],mus['interp'])]
        else:
            raise Exception
        
        return omegas, lcte

    def get_data_idx(self,data,idxs):
        return [[[] if len(_)==0 else _[sel,:] for _ in d] for sel,d in zip(idxs,data)]

    def _infer_dp(self,pm_dp,edx):
        dp    = self._cinv(self.avgqinv(pm_dp)).dot(self.avgproj(pm_dp,edx))
        return dp

    def _infer_exl(self,pm_dx,pm_dp, edx, ex):    
        dp    = self.infer_dp(pm_dp,edx) 
        exo   = ex[:,:self.nobs]
        gso   = pm_dx[:,:,:self.nobs]
        gsl   = jnp.dot(pm_dx[:,:,self.nobs:],self.Blr)
        Ainv  = self.vsubinv(gsl)
        b     = self.fb(gsl,edx,gso,exo,pm_dp,1.*dp.flatten())
        exl   = self.fiv(Ainv,b).dot(self.Blr.T)
        return exl, dp

    def _pred_sm(self,params,ts,zn):
        return [self.nn[i].batched_state(params['sm'][i],ts[i],zn[i]) for i in range(len(self.nn))]    

    def _pred_sm_dt(self,params,ts,zn):
        return [self.nn[i].diff_state(params['sm'][i],ts[i],zn[i])[:,:,0] for i in range(len(self.nn))]    

    def _pred_pm(self,params,xs):
        return [self.model.batched_eval(params['pm'],[xs[i]]) for i in range(len(self.nn))]
    
    def _pred_pm_dx(self,params,xs):
        return [self.model.diff_eval(params['pm'],[xs[i]])[0] for i in range(len(self.nn))]

    def _pred_ddx_dx(self,params,ts,zn):
        return [self.nn[i].ddxdx(params['sm'][i],ts[i],zn[i]) for i in range(len(self.nn))]    

    def _pred_pm_dp(self,params,xs):
        return [self.model.diff_params(params['pm'],[xs[i]])[0] for i in range(len(self.nn))]

    def _res_pm(self,params,ts,zn):
        xs_sm    = self._pred_sm(params,ts,zn)
        xs_sm_dt = self._pred_sm_dt(params,ts,zn)
        xs_pm_dt = self._pred_pm(params,xs_sm)
        return [delta(x_sm_dt,x_pm_dt) for x_sm_dt,x_pm_dt in zip(xs_sm_dt,xs_pm_dt)], xs_sm, xs_sm_dt, xs_pm_dt   

    def _get_commutator(self,params,xs,ts,zn):
        return [self.comm_v(self.nn[i].ddxpdx(params['sm'][i],ts[i],zn[i]),self.quad_nv(self.Ur.T,self.model.diff_eval(params['pm'],[xs[i]])[0])) for i in range(len(self.nn))]    

    def _res_fun(self,params,batch,zn):

        ts, xs, tsm = [[batch[i][j] for i in range(len(batch))] for j in range(3)]
        errs_pm, xs_sm, xs_sm_dt, xs_pm_dt = dict(), dict(), dict(), dict()

        #xs = [x*(jnp.concatenate((jnp.ones(self.nobs),params['scales']+self.sUn.dot(params['sm'][i][-1][0][:self.sUn.shape[1]])))) for i,x in enumerate(xs)]
        xs = [x*(jnp.concatenate((jnp.ones(self.nobs),params['scales']))) for i,x in enumerate(xs)]
        
        errs_pm['pm'], xs_sm['pm'], xs_sm_dt['pm'], xs_pm_dt['pm'] = self.res_pm(params,tsm,zn)
        errs = {'interp':[], 'pm': errs_pm}                 

        if self.mode == 'inverse':
            errs_pm['interp'], xs_sm['interp'], xs_sm_dt['interp'], xs_pm_dt['interp'] = self.res_pm(params,ts,zn)
            errs['interp']   = [delta(x_sm[:,:x.shape[1]],x) for x_sm,x in zip(xs_sm['interp'],xs)]
            #errs['commutator'] = [jnp.linalg.norm(_,axis=-1) for _ in self.get_commutator(params,xs,ts,zn)]
        return errs, xs_sm, xs_sm_dt, xs_pm_dt, ts, xs, tsm

    def _err_fun(self,errs,omegas):

        if self.mode == 'forward':
            mle_model      = [self.quad_vn(self.proj(self.Ur.T,errs['pm']['pm'][i]),omegas['pm']['r'][i])+self.quad_vn(self.proj(self.Un.T,errs['pm'][i]),omegas['pm']['n'][i]) for i in range(len(errs['pm']))]

        elif self.mode == 'inverse':
            mle_model = [self.quad_vv(self.proj(self.Ur.T,errs['pm']['pm'][i]),omegas['pm']['r'][i]) for i in range(len(errs['pm']['pm']))]
            mle_data  = [self.quad_vn(self.proj(self.proj_r.T,errs['interp'][i][:,:self.ndata]),omegas['interp']['r'][i]) for i in range(len(errs['interp']))]
        else:
            raise Exception()
        # sigmas = jnp.array(tree_map(jit(lambda x : jnp.cov(x,rowvar=False)),errs['interp']))
        # div = jnp.log(jnp.linalg.eigvalsh(sigmas.mean(axis=0))+1e-12).sum()-jnp.log(jnp.linalg.eigvalsh(sigmas)+1e-12).sum(axis=1).mean()
        return [jnp.array(_) for _ in [mle_model,mle_data]]#+[jnp.array([jnp.linalg.norm(_) for _ in errs['commutator']]).mean()]#+[1e-6*jnp.array([div])]

    def _hessian_fun(self,params,data,zn,reperror=False,mureg=False,center=True):
        errs, xs_sm, xs_sm_dt, xs_pm_dt, ts, xs, tsm = self.res_fun(params,data,zn) 
        if self.inference:
            errs['interp'] =  [jnp.hstack((exo,self.infer_exl(pm_dx,pm_dp, edx, exo)[0])) for pm_dx,pm_dp,edx,exo in\
                                    zip(self.pred_pm_dx(params,self.xs_sm['interp']),\
                                        self.pred_pm_dp(params,self.xs_sm['interp']),\
                                        errs['pm']['interp'],\
                                        errs['interp'])] 
        sigmas, mus, vars = self.get_stats(errs,center)
        if not mureg:
            mus = tree_map(lambda x : 0.*x,mus)
            vars = tree_map(lambda x : 0.*x,vars)
        projs = self.upd_projs(params,xs_sm,[ts,tsm],zn,reperror)
        errs['mpars'], sigmas['mpars'], mus['mpars'], vars['mpars'] = self.get_pm_stats(errs,projs)
        sigmas['full'] = self.get_sigmas_full(errs)
        omegas, lcte = self.upd_omegas(xs_sm,sigmas,projs,mus,vars)

        def _dummy_res_pm(params,ts,zn,invscale):
            xs_sm    = self._pred_sm(params,ts,zn)
            xs_sm_dt = [_*(jnp.concatenate((jnp.ones(self.nobs),invscale)))*(jnp.concatenate((jnp.ones(self.nobs),params['scales']))) for _ in self._pred_sm_dt(params,ts,zn)]
            xs_pm_dt = self._pred_pm(params,[_*(jnp.concatenate((jnp.ones(self.nobs),invscale)))*(jnp.concatenate((jnp.ones(self.nobs),params['scales'])))for _ in xs_sm])
            xs_sm = [_*(jnp.concatenate((jnp.ones(self.nobs),invscale)))*(jnp.concatenate((jnp.ones(self.nobs),params['scales'])))for _ in xs_sm]
            return [delta(x_sm_dt,x_pm_dt) for x_sm_dt,x_pm_dt in zip(xs_sm_dt,xs_pm_dt)], xs_sm, xs_sm_dt, xs_pm_dt
        
        def _dummy_res_fun(params,batch,zn,invscale):

            ts, xs, tsm = [[batch[i][j] for i in range(len(batch))] for j in range(3)]
            errs_pm, xs_sm, xs_sm_dt, xs_pm_dt = dict(), dict(), dict(), dict() 

            #xs = [x*(jnp.concatenate((jnp.ones(self.nobs),params['scales']+self.sUn.dot(params['sm'][i][-1][0][:self.sUn.shape[1]])))) for i,x in enumerate(xs)]
            xs = [x*(jnp.concatenate((jnp.ones(self.nobs),params['scales']))) for i,x in enumerate(xs)]
            
            errs_pm['pm'], xs_sm['pm'], xs_sm_dt['pm'], xs_pm_dt['pm'] = _dummy_res_pm(params,tsm,zn,invscale)
            errs = {'interp':[], 'pm': errs_pm}                 

            if self.mode == 'inverse':
                errs_pm['interp'], xs_sm['interp'], xs_sm_dt['interp'], xs_pm_dt['interp'] = self._res_pm(params,ts,zn)
                #errs_pm['interp'], xs_sm['interp'], xs_sm_dt['interp'], xs_pm_dt['interp'] = _dummy_res_pm(params,ts,zn,invscale)
                errs['interp']   = [delta(x_sm[:,:x.shape[1]],x) for x_sm,x in zip(xs_sm['interp'],xs)]

            return errs, xs_sm, xs_sm_dt, xs_pm_dt, ts, xs, tsm

        def _dummy_lkhd(params,omegas,lcte,data,invscale):
            zn = [(jnp.hstack((data[_][1][:,:self.nobs],\
                                         data[_][1][:,self.nobs:]*self.scales)).dot(self.nn[_].Un)).mean(axis=0) for _ in range(len(data))]
            errs, xs_sm, xs_sm_dt, xs_pm_dt, ts, xs, tsm = _dummy_res_fun(params,data,zn,invscale) 
            #
            # projs = self.upd_projs(params,xs_sm,[ts,tsm],zn,reperror)
            # errs['mpars'], sigmas['mpars'], mus['mpars'], vars['mpars'] = self.get_pm_stats(errs,projs)
            # sigmas['full'] = self.get_sigmas_full(errs)
            # omegas, lcte = self.upd_omegas(xs_sm,sigmas,projs,mus,vars)
            #
            mles = [jnp.mean(_) for _ in self.err_fun(errs,omegas)]
            lkhd =  [mles[0]/2.+jnp.mean(jnp.array(lcte['pm'])), mles[1]/2.+jnp.mean(jnp.array(lcte['interp']))]
            return lkhd

        return hessian(lambda pm_scales, sm, omegas, lcte, data, invscale: _dummy_lkhd({'pm':[pm_scales[:len(params['pm'][0])]],'sm':sm, 'scales':pm_scales[len(params['pm'][0]):]},omegas,lcte,data,invscale)[0],argnums=(0))\
                            (jnp.concatenate((params['pm'][0],params['scales'])),params['sm'],omegas,lcte,data,jnp.reciprocal(params['scales']))

    def _delta_domain_fun(self,params,data):
        runs = [list(_[:2]) for _ in data]
        ts   = [[_[2]] for _ in data]
        return grad(lambda params, runs, ts : -jnp.trace(self.hessian_fun(params,[run+t for run,t in zip(runs,ts)])[0][0]), argnums=(2))(params,runs,ts)

    def _sens_domain_fun(self,params,data):
        runs = [list(_[:2]) for _ in data]
        ts   = [[_[2]] for _ in data]
        return grad(lambda params, runs, ts : -jnp.linalg.norm(jnp.vstack(self.res_fun(params,[run+t for run,t in zip(runs,ts)])[0]['pm']['pm'])), argnums=(2))(params,runs,ts)

    def _lkhd_fun(self,params,data,zn,reperror=False,mureg=False,center=True):
        errs, xs_sm, xs_sm_dt, xs_pm_dt, ts, xs, tsm = self.res_fun(params,data,zn) 
        if self.inference:
            errs['interp'] =  [jnp.hstack((exo,self.infer_exl(pm_dx,pm_dp, edx, exo)[0])) for pm_dx,pm_dp,edx,exo in\
                                    zip(self.pred_pm_dx(params,self.xs_sm['interp']),\
                                        self.pred_pm_dp(params,self.xs_sm['interp']),\
                                        errs['pm']['interp'],\
                                        errs['interp'])] 
        sigmas, mus, vars = self.get_stats(errs,center)
        if not mureg:
            mus = tree_map(lambda x : 0.*x,mus)
        projs = self.upd_projs(params,xs_sm,[ts, tsm],zn,reperror)
        errs['mpars'], sigmas['mpars'], mus['mpars'], vars['mpars'] = self.get_pm_stats(errs,projs)
        sigmas['full'] = self.get_sigmas_full(errs)
        omegas, lcte = self.upd_omegas(xs_sm,sigmas,projs,mus,vars)
        mles = [jnp.mean(_) for _ in self.err_fun(errs,omegas)]
        return [mles[0]/2.+jnp.mean(jnp.array(lcte['pm'])), mles[1]/2.+jnp.mean(jnp.array(lcte['interp']))]

    def _smp_khd_fun(self,errs,xs_sm,params,sigmas,mus,vars,ts,tsm,zn,reperror=False,mureg=False):
        projs = self.upd_projs(params,xs_sm,[ts, tsm],zn,reperror)
        errs['mpars'], sigmas['mpars'], mus['mpars'], vars['mpars'] = self.get_pm_stats(errs,projs)
        sigmas['full'] = self.get_sigmas_full(errs)
        if not mureg:
            mus = tree_map(lambda x : 0.*x,mus)
        omegas, lcte = self.upd_omegas(xs_sm,sigmas,projs,mus,vars)
        mles = [jnp.mean(_) for _ in self.err_fun(errs,omegas)]
        return [mles[0]/2.+jnp.mean(jnp.array(lcte['pm'])), mles[1]/2.+jnp.mean(jnp.array(lcte['interp']))]
        
    def upd(self,params,data,zn,reperror=False,mureg=False,center=True):
        self.errs, self.xs_sm, self.xs_sm_dt, self.xs_pm_dt, self.ts, self.xs, self.tsm = self.res_fun(params,data,zn) 
        if self.inference:
            self.errs['interp'] =  [jnp.hstack((exo,self.infer_exl(pm_dx,pm_dp, edx, exo)[0])) for pm_dx,pm_dp,edx,exo in\
                                    zip(self.pred_pm_dx(params,self.xs_sm['interp']),\
                                        self.pred_pm_dp(params,self.xs_sm['interp']),\
                                        self.errs['pm']['interp'],\
                                        self.errs['interp'])] 
        self.sigmas, self.mus, self.vars = self.get_stats(self.errs,center)
        if not mureg:
            self.mus = tree_map(lambda x : 0.*x,self.mus)
        if self.inference:
            self.sigmas['interp'] = [jnp.mean(jnp.array(self.sigmas['interp']),axis=0) for _ in range(len(self.sigmas['interp']))]
        if self.mode == 'inverse':
            self.projs = self.upd_projs(params,self.xs_sm,[self.ts, self.tsm],self.zn,reperror)
            self.errs['mpars'], self.sigmas['mpars'], self.mus['mpars'], self.vars['mpars'] = self.get_pm_stats(self.errs,self.projs)
        self.sigmas['full'] = self.get_sigmas_full(self.errs)
        self.omegas, self.lcte = self.upd_omegas(self.xs_sm,self.sigmas,self.projs,self.mus, self.vars)

    # AL
    # def _ts_step(self, ts, opt_state, args):
    #     params, runs, omegas = args
    #     #loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    #     value, grads = value_and_grad(lambda params, runs, ts : jnp.var(jnp.vstack(self.err_fun(self.res_fun(params,[run+t for run,t in zip(runs,ts)])[0],omegas)[0])), argnums=(2))(params,runs,ts)
    #     updates, opt_state = self.ts_optimizer.update(grads, opt_state, ts)
    #     ts = optax.apply_updates(ts, updates)
    #     return ts, opt_state, grads, value
    #

    def train(self, data, frac=1., alpha=1, beta= 0., mu=0., extfuns=[],reperror=False,mureg=True,center=False):

        self.alpha = alpha 
        self.mu  = mu
        
        if not self.__isinitialized__:
            if all([_[1].shape[1] == len(self.Ur) for _ in data]):
                self.params['scales'] = self.get_scales(data)
            else:
                self.params['scales'] = jnp.empty(0)
                self.inference = True
            self.initialize()            
            self.subproj = jnp.eye(len(self.params['pm'][0]))
            if self.ndata == self.nobs and self.ndata==len(self.model.M): # homogeneous (set nns nullspace)
                for _,nn in enumerate(self.nn):
                    nn.set_nullspace(jnp.mean(data[_][1].dot(nn.Un),axis=0))
                    if any(jnp.var(data[_][1].dot(nn.Un),axis=0)/jnp.mean(data[_][1].dot(nn.Un),axis=0)>0.05):
                        raise Exception('Nullspace error for homogeneous reaction.')

            # AL
            # self.ts_optimizer = optax.adam(learning_rate=1e-5)
            # self.ts_opt_state = self.ts_optimizer.init([[_[2]] for _ in data])
            #

            self.upd(self.params,data,self.zn,reperror=reperror,mureg=mureg,center=center)

            self.__isinitialized__ = True
        elif not self.__hasoptimizer__:
            raise Exception('Optimizer not defined.')
        else:
            pass
      
        self.error = jnp.inf
        self.rerror = jnp.inf
        self._epoch = 0

        if not self.best_params['params']:
            self.best_params['error']  = jnp.inf
            self.best_params['params'] = self.params.copy()
        
        while self._epoch < self.num_epochs and self.tol < self.error and self.rtol < self.rerror:
            clear_output(wait=True)
            self.batch = data

            for i in range(int(self.num_iter)):
                #next(itercount)
                self.params, self.opt_state = self.step(self.opt_state, self.params, self.omegas, self.batch, self.zn, self.subproj) 
            #self.e, self.p=jnp.linalg.eigh(self.hessian_fun(self.params,self.batch)[0][0])
            #self.subproj = self.p[:,self.e<0]
            #self.subproj = self.p.dot(jnp.diag(jnp.reciprocal(self.e*(self.e>0)+1e-12))).dot(self.p.T)

                # AL                
                # runs = [list(_[:2]) for _ in self.batch]
                # ts   = [[_[2]] for _ in self.batch]
                # ts, self.ts_opt_state, grads, value = self.ts_step(ts, self.ts_opt_state, [self.params, runs, self.omegas])
                # self.batch = [run+[jnp.sort(t[0],axis=0)] for run,t in zip(runs,ts)]
                #
            
            self._epoch += 1
            self.epoch  += 1
            
            loss_it_sample              = self.loss(self.params, self.omegas, data, self.zn)
            loss_it_test                = self.loss(self.params, self.omegas, data, self.zn)
            self.rerror = self.error
            self.error = loss_it_batch  = self.loss(self.params, self.omegas, data, self.zn)
            self.rerror = self.error if jnp.isposinf(self.rerror) else jnp.abs(2*jnp.divide(self.error-self.rerror,self.error+self.rerror))
            self.upd(self.params,data,self.zn,reperror=reperror,mureg=mureg,center=center)
            self.errors = [jnp.mean(_) for _ in self.err_fun(self.errs,self.omegas)]
            print('Epoch: {:4d}, Loss Batch: {:.5e}, Loss Data: {:.5e} Loss CV: {:.5e}'.format(self.epoch,loss_it_sample,loss_it_batch, loss_it_test)+\
                     ''.join([', Fit {}: {:.5e}'.format(_,__) for _,__ in zip(self.err_tags,self.errors)]))

            if self.__historian__:
                self.lkhd = self.lkhd_fun(self.params,data,self.zn)
                self.iter_data  += [[self.epoch,[self.params.copy()],\
                                     [self.omegas.copy()],[self.errors.copy(),self.lkhd.copy()],\
                                        [repr(self.optimizer),self.opt_dict],\
                                      [self.sigmas.copy(),self.mus.copy()],\
                                        [self.errs.copy()],[self.xs_sm.copy()]]]
        
            if jnp.linalg.norm(jnp.array([jnp.mean(_) for _ in self.errors]))<self.best_params['error']:
                self.best_params['error']  = jnp.linalg.norm(jnp.array([jnp.mean(_) for _ in self.errors]))
                self.best_params['params'] = self.params.copy()
        
        self.send_parameters()
        return self.batch

    def get_error_metrics(self, batches, ground_truth=True):
    
        metrics = dict()
            
        sc = jnp.concatenate((jnp.ones(self.nobs),self.params['scales']))
        
        metrics['model_pars']  = self.params['pm']
        metrics['model_scale'] = sc


        errs, xs_sm, xs_sm_dt, xs_pm_dt, ts, xs, tsm = self.res_fun(self.params,batches,self.zn) 

        #if ground_truth and self.scale:
        xs = [x*jnp.reciprocal(sc) for x in xs]
        
        self.source = source = list(zip(xs_sm_dt['pm'], xs_pm_dt['pm']))
        labels = ['diff_obs'] 
        if self.mode=='inverse':
           # if self.scale and not ground_truth:
            source = list(zip(source,[(x_sm, x*sc) for (x_sm,x) in zip(xs_sm['interp'],xs)]))
            #else:
             #   source = list(zip(source,[(x_sm, x) for (x_sm,x) in zip(xs_sm['interp'],xs)]))
            labels += ['state_obs']
        else:
            pass
        if self.nobs!=self.model.M.shape[0]:
            labels += [_.split('_')[0]+'_latent' for _ in labels]
            source = [[[_[:,:self.nobs] for _ in s] for s in source[i]] for i in range(len(source))]+\
                                [[[_[:,self.nobs:] for _ in  s] for s in source[i]] for i in range(len(source))]
        datas = [list(zip(labels,s)) for s in source]
        metrics = [{'metrics': {tag : {_:__metrics__[_](v[1],v[0]) for _ in __metrics__.keys()} 
                        for tag,v in data}} for data in datas]
        for i,(x_sm,x_sm_dt,x_pm_dt,t,x) in enumerate(zip(xs_sm['interp'],xs_sm_dt['pm'],xs_pm_dt['pm'],ts,xs)):
            metrics[i]['raw_data'] = {'state': (x_sm,x),
                                    'diff' : (x_sm_dt, x_pm_dt)}   
            metrics[i]['data'] = dict(zip(labels,source[i]))
            metrics[i]['data']['t'] = t
            metrics[i]['raw_data']['t'] = t
        return metrics






