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

from . import jnp, random, jit, itertools, clear_output, grad, jacfwd, hessian, pmap
from jax.nn import softplus


class TrainerCV(object):
    
    def __init__(self, nn, model, num_iter=1e2, num_epochs=1e2, batch_size=0.99, split=0.99, tol=0., \
                 verbose=False, mode='forward', bc = None, scale = False, historian = False, iter_data = [None], err_tags=[], nobs = None, alpha = 1, beta = 0):
        if not hasattr(nn,'__iter__'):
            nn = [nn] # making nn subclass"able"
        self.nn    = nn
        self.model = model
        self.mode  = mode
        self.bc    = bc
        if mode == 'forward':
            self.params = [[_.params for _ in self.nn]]
            self.sparams = [self.model.params] # pseudostatic params
        elif mode == 'inverse':
            self.params  = [[_.params for _ in self.nn], [jnp.log(jnp.abs(self.model.params[0]))]]
            self.sparams = [] # pseudostatic params 
            if scale:
                if isinstance(nobs,int) and nobs<self.model.M.shape[0]:
                    self.params = self.params + [jnp.hstack((jnp.ones(self.model.M.shape[0]-nobs)[:,jnp.newaxis]))]
                else:
                    scale = False # do not scale observable variables
        else: 
            raise Exception("mode parameter must be either 'forward' or 'inverse'")
        
        self.alpha       = alpha
        self.beta        = beta
        self.num_iter    = int(num_iter)
        self.num_epochs  = int(num_epochs)
        self.tol         = tol
        self.batch_size  = batch_size
        self.split       = split
        self.verbose     = verbose
        self.loss        = jit(self._loss)
        self.step        = jit(self._step)
        self.scale       = scale
        self.nobs        = nobs
        self.best_params = {'error':jnp.inf,'params':[]}
        self.errors      = [0.]
        self.epoch       = 0
        self.images      = []
        
        self.__historian__     = historian
        self.iter_data         = iter_data
        self.__hasoptimizer__  = False
        self.__isinitialized__ = False
        self.err_tags          = err_tags
        self.save_tags         = ['params','iter_data','best_params','error','errors','epoch','err_tags','scale','nobs','alpha']
        
        if hasattr(self,'_err_fun'):
            pass
        else:
            raise Exception('A trainer class must be defined as a subclass of the Trainer superclass'\
                            +' and include an _err_fun(params,batch).')
            
    def _sparams(self):
        if self.mode == 'forward':
            sparams = [self.model.params] # pseudostatic params
        elif self.mode == 'inverse':
            sparams = {'error':self.errors}# pseudostatic params 
        return sparams
    
    def _rescale(self, errors):
        # if errors need to be retransformed.
        return errors

    def _loss(self, params, sparams, batch):
        return jnp.array([_.mean() for _ in self.err_fun(params, sparams, batch)]).sum()
    
    def set_optimizer(self, optimizer, kwargs):
        self._opt_kwargs = kwargs
        self._opt        = optimizer
        for _0, _1 in zip(['_opt_init', '_opt_update', '_get_params'],self._opt(**self._opt_kwargs)):
            self.__setattr__(_0,_1)
        self.__hasoptimizer__ = True
            
    def _step(self, i, opt_state, sparams, batch):
        params = self._get_params(opt_state)
        return self._opt_update(i, self.grads(params, sparams, batch), opt_state)
    
    def send_parameters(self):
        if  self.mode == 'inverse':
            if self.scale:
                nn_par, model_params_, scale = self.params
            else:
                nn_par, model_params_ = self.params
            self.model.params = [jnp.exp(model_params_[0])]
            for i, _ in enumerate(nn_par):
                self.nn[i].set_params(_)
        else:
            for i, _ in enumerate(self.params[0]):
                self.nn[i].set_params(_)
    
    def bind_hessian(self,data=None):
        if self.scale:
            self.hessian = lambda data : self.hessian_model(self.params[0], \
                                            self.params[1], self.params[2], self._sparams(), data)
        else:
            self.hessian = lambda data : self.hessian_model(self.params[0], \
                                            self.params[1], self._sparams(), data)
        if data:
            print(self.loss(self.params, self._sparams(), data ))
            self.eval_hessian = lambda : self.hessian(data)

    def reinit(self,nn=False):
        if nn:
            self.nn.init()
            self.params[0] = [_.params for _ in self.nn]
        self._opt_state = self._opt_init(self.params)
        self.best_params = {'error':jnp.inf,'params':[]}
    
    def rejit(self,bind_hessian=True):
        # 're'JIT in case static memory maps have changed
        self.err_fun = jit(lambda params, sparams, batch : self._err_fun(params, sparams, batch))
        self.loss = jit(self._loss) 
        f = lambda params, sparams, batch : jnp.nan_to_num(self.loss(params, sparams, batch))
        self.grads = jit(grad(f, argnums=0))
        self.step = jit(self._step)
        if hessian:
            if self.scale:
                g = lambda nn_params, model_params, scale, sparams, batch : f([nn_params, model_params, scale], sparams, batch) 
            else:
                g = lambda nn_params, model_params, sparams, batch : f([nn_params, model_params], sparams, batch) 
            self.hessian_model = jit(hessian(g, argnums=1))
            self.bind_hessian()
    
    def initialize(self):
        if self.__hasoptimizer__:
            self._opt_state = self._opt_init(self.params)
            self.rejit()
        else:
            raise Exception('Optimizer not defined.')
            
    def dump(self,filename):
        jnp.savez(filename,\
                 data={tag:getattr(self,tag) for tag in self.save_tags})
    
    def load(self,filename):
        objs = jnp.load(filename, allow_pickle=True)['data'].tolist()
        for tag in self.save_tags:
            try:
                setattr(self,tag,objs[tag])
            except:
                print('Load error: missing attribute {}'.format(tag))
        if self.__hasoptimizer__:
            self.initialize()
        self.send_parameters()
        
    def get_state(self,state):
        iter_data = self.iter_data[state]
        self.iter_data += [[self.epoch,[self.params.copy()],[self._sparams()],[self.errors.copy()],[repr(self._opt),self._opt_kwargs]]]
        self.epoch      = iter_data[0]
        self.params     = iter_data[1][0]
        self.errors     = iter_data[3][0]
        self.iter_data += [[self.epoch,[self.params.copy()],[self._sparams()],[self.errors.copy()],[repr(self._opt),self._opt_kwargs]]]
        
    def train(self, data, frac=1., shuffle=False, alpha=1, beta= 0., extfuns=[]):
        
        if not self.__isinitialized__:
            self.initialize()
            self.__isinitialized__ = True
        elif not self.__hasoptimizer__:
            raise Exception('Optimizer not defined.')
        else:
            pass
        self.alpha = alpha 
        self.beta  = beta
        
        if self.mode == 'inverse':
            data = [[_[:int(jnp.floor(d[0].shape[0]*frac)),:] for _ in d] for d in data]
        else:  
            data = [[_[:int(jnp.floor(d[0].shape[0]*frac))] for _ in d] for d in data]
        
        params = self._get_params(self._opt_state)
        itercount = itertools.count()        
        if not self.batch_size:
            self.batch_size = 1.
        self.error = jnp.inf
        self._epoch = 0
        log_error = 0.
        while self._epoch < self.num_epochs and self.tol < self.error:
            clear_output(wait=True)
            batch = []
            if self.batch_size<1:
                for d in data:
                    sel = random.shuffle(random.PRNGKey(self.epoch),jnp.arange(int(jnp.floor(d[0].shape[0])))) # random shuffle
                    batch += [[_[sel[:int(jnp.floor(d[0].shape[0]*self.batch_size))],:] for _ in d]]
            else:
                batch = data
            
            if self.split < 1:
                batch_train = []
                batch_test  = []
                for d in batch:
                    batch_train += [[_[:int(jnp.floor(_.shape[0]*self.split)),:] for _ in d]]
                    batch_test  += [[_[int(jnp.floor(_.shape[0]*self.split)):,:] for _ in d]]                
            else:
                batch_train = batch_test = batch
            
            for i in range(int(self.num_iter)):
                self._opt_state = self.step(next(itercount), self._opt_state, self._sparams(), batch_train) 
                
            self.params = self._get_params(self._opt_state) 
            
            loss_it_sample              = self.loss(self.params, self._sparams(), batch_train)    
            loss_it_test                = self.loss(self.params, self._sparams(), batch_test)    
            self.error = loss_it_batch  = self.loss(self.params, self._sparams(), data )
            self.errors = self._rescale([_.mean() for _ in self.err_fun(self.params, self._sparams(), data)],self._sparams())
            print('Epoch: {:4d}, Loss Batch: {:.5e}, Loss Data: {:.5e} Loss CV: {:.5e}'.format(self.epoch,loss_it_sample,loss_it_batch, loss_it_test)+\
                     ''.join([', Fit {}: {:.5e}'.format(_,__) for _,__ in zip(self.err_tags,self.errors)]))
            self._epoch += 1
            self.epoch  += 1
            if jnp.linalg.norm([jnp.log(_) for _ in self.errors])>log_error:
                self.best_params['error']  = self.errors
                self.best_params['params'] = self.params.copy()
        if self.__historian__:
            self.images     += [self.epoch]
            self.iter_data  += [[self.epoch,[self.best_params['params'].copy()],\
                                 [self._sparams()],[self.best_params['error']],[repr(self._opt),self._opt_kwargs]]]
        self.bind_hessian(data)
        self.send_parameters()

