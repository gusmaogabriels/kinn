from kinn.basis.trainer import TrainerCV
from kinn.basis.nnx import nn
from kinn.basis import jit, jnp, jacfwd, vmap
from jax.nn import relu
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

#@jit
def r2_score(y,ypred):
    y_, ypred_   = [(_-_.mean(axis=0).reshape(1,-1))/_.std(axis=0).reshape(1,-1) for _ in [y,ypred]]
    return jnp.mean(jnp.diag(jnp.dot(y_.T,ypred_)*jnp.reciprocal(len(y_)))**2)    


__metrics__ = {'r2' : r2_score,
              'MAE' : mean_absolute_error,
              'MSE' : mean_squared_error,
              'TotalVar': explained_variance_score}

@jit
def mrse(y,yhat):#,mode=0):
    #if mode == 0:
    return jnp.mean((y-yhat)**2,axis=1)
    #elif mode == 1:
    #return jnp.mean((y-yhat)**2/(1+(y+yhat)**2),axis=1)
    #else:
    #    return 0.
    
class trainer(TrainerCV):
    
    def _sparams(self):
        if self.mode == 'forward':
            sparams = [self.model.params] # pseudostatic params
        elif self.mode == 'inverse':
            sparams = [self.alpha] # pseudostatic params 
        return sparams
    
    def get_error_metrics(self,batches, ground_truth=True):
        
        metrics = dict()
        
        nns_params, model_params, scales = self._get_pars(self.params)
        
        mpars = model_params[0].copy()
        if self.scale:
            #sc = jnp.exp(jnp.concatenate((jnp.zeros(self.nobs),scales)))
            sc = jnp.concatenate((jnp.ones(self.nobs),scales))
            #fac = self.model.f([[],sc])
        else:
            sc  = jnp.ones(self.model.M.shape[0])
        
        metrics['model_pars']  = mpars
        metrics['model_scale'] = sc
                   
        for i, (nn_params, batch) in enumerate(zip(nns_params,  batches)):
            
            t, (x_sm, x), (x_sm_dt, x_pm_dt) = self._get_state(batch, nn_params, model_params, scales, i)
            
            if ground_truth and self.scale:
                x = x*jnp.reciprocal(sc)
            
            source = [(x_sm_dt, x_pm_dt)]
            labels = ['diff_obs'] 
            if self.mode=='inverse':
                if self.scale and not ground_truth:
                    source += [(x_sm, x*sc)]
                else:
                    source += [(x_sm, x)]
                labels += ['state_obs']
            else:
                pass
            if self.nobs==self.model.M.shape[0]:
                data = zip(labels,source) 
            else:
                labels += [_.split('_')[0]+'_latent' for _ in labels]
                source = [[_[:,:self.nobs] for _ in source[i]] for i in range(len(source))]+\
                                   [[_[:,self.nobs:] for _ in source[i]] for i in range(len(source))]
                data = zip(labels,source) 
            metrics[i] = {'metrics': {tag : {_:__metrics__[_](v[1],v[0]) for _ in __metrics__.keys()} 
                          for tag,v in data}}
            metrics[i]['raw_data'] = {'state': (x_sm,x),
                               'diff' : (x_sm_dt, x_pm_dt)}   
            metrics[i]['data'] = dict(zip(labels,source))
            metrics[i]['data']['t'] = t
            metrics[i]['raw_data']['t'] = t
            print(labels)
        return metrics
        
    def _get_pars(self,params):
        scales = None
        if self.mode == 'inverse':
            if self.scale == True:
                nns_params,  model_params_, scales_ = params # <- if inverse
                scales = jnp.exp(scales_)
            else:
                nns_params,  model_params_ = params # <- if inverse
            model_params = [jnp.exp(model_params_[0])]
        elif self.mode == 'forward':
            nns_params, = params # <-  if forward
            model_params = self.model.params
        else:
            raise Exception('mode not supported {}.'.format(self.mode))
        return nns_params, model_params, scales
    
    def _get_state(self, batch, nn_params, model_params, scales, i):
        t, x = batch
        nn = self.nn[i]
        batched_state = nn.batched_state
        diff_state    = nn.diff_state
        batched_model_latent = self.model.batched_eval
        
        if self.scale and self.nobs:
            #x_    = x*jnp.exp(jnp.concatenate((jnp.zeros(self.nobs),scales)))
            x_    = x*jnp.concatenate((jnp.ones(self.nobs),scales))
        else:
            x_    = x

        x_sm = nn.batched_state(nn_params[0],t)

        x_pm_dt = self.model.batched_eval(model_params,[t,x_sm])
        x_sm_dt = nn.diff_state(nn_params[0],t)[:,:,0]
        
        return t, (x_sm, x_), (x_sm_dt, x_pm_dt)
        
    def _rescale(self,errors, sparams):
        if self.mode == 'forward':
            return errors
        elif self.mode == 'inverse':
            alpha,     = sparams
            return [errors[0], errors[1]/alpha]
        else:
            return []
    
    def _err_fun(self, params, sparams, batches):
        
        nns_params, model_params, scales = self._get_pars(params)
            
        err_model, err_data = [0.]*2
        
        for i, (nn_params, batch) in enumerate(zip(nns_params,  batches)):
            
            t, (x_sm, x), (x_sm_dt, x_pm_dt) = self._get_state(batch, nn_params, model_params, scales, i)

            if self.mode == 'forward':
                err_model += mrse(x_sm_dt,x_pm_dt)
            elif self.mode == 'inverse':
                alpha,     = sparams
                err_model += mrse(x_sm_dt,x_pm_dt)#,mode=1)
                err_data  += alpha*mrse(x_sm[:,:x.shape[1]],x)#,mode=1)
            else:
                raise Exception()
            
        return [err_model*jnp.reciprocal(len(batches)), err_data*jnp.reciprocal(len(batches))]
    
class nn_c(nn):
    
    def __init__(self,*args,**kwargs):
        self.bc = kwargs.pop('bc')
        super().__init__(*args,**kwargs)
    
    def constraints(self,x,t):
        t0, x0 = self.bc
        return x0+(x-x0)*(t-t0)#*jnp.tanh((t-t0))

class nn_cn(nn):
    
    def constraints(self,x,t):
        x_cov_ = jnp.sin(jnp.concatenate((x,jnp.array([0.]))))**2
        x_cov  = 1.-x_cov_
        for _ in range(len(x_cov)):
            x_cov *= jnp.concatenate((jnp.array([1.]*(_+1)),x_cov_[:-(_+1)]))
        return x_cov

class nn_cn_bc(nn_c):
    
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        ang0 = []
        for _ in range(len(self.bc[1])-1):
            den = jnp.prod([jnp.sin(ang0[ix])**2. for ix in range(_)])
            if den != 0.:
                ang0 += [jnp.arcsin(jnp.sqrt(1.-self.bc[1][_]/den))]
            else:
                ang0 += [jnp.array(jnp.pi/2.)]
        self.bc_ang = [self.bc[0],jnp.array(ang0)]
    
    def constraints(self,x,t):
        t0, x0 = self.bc_ang
        x      = x0+jnp.tanh((t-t0))*(x-x0)
        x_cov_ = jnp.sin(jnp.concatenate((x,jnp.array([0.]))))**2
        x_cov  = 1.-x_cov_
        for _ in range(len(x_cov)):
            x_cov *= jnp.concatenate((jnp.array([1.]*(_+1)),x_cov_[:-(_+1)]))
        return x_cov
    
def normtrig(x,t):
    n = len(x)
    y = jnp.sin(x)**2.
    z = jnp.prod((jnp.ones([n]*2)-jnp.triu(jnp.ones([n]*2))).dot(jnp.diag(y))+jnp.diag(-y)+jnp.triu(jnp.ones([n]*2)),axis=1)
    return jnp.concatenate((z,jnp.array([jnp.prod(y)]))) 

class nn_combo(object):
    
    def __init__(self, nns, pmap=False, bc=None, nobs=None, trig=False, damp=False, gain=1.,mode=None):
        self.nns = nns
        self.bc  = bc
        self.params = list(zip(*[_.params for _ in nns]))
        if pmap :
            self.batched_state = jit(pmap(self.state, in_axes=(None,0)))
        else:
            self.batched_state = jit(vmap(self.state, in_axes=(None,0)))
        self.diff_state    = jit(lambda params,t:vmap(jacfwd(self.state,argnums=(1)),in_axes=(None,0))(params,t))
        self.nobs = nobs
        self.trig = trig
        self.damp = damp
        self.gain = gain
        self.mode = mode
        #self.diff_state2   = jit(lambda params,t:self._diff_state2(self.batched_state,params,t))
        
    def set_params(self,params):
        for i, nn in enumerate(self.nns):
            nn.set_params([param[i] for param in params])
        
    def normtrig(self,x,t):
        n = len(x)
        y = jnp.sin(jnp.tanh(x)*jnp.pi*3./4.+jnp.pi/4.)**2.
        z = jnp.prod((jnp.ones([n]*2)-jnp.triu(jnp.ones([n]*2))).dot(jnp.diag(y))+jnp.diag(-y)+jnp.triu(jnp.ones([n]*2)),axis=1)
        return jnp.concatenate((z,jnp.array([jnp.prod(y)]))) 
        
    def __call__(self,t,nn):
        return self.batched_state(self.params[nn],t)

    def state(self,params,t):
        if self.mode == 'forward':
            t0, x0 = self.bc
        else:
            x0 = 0.
        #dx    = jnp.concatenate([self.nns[i].state(params[i],t) for i in range(len(params))])-x0
        out = self.nns[0].state(params[0],t)
        if self.trig and len(out)>self.nobs:
            dx    = jnp.concatenate((out[:self.nobs],self.normtrig(out[self.nobs:],t)))-x0
        else:
            dx    = out-x0
        #return dx*jnp.tanh(t-t0)+x0
        if self.mode == 'forward':
            if self.damp:
                xt = self.nns[1].state(params[1],t)
            else:
                xt = 0.
            x = dx*jnp.tanh((t-t0)*self.gain*jnp.exp(xt))+x0
        elif self.mode == 'inverse':
            x = dx
        else:
             raise Exception('mode not implemented ({})'.format(self.mode))
        return x
    
# useful activation functions

swish      = jit(lambda x : x/(1.+jnp.exp(-x)))
relu       = jit(lambda x : x*(x>0))
tanh       = jit(lambda x : jnp.tanh(x))
sigmoid    = jit(lambda x : jnp.reciprocal(1.+jnp.exp(-x)))
gauss      = jit(lambda x : jnp.exp(-x**2))
sin        = jit(lambda x : jnp.sin(x))
cos        = jit(lambda x : jnp.cos(x))
    
pars = {i:dict() for i in range(4)}

# MKM Gas Phase ICs

t0                  = 0.
#bcv                 = [[0.6,0.4,0.0],[0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9]]
bcv                 = [[0.6,0.4,0.0],[0.2,0.3,0.5]]
trig                = True

### MKM parameters

pars[0]['type']     = 'homogeneous'
pars[0]['sps']      = ['A','B','C']
pars[0]['stoich']   = jnp.array([[-1,-1, 1],
                                [ 1, 1,-1]]).T
pars[0]['kijnpars']  = jnp.array([10.,1.])
pars[0]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_]] for bcv_ in bcv]
pars[0]['nncpars']  = [{'forward': {'trig':False, 'damp':True,'nobs':3,'bc':bc_},\
                       'inverse': {'trig':False, 'damp':False,'nobs':3,'bc':bc_}} for bc_ in pars[0]['bc'] ]
pars[0]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,5,5,5,pars[0]['stoich'].shape[0]]],
                       'act_fun'                   : [tanh,swish,tanh],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,5,5,5,pars[0]['stoich'].shape[0]-trig*0]],
                       'act_fun'                   : [tanh,swish,tanh],
                       'nn_scale'                  : 0.001}}
pars[0]['nntpars']  = {'layers_sizes' : [[1,3,1]],
                       'act_fun'      : [swish,tanh,swish,tanh],
                       'nn_scale'     : 0.001}
pars[0]['kijnpars_tags'] = ['k_{1}^g','k_{\text{-}1}^g']

pars[1]['type']     = 'heterogeneous'
pars[1]['sps']      = ['A','B','C','A*','B*','C*','*']
pars[1]['stoich']   = jnp.array([[-1, 0, 0, 1, 0, 0,-1],
                                [ 1, 0, 0,-1, 0, 0, 1],
                                [ 0,-1, 0, 0, 1, 0,-1],
                                [ 0, 1, 0, 0,-1, 0, 1],
                                [ 0, 0,-1, 0, 0, 1,-1],
                                [ 0, 0, 1, 0, 0,-1, 1],
                                [ 0, 0, 0,-1,-1, 1, 1],
                                [ 0, 0, 0, 1, 1,-1,-1]]).T
pars[1]['kijnpars']  = jnp.array([.5,.2,2.,3.,10.,2.,5.,4])*20.
pars[1]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_+[0.]*(pars[1]['stoich'].shape[0]-1-len(bcv_))+[1.]]] for bcv_ in bcv]
pars[1]['nncpars']  = [{'forward': {'trig':False, 'damp':True,'nobs':3,'bc':bc_},\
                       'inverse': {'trig':True, 'damp':False,'nobs':3,'bc':bc_}}  for bc_ in pars[1]['bc']]
pars[1]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,10,10,10,pars[1]['stoich'].shape[0]]],
                       'act_fun'                   : [tanh,swish,tanh],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,10,10,10,pars[1]['stoich'].shape[0]-1]],
                       'act_fun'                   : [tanh,swish,tanh,tanh,swish],
                       'nn_scale'                  : 0.001}}
pars[1]['nntpars']  = {'layers_sizes' : [[1,3,1]],
                       'act_fun'      : [swish,tanh,swish,tanh],
                       'nn_scale'     : 0.001}
pars[1]['kijnpars_tags'] = ['k_{1}^d','k_{\text{-}1}^d','k_{2}^d','k_{\text{-}2}^d','k_{3}^d','k_{\text{-}3}^d']+\
                              ['k_{1}^a','k_{\text{-}1}^a']

pars[2]['type']     = 'heterogeneous'
pars[2]['sps']      = ['A','B','C','A*','B*','C*','D*','*']
pars[2]['stoich']   = jnp.array([[-1, 0, 0, 1, 0, 0, 0,-1],
                                [ 1, 0, 0,-1, 0, 0, 0, 1],
                                [ 0,-1, 0, 0, 1, 0, 0,-1],
                                [ 0, 1, 0, 0,-1, 0, 0, 1],
                                [ 0, 0,-1, 0, 0, 1, 0,-1],
                                [ 0, 0, 1, 0, 0,-1, 0, 1],
                                [ 0, 0, 0, 0,-1, 0, 2,-1],
                                [ 0, 0, 0, 0, 1, 0,-2, 1],
                                [ 0, 0, 0,-1, 0, 1,-1, 1],
                                [ 0, 0, 0, 1, 0,-1, 1,-1]]).T
pars[2]['kijnpars']  = jnp.array([.5,.2,.4,.1,.3,.2,30.,10.,50.,40.])*40.
pars[2]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_+[0.]*(pars[2]['stoich'].shape[0]-1-len(bcv_))+[1.]]] for bcv_ in bcv]
pars[2]['nncpars']  = [{'forward': {'trig':False, 'damp':True,'nobs':3,'bc':bc_},\
                       'inverse': {'trig':True, 'damp':False,'nobs':3,'bc':bc_}} for bc_ in pars[2]['bc']]
pars[2]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,16,16,16,pars[2]['stoich'].shape[0]]],
                       'act_fun'                   : [tanh,swish,tanh],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,14,14,14,pars[2]['stoich'].shape[0]-1]],
                       'act_fun'                   : [tanh,swish,tanh],
                       'nn_scale'                  : 0.001}}
pars[2]['nntpars']  = {'layers_sizes' : [[1,6,1]],
                       'act_fun'      : [swish,tanh,swish,tanh],
                       'nn_scale'     : 0.001}
pars[2]['kijnpars_tags'] = ['k_{1}^d','k_{\text{-}1}^d','k_{2}^d','k_{\text{-}2}^d','k_{3}^d','k_{\text{-}3}^d']+\
                              ['k_{1}^c','k_{\text{-}1}^c','k_{2}^c','k_{\text{-}2}^c']

pars[3]['type']     = 'heterogeneous'
pars[3]['sps']      = ['A','B','C','A*','B*','C*','D*','E*','F*','*']
pars[3]['stoich']   = jnp.array([[-1, 0, 0, 1, 0, 0, 0, 0, 0,-1],
                                [ 1, 0, 0,-1, 0, 0, 0, 0, 0, 1],
                                [ 0,-1, 0, 0, 1, 0, 0, 0, 0,-1],
                                [ 0, 1, 0, 0,-1, 0, 0, 0, 0, 1],
                                [ 0, 0,-1, 0, 0, 1, 0, 0, 0,-1],
                                [ 0, 0, 1, 0, 0,-1, 0, 0, 0, 1],
                                [ 0, 0, 0,-1, 0, 0, 2, 0, 0,-1],
                                [ 0, 0, 0, 1, 0, 0,-2, 0, 0, 1],
                                [ 0, 0, 0, 0,-1, 0, 0, 2, 0,-1],
                                [ 0, 0, 0, 0, 1, 0, 0,-2, 0, 1],
                                [ 0, 0, 0, 0, 0, 0,-1,-1, 1, 1],
                                [ 0, 0, 0, 0, 0, 0, 1, 1,-1,-1],
                                [ 0, 0, 0, 0, 0, 1, 0,-1,-1, 1],
                                [ 0, 0, 0, 0, 0,-1, 0, 1, 1,-1]]).T
pars[3]['kijnpars']  = jnp.array([25.,10.,30.,15.,20.,50.,800.,1200.,200.,100.,800.,300.,700.,200.])*0.8
pars[3]['bc']       = [[jnp.array(_) for _ in [[t0],bcv_+[0.]*(pars[3]['stoich'].shape[0]-1-len(bcv_))+[1.]]] for bcv_ in bcv]
pars[3]['nncpars']  = [{'forward': {'trig':False, 'damp':True,'nobs':3,'bc':bc_},\
                       'inverse': {'trig':True, 'damp':False,'nobs':3,'bc':bc_}} for bc_ in pars[3]['bc']]
pars[3]['nnmpars']  = {'forward': {'layers_sizes'  : [[1,20,20,20,pars[3]['stoich'].shape[0]]],
                       'act_fun'                   : [tanh,swish,tanh],
                       'nn_scale'                  : 0.001},
                       'inverse' : {'layers_sizes' : [[1,20,20,20,pars[3]['stoich'].shape[0]-1]],
                       'act_fun'                   : [tanh,swish,tanh],
                       'nn_scale'                  : 0.001}}
pars[3]['nntpars']  = {'layers_sizes' : [[1,6,1]],
                       'act_fun'      : [swish,tanh,swish],
                       'nn_scale'     : 0.001}
pars[3]['kijnpars_tags'] = ['k_{1}^d','k_{\text{-}1}^d','k_{2}^d','k_{\text{-}2}^d','k_{3}^d','k_{\text{-}3}^d']+\
                              ['k_{3}^c','k_{\text{-}3}^c','k_{4}^c','k_{\text{-}4}^c','k_{1}^s','k_{\text{-}1}^s','k_{5}^c','k_{\text{-}5}^c'] 