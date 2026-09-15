"""Original KINNs trajectory and weighted-residual trainer.

The nn_combo parameterization is retained from paper/trainer_source.py.
The trainer uses TrainerCV and its original physics MSE + alpha * data MSE
objective, with the current model's log-rate convention and CSV-sized batches.
"""
from . import jnp, jit, jacfwd, vmap, pmap
from .trainer import TrainerCV

class nn_combo(object):

    def __init__(self, nns, usepmap=False, bc=None, nobs=None, trig=False, damp=False, gain=1.,mode=None):
        self.nns = nns
        self.bc  = bc
        self.params = list(zip(*[_.params for _ in nns]))
        if usepmap :
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
        if self.trig:
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

class trainer(TrainerCV):
    """Original weighted KINNs objective for a calibrated observation operator."""
    def _sparams(self):
        return [self.alpha]

    def _res_fun(self, params, sparams, batches):
        network_params = params[0]
        rates = params[1] if self.mode == 'inverse' else self.model.params
        physics, measured, predictions = [], [], []
        for net, pars, (times, values, collocation) in zip(self.nn, network_params, batches):
            states = net.batched_state(pars[0], collocation)
            derivative = net.diff_state(pars[0], collocation)[:, :, 0]
            residual = (derivative - self.model.batched_eval(rates, [states])) / self.model.time_scale
            prediction = net.batched_state(pars[0], times)
            physics.append(residual)
            measured.append(prediction[:, :values.shape[1]] - values if self.mode == 'inverse' else jnp.zeros_like(prediction))
            predictions.append(prediction)
        return physics, measured, predictions

    def _err_fun(self, params, sparams, batches):
        physics, measured, _ = self._res_fun(params, sparams, batches)
        # Average each dataset first so unequal sample counts remain valid.
        return [jnp.stack([jnp.mean(e**2) for e in physics]),
                sparams[0] * jnp.stack([jnp.mean(e**2) for e in measured])]
