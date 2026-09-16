"""Fresh fixed-weight training using the saved paper architectures and schedules.

This is a modern-JAX rerun of the original objective. Archived parameters are
used only to read architecture shapes and schedule metadata, never as initial
weights. Historical initialization is not known to be bitwise reproducible.
"""
from copy import deepcopy
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import optimizers

from ..basis import nn, model
from ..basis.pareto import nn_combo
from ..numerics import require_finite
from ..rkinn import ACTIVATIONS
from . import load_archive, models, record
from .reference import observations, evaluate_archive


def plan(identifier):
    info, archive = record(identifier), load_archive(identifier)
    spec = models()[str(info['model_index'])]
    forward = info['scenario'] == 'fwd'
    architecture = [[np.shape(p[0][0])[1], *[np.shape(layer[0])[0] for layer in p]] for p in archive['params'][0][0][0]]
    stages, previous = [], 0
    for state in archive['iter_data']:
        stages.append({'epochs': int(state[0])-previous, 'steps_per_epoch':150 if info['schedule']=='alpha17sens' else 100,
                       'alpha':1. if forward else float(state[2][0][0]), 'optimizer': state[4][1]})
        previous = int(state[0])
    if info['schedule'] == 'alpha13sens':
        # This earlier sensitivity run has no unambiguous iteration-count
        # recipe in the surviving source; evaluation of its archives is valid.
        raise ValueError('The alpha13sens archive can be evaluated, but its fresh-training iteration count is not documented; use alpha17sens for the preserved Pareto recipe')
    return {'schema_version':1, 'id':identifier, 'method':'fixed', 'mode':'forward' if forward else 'inverse',
            'source':info, 'species':spec['sps'], 'stoichiometry':spec['stoich'], 'true_rate_constants':spec['kijnpars'],
            'initial_states':[bc[1] for bc in spec['bc']], 'neural_layers':architecture,
            'activations':[['tanh','swish','tanh']] + ([['swish']] if forward else []),
            'nn_initial_scale':.001, 'kinetic_initial_scale':.1 if info['schedule']=='alpha17sens' else .01,
            'seed':0, 'time_grid':{'points':100,'formula':'(logspace(0, log10(21), 100)-1)/20'},
            'observations':info['scenario'], 'noise_standard_deviation':.025 if info['scenario']=='invvwn' else 0.,
            'surface_signal_normalization':'standard deviation across both datasets',
            'stages':stages, 'stopping_loss':1e-12,
            'scope':'Fresh initialization with archived architecture and stage schedule; numerical equivalence to the historical training run is unverified.'}


def build(recipe):
    forward = recipe['mode'] == 'forward'
    info = recipe['source']
    data, _ = observations(info['model_index'], info['scenario'])
    nets = []
    for initial in recipe['initial_states']:
        raw = [nn([widths], [ACTIVATIONS[name] for name in names], nn_scale=recipe['nn_initial_scale'])
               for widths, names in zip(recipe['neural_layers'], recipe['activations'])]
        nets.append(nn_combo(raw, mode=recipe['mode'], nobs=3, trig=not forward and len(initial)>3,
                            damp=forward, bc=(jnp.array([0.]),jnp.asarray(initial))))
    physical = model(jnp.asarray(recipe['stoichiometry']), nobs=3, model_scale=recipe['kinetic_initial_scale'])
    params = [[net.params for net in nets]]
    scaled = not forward and info['scenario'] in ('invsc','invvwn') and len(recipe['species'])>3
    if not forward:
        params.append([jnp.log(jnp.abs(physical.params[0]))])
    if scaled:
        params.append(jnp.zeros(len(recipe['species'])-3))
    fixed_logs = [jnp.log(jnp.asarray(recipe['true_rate_constants']))]
    def errors(params):
        rates = fixed_logs if forward else params[1]
        physics, measured = [], []
        for net, p, (times, values) in zip(nets, params[0], data):
            x = net.batched_state(p[0], times)
            dx = net.diff_state(p[0], times)[:,:,0]
            physics.append(jnp.mean((dx-physical.batched_eval(rates,[x]))**2))
            if not forward:
                target = values * jnp.concatenate((jnp.ones(3),jnp.exp(params[2]))) if scaled else values
                measured.append(jnp.mean((x[:,:target.shape[1]]-target)**2))
        return jnp.mean(jnp.stack(physics)), jnp.mean(jnp.stack(measured)) if measured else jnp.array(0.)
    return params, jax.jit(errors)


def train(identifier, *, epochs_per_stage=None, steps_per_epoch=None, max_stages=None, progress=None, checkpoint=None):
    started = perf_counter()
    recipe = plan(identifier)
    overrides = {'epochs_per_stage':epochs_per_stage, 'steps_per_epoch':steps_per_epoch, 'max_stages':max_stages}
    for key, value in overrides.items():
        if value is not None and (isinstance(value,bool) or not isinstance(value,int) or value < 1):
            raise ValueError(f'{key} must be a positive integer')
    recipe['overrides'] = {k:v for k,v in overrides.items() if v is not None}
    stages = deepcopy(recipe['stages'][:max_stages])
    for stage in stages:
        if epochs_per_stage is not None:
            stage['epochs'] = epochs_per_stage
        if steps_per_epoch is not None:
            stage['steps_per_epoch'] = steps_per_epoch
    params, errors = build(recipe)
    def loss(params, alpha):
        physics, measured = errors(params)
        return physics + alpha*measured
    gradient = jax.grad(loss)
    steps = stages[0]['steps_per_epoch']
    @jax.jit
    def epoch(state, offset, alpha, lr, b1, b2, eps):
        _, update, get = optimizers.adam(lr,b1=b1,b2=b2,eps=eps)
        def step(state, i):
            return update(i+offset,gradient(get(state),alpha),state), None
        return jax.lax.scan(step,state,jnp.arange(steps))[0]
    archive = {'params':params,'iter_data':[], 'epoch':0, 'origin':'fresh_training'}
    compile_seconds, execution_seconds = 0., 0.
    compiled = None
    last_progress = perf_counter()
    for stage_index, stage in enumerate(stages):
        settings = stage['optimizer']
        init, _, get = optimizers.adam(**settings)
        opt_state = init(params)
        args = tuple(jnp.asarray(v) for v in (stage['alpha'],settings['step_size'],settings['b1'],settings['b2'],settings['eps']))
        if compiled is None:
            before = perf_counter()
            compiled = epoch.lower(opt_state,jnp.asarray(0),*args).compile()
            compile_seconds += perf_counter()-before
        for index in range(stage['epochs']):
            before = perf_counter()
            opt_state = compiled(opt_state,jnp.asarray(index*steps),*args)
            params = get(opt_state)
            jax.block_until_ready(params)
            execution_seconds += perf_counter()-before
            require_finite(params,'paper training')
            physics, measured = map(float,errors(params))
            require_finite((physics,measured),'paper residuals')
            archive['epoch'] += 1
            if progress and perf_counter()-last_progress >= 5:
                progress({'stage':stage_index+1,'stages':len(stages),'epoch':archive['epoch'], 'physics_mse':physics,'data_mse':measured})
                last_progress = perf_counter()
            if physics+stage['alpha']*measured <= recipe['stopping_loss']:
                break
        archive['params'] = params
        archive['iter_data'].append([archive['epoch'],[params],[[stage['alpha']]],[[physics,measured]],['adam',settings]])
        if checkpoint:
            checkpoint(archive, recipe)
        if progress:
            progress({'stage':stage_index+1,'stages':len(stages),'epoch':archive['epoch'], 'physics_mse':physics,'data_mse':measured})
    result = evaluate_archive(identifier,archive,state='final')
    result.update(origin='fresh_training', status='budget_override' if recipe['overrides'] else 'schedule_completed', recipe=recipe)
    result['timing'] = {'compilation_seconds':compile_seconds, 'warm_training_seconds':execution_seconds,
                        'total_seconds':perf_counter()-started, 'scope':'Includes setup, explicit epoch compilation, synchronized training, diagnostics and final evaluation; excludes CLI file/plot export.'}
    return result, archive
