"""Local CLI driver for the original weighted KINNs formulation."""
from time import perf_counter
import jax
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import optimizers

from .basis import nn
from .basis.pareto import nn_combo, trainer
from .numerics import boundary_gain, constraints, require_finite, timing
from .rkinn import ACTIVATIONS, _ScaledModel


def build(problem):
    order = jnp.asarray(problem['order'], dtype=jnp.int32)
    physical = _ScaledModel(jnp.asarray(problem['stoichiometry'])[order], problem['n_bulk'], problem['time_scale'])
    physical.params = [jnp.asarray(problem['rates'])]
    networks, data = [], []
    architecture = problem['surrogate']
    for i, row in enumerate(problem['datasets']):
        raw = nn([[1, *architecture['layers'], len(order) - bool(problem['surface_species'])]],
                 [ACTIVATIONS[name] for name in architecture['activations']], nn_scale=architecture['init_scale'])
        raw.seed = (problem['training']['seed'] + i) % 2**32
        raw._init_params()
        initial = jnp.asarray(row['initial_state'])[order] if row['initial_state'] is not None else None
        networks.append(nn_combo([raw], nobs=problem['n_bulk'], trig=bool(problem['surface_species']),
                                 mode='forward' if initial is not None else 'inverse', bc=(jnp.array([0.]), initial),
                                 gain=boundary_gain(problem, physical, initial)))
        times = (jnp.asarray(row['times']) - row['times'][0])[:, None] / problem['time_scale']
        collocation = (jnp.asarray(row['collocation_times']) - row['times'][0])[:, None] / problem['time_scale']
        values = jnp.asarray(row['values']) if problem['mode'] == 'inverse' else jnp.zeros((len(times), len(order)))
        data.append((times, values, collocation))
    engine = trainer(networks, physical, mode=problem['mode'], nobs=problem['n_bulk'], alpha=problem['training']['alpha'])
    # The original TrainerCV constructor accepts positive rates for initialization;
    # the current physical model evaluates log rates internally in either mode.
    physical.params = [jnp.log(jnp.asarray(problem['rates']))]
    learning_rate = problem['training']['learning_rate']
    if problem['training']['learning_rate_schedule'] == 'cosine':
        import optax
        learning_rate = optax.cosine_decay_schedule(learning_rate, problem['training']['epochs'] * problem['training']['steps_per_epoch'], alpha=.01)
    engine.set_optimizer(optimizers.adam, {'step_size': learning_rate})
    engine.initialize()
    engine.grads = jax.jit(jax.grad(engine._loss))
    return engine, data


def run(problem):
    started = perf_counter()
    engine, data = build(problem)
    settings = problem['training']
    warmup_started = perf_counter()
    if problem['mode'] == 'inverse' and settings['warmup_steps']:
        init, update, get = optimizers.adam(settings['learning_rate'])
        def data_loss(params):
            predictions = [net.batched_state(pars[0], row[0]) for net, pars, row in zip(engine.nn, params[0], data)]
            return sum(jnp.mean((x[:, :row[1].shape[1]] - row[1])**2) for x, row in zip(predictions, data)) / len(data)
        @jax.jit
        def warmup(state):
            def step(state, i):
                return update(i, jax.grad(data_loss)(get(state)), state), None
            return jax.lax.scan(step, state, jnp.arange(settings['warmup_steps']))[0]
        engine.params = get(warmup(init(engine.params)))
        jax.block_until_ready(engine.params)
        require_finite(engine.params, 'data initialization')
        engine._opt_state = engine._opt_init(engine.params)
    warmup_seconds = perf_counter() - warmup_started
    alpha = [jnp.asarray(settings['alpha'])]
    @jax.jit
    def epoch(state, offset):
        def step(state, i):
            return engine.step(i + offset, state, alpha, data), None
        return jax.lax.scan(step, state, jnp.arange(settings['steps_per_epoch']))[0]
    setup_seconds = perf_counter() - started
    durations, epochs, history = [], [], []
    success = False
    for index in range(settings['epochs']):
        before = perf_counter()
        engine._opt_state = epoch(engine._opt_state, index * settings['steps_per_epoch'])
        engine.params = engine._get_params(engine._opt_state)
        jax.block_until_ready(engine.params)
        durations.append(perf_counter() - before)
        require_finite(engine.params, 'optimization')
        physics, measured, predictions = engine.res_fun(engine.params, alpha, data)
        require_finite((physics, measured, predictions), 'residual evaluation')
        physics_rmse = float(jnp.sqrt(jnp.mean(jnp.concatenate([x.ravel() for x in physics])**2)))
        state_rmse = float(jnp.sqrt(jnp.mean(jnp.concatenate([x.ravel() for x in measured])**2)))
        history.append({'epoch': index+1, 'state_rmse': state_rmse, 'physics_rmse': physics_rmse})
        epochs.append(perf_counter() - before)
        success = physics_rmse <= settings['physics_tolerance'] and (problem['mode'] == 'forward' or state_rmse <= settings['data_tolerance'])
        if success:
            break
    inverse_order = np.argsort(problem['order'])
    arrays = [np.asarray(x)[:, inverse_order] for x in predictions]
    collocation = [np.asarray(net.batched_state(pars[0], row[2]))[:, inverse_order] for net, pars, row in zip(engine.nn, engine.params[0], data)]
    logs = engine.params[1][0] if problem['mode'] == 'inverse' else engine.model.params[0]
    checks = constraints(problem, arrays + collocation)
    status = 'converged' if success else 'max_epochs'
    if success and not (checks['nonnegative_at_sampled_points'] and checks['surface_site_balance_satisfied']):
        status = 'physical_constraint_violation'
    return {'schema_version': 1, 'method': 'fixed', 'status': status,
            'mode': problem['mode'], 'species': problem['species'],
            'rate_constants': np.asarray(jnp.exp(logs)).tolist(), 'log_rate_constants': np.asarray(logs).tolist(),
            'predictions': [{'times': row['times'], 'states': x.tolist()} for row, x in zip(problem['datasets'], arrays)],
            'history': history, 'objective': 'physics_mse + alpha * data_mse', 'alpha': settings['alpha'],
            'uncertainty': {'method': 'not_estimated', 'scope': 'The fixed-weight objective does not estimate covariance; use method mle for automatic residual propagation.'},
            'physical_checks': checks,
            'units': problem['units'], 'surrogate': problem['surrogate'], 'time_scale': problem['time_scale'],
            'timing': timing(started, setup_seconds, warmup_seconds, durations, epochs)}
