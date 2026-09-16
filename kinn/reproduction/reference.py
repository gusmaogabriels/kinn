"""Evaluate original saved networks using the original JAX parameterizations."""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp

from ..basis import nn, model
from ..basis.pareto import nn_combo
from ..rkinn import ACTIVATIONS
from . import models, record, load_archive


@lru_cache(maxsize=4)
def reference_data(index):
    spec = models()[str(index)]
    matrix = np.asarray(spec['stoich'])
    rates = np.asarray(spec['kijnpars'])
    orders = np.maximum(-matrix, 0)
    # The original scripts use 100 logarithmically spaced points in [0, 1].
    times = (np.logspace(0, np.log10(21.), 100) - 1.) / 20.
    def rhs(t, state):
        return matrix @ (rates * np.prod(state[:, None] ** orders, axis=0))
    rows = []
    for _, initial in spec['bc']:
        # The historical 1e-20 tolerance is below double precision. Explicit,
        # achievable tolerances avoid depending on SciPy's warning/clamp behavior.
        sol = solve_ivp(rhs, (0., 1.), initial, t_eval=times, method='LSODA', rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise ValueError('Reference integration failed: ' + sol.message)
        rows.append({'times': times, 'states': sol.y.T})
    return spec, rows


def network(combined_params, initial, forward):
    """Infer widths from saved weights, including the forward time-scale NN."""
    nets = []
    for index, params in enumerate(combined_params):
        widths = [np.shape(params[0][0])[1], *[np.shape(layer[0])[0] for layer in params]]
        names = ['tanh', 'swish', 'tanh'] if index == 0 else ['swish']
        if len(widths) - 2 != len(names):
            raise ValueError('Unexpected historical network architecture')
        nets.append(nn([widths], [ACTIVATIONS[name] for name in names], init=False))
        nets[-1].act_fun = [ACTIVATIONS[name] for name in names]
        nets[-1].params = [params]
    return nn_combo(nets, bc=(jnp.array([0.]), jnp.asarray(initial)),
                    nobs=3, trig=not forward and len(initial) > 3,
                    damp=forward, mode='forward' if forward else 'inverse')


def metric(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape[1] == 0:
        return None
    difference = actual - expected
    numerator = np.sum(difference**2, axis=0)
    denominator = np.sum((expected - expected.mean(axis=0))**2, axis=0)
    scores = np.where(denominator > 0, 1 - numerator / np.where(denominator > 0, denominator, 1), np.where(numerator == 0, 1., 0.))
    a, b = expected - expected.mean(axis=0), actual - actual.mean(axis=0)
    norm = np.sum(a*a, axis=0) * np.sum(b*b, axis=0)
    # The paper's r2_score is mean squared Pearson correlation, not sklearn R².
    squared_correlation = float(np.mean(np.sum(a*b, axis=0)**2 / norm)) if np.all(norm > 0) else None
    return {'r2': squared_correlation, 'coefficient_of_determination': float(scores.mean()),
            'mae': float(np.mean(np.abs(difference))), 'mse': float(np.mean(difference**2))}


def paper_state(archive):
    """The checkpoint heuristic in paper/kinn_plotsgen_reg.py, lines 416–424."""
    rows = np.asarray([saved[2][0] + saved[3][0] for saved in archive['iter_data'][2:]], dtype=float)
    rows = np.column_stack((rows[:, 0], np.log(rows[:, 1:])))
    rows = rows[np.diff(rows[:, 0], prepend=0.) > 0]
    if len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError('Archive does not support the historical noisy-checkpoint selection')
    slopes = np.diff(rows[:, 2]) / np.diff(rows[:, 1])
    return int(np.argmin(slopes)) + 3


def evaluate(identifier, *, state='paper'):
    return evaluate_archive(identifier, load_archive(identifier), state=state)


def evaluate_archive(identifier, archive, *, state='paper'):
    info = record(identifier)
    spec, truth = reference_data(info['model_index'])
    if state == 'paper':
        state = paper_state(archive) if info['scenario'] == 'invvwn' else None
    elif state == 'final':
        state = None
    if state is not None and (isinstance(state, bool) or not isinstance(state, int) or not 0 <= state < len(archive['iter_data'])):
        raise ValueError('state must be paper, final, or a valid nonnegative checkpoint index')
    params = archive['params'] if state is None else archive['iter_data'][state][1][0]
    forward = info['scenario'] == 'fwd'
    rates = np.asarray(spec['kijnpars']) if forward else np.exp(np.asarray(params[1][0]))
    physical = model(jnp.asarray(spec['stoich']), nobs=3)
    data, _ = observations(info['model_index'], info['scenario'])
    rows = []
    for i, row in enumerate(truth):
        combined = params[0][i][0]
        net = network(combined, spec['bc'][i][1], forward)
        times = jnp.asarray(row['times'])[:, None]
        predicted = np.asarray(net.batched_state(combined, times))
        derivative = np.asarray(net.diff_state(combined, times))[:, :, 0]
        rhs = np.asarray(physical.batched_eval([jnp.log(jnp.asarray(rates))], [jnp.asarray(predicted)]))
        if not all(np.isfinite(a).all() for a in (predicted, derivative, rhs, rates)):
            raise ValueError(f'Nonfinite evaluation of archived state: {identifier}')
        observed = np.asarray(data[i][1]) if not forward else np.empty((len(times),0))
        calibrated = observed * np.r_[np.ones(3), np.exp(np.asarray(params[2]))] if len(params)==3 else observed
        rows.append({'times': row['times'], 'truth': row['states'], 'states': predicted,
                     'observations':observed, 'calibrated_observations':calibrated,
                     'derivatives': derivative, 'rhs': rhs,
                     'metrics': {'bulk_states': metric(row['states'][:, :3], predicted[:, :3]),
                                 'surface_states': metric(row['states'][:, 3:], predicted[:, 3:]),
                                 'bulk_derivatives': metric(rhs[:, :3], derivative[:, :3]),
                                 'surface_derivatives': metric(rhs[:, 3:], derivative[:, 3:])}})
    logs, true_logs = np.log(rates), np.log(spec['kijnpars'])
    correlation = float(np.corrcoef(true_logs, logs)[0, 1]) if np.std(logs) > 0 else None
    history = []
    for saved in archive['iter_data']:
        alpha = float(np.asarray(saved[2][0][0])) if not forward else 1.
        errors = [float(np.asarray(v)) for v in saved[3][0]]
        entry = {'epoch': int(saved[0]), 'alpha': alpha, 'physics_mse': errors[0],
                 'data_mse': errors[1] if not forward else 0.}
        if not forward:
            entry['log_rate_constants'] = np.asarray(saved[1][0][1][0]).tolist()
        history.append(entry)
    return {'schema_version':1, 'id': identifier, 'source': info, 'mode': 'forward' if forward else 'inverse', 'method': 'fixed',
            'origin': 'archived_parameters', 'species': spec['sps'],
            'rate_constants': rates.tolist(), 'true_rate_constants': spec['kijnpars'],
            'log_rate_mae': float(np.mean(np.abs(logs-true_logs))), 'log_rate_correlation': correlation,
            'calibration_scales': np.exp(np.asarray(params[2])).tolist() if len(params) == 3 else None,
            'selected_state': state,
            'epoch': int(archive['epoch'] if state is None else archive['iter_data'][state][0]),
            'history': history, 'datasets': rows}


def observations(index, scenario):
    """Recreate Q, SQ and SQ+n data from the original fixed-weight paper."""
    spec, truth = reference_data(index)
    scale = np.std(np.vstack([r['states'][:, 3:] for r in truth]), axis=0)
    data = []
    for row in truth:
        x = jnp.asarray(row['states'])
        if scenario == 'inv':
            x = x[:, :3]
        elif scenario in ('invsc', 'invvwn'):
            x = jnp.concatenate((x[:, :3], x[:, 3:] / jnp.asarray(scale)), axis=1)
            if scenario == 'invvwn':
                # The original fixed-weight script deliberately reuses key 0.
                x = x + .025 * jax.random.normal(jax.random.PRNGKey(0), x.shape)
        data.append((jnp.asarray(row['times'])[:, None], x))
    return data, scale
