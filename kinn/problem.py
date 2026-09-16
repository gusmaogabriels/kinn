"""Validated JSON inputs for local, closed-batch kinn problems."""
import csv
import json
from pathlib import Path

import numpy as np

ACTIVATIONS = ('tanh', 'sigmoid', 'swish', 'softplus', 'sin', 'gaussian')


class ProblemError(ValueError):
    """Invalid or unsupported scientific input."""


def _keys(obj, allowed, where):
    if not isinstance(obj, dict):
        raise ProblemError(f'{where} must be an object')
    unexpected = set(obj) - set(allowed)
    if unexpected:
        raise ProblemError(f'{where}: unknown fields {sorted(unexpected)}')


def _names(value, where, *, empty=False):
    if not isinstance(value, list) or (not empty and not value) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ProblemError(f'{where} must be a list of nonempty names')
    if len(value) != len(set(value)):
        raise ProblemError(f'{where} contains duplicate names')
    return value


def _array(value, where, ndim):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf' or raw.ndim != ndim or any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
            raise ValueError()
        array = raw.astype(float)
        if not np.all(np.isfinite(array)):
            raise ValueError()
        return array
    except (TypeError, ValueError, OverflowError):
        raise ProblemError(f'{where} must contain finite numbers with {ndim} dimensions') from None


def _positive(value, where, *, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ProblemError(f'{where} must be a finite {"nonnegative" if zero else "positive"} number')
    if integer and (type(value) is not int):
        raise ProblemError(f'{where} must be an integer')
    return value


def _times(value, where):
    result = _array(value, where, 1)
    if len(result) < 3 or not np.all(np.diff(result) > 0):
        raise ProblemError(f'{where} needs at least three strictly increasing times')
    return result


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProblemError(f'duplicate JSON field: {key}')
        result[key] = value
    return result


def load(path):
    path = Path(path)
    try:
        obj = json.loads(path.read_text(), object_pairs_hook=_pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ProblemError(f'nonfinite JSON value: {value}')))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProblemError(f'cannot read problem JSON: {error}') from None
    return validate(obj, base=path.parent)


def validate(obj, *, base=Path('.')):
    _keys(obj, ('schema_version', 'method', 'mode', 'species', 'surface_species', 'stoichiometry',
                'rate_constants', 'initial_rate_constants', 'observed_species', 'datasets',
                'surrogate', 'training', 'uncertainty', 'units'), 'problem')
    if type(obj.get('schema_version')) is not int or obj['schema_version'] != 1:
        raise ProblemError('schema_version must be 1')
    mode = obj.get('mode')
    if mode not in ('forward', 'inverse'):
        raise ProblemError('mode must be forward or inverse')
    method = obj.get('method', 'mle')
    if method not in ('fixed', 'mle'):
        raise ProblemError('method must be fixed or mle')
    species = _names(obj.get('species'), 'species')
    surface = _names(obj.get('surface_species', []), 'surface_species', empty=True)
    if not set(surface) <= set(species):
        raise ProblemError('surface_species must refer to species names')
    bulk = [name for name in species if name not in surface]
    if not bulk:
        raise ProblemError('this kinn interface requires at least one nonsurface species')
    order = [species.index(name) for name in bulk + surface]
    matrix = _array(obj.get('stoichiometry'), 'stoichiometry', 2)
    if matrix.shape[0] != len(species) or matrix.shape[1] == 0 or not np.all(matrix == np.round(matrix)):
        raise ProblemError('stoichiometry must have integer entries, one row per species and one column per directed elementary reaction')
    if np.any(np.abs(matrix) > 2**31-1):
        raise ProblemError('stoichiometry entries must fit signed 32-bit integers')
    if np.any(np.all(matrix == 0, axis=0)) or not np.any(matrix < 0):
        raise ProblemError('each reaction must change species; this mass-action interface needs reactants')
    if surface and not np.all(matrix[[species.index(x) for x in surface]].sum(axis=0) == 0):
        raise ProblemError('surface_species must conserve one site balance, including the vacant-site species')
    rate_key = 'rate_constants' if mode == 'forward' else 'initial_rate_constants'
    other_key = 'initial_rate_constants' if mode == 'forward' else 'rate_constants'
    if other_key in obj:
        raise ProblemError(f'{mode} mode accepts {rate_key}, not {other_key}')
    rates = _array(obj.get(rate_key), rate_key, 1)
    if rates.shape != (matrix.shape[1],) or np.any(rates <= 0):
        raise ProblemError(f'{rate_key} requires one strictly positive value per matrix column')
    observed = _names(obj.get('observed_species', bulk if surface else species), 'observed_species')
    if not set(observed) <= set(species):
        raise ProblemError('observed_species must refer to species names')
    # The original rKINNs observation operator partitions measured bulk from latent surface states.
    if set(observed) not in (set(bulk), set(species)):
        raise ProblemError('observations must cover all nonsurface species, or all species; arbitrary missing bulk observations are not supported')
    output_observed = bulk if len(observed) == len(bulk) else bulk + surface
    value_order = [observed.index(name) for name in output_observed]
    architecture = {'layers': [16, 16], 'activations': 'tanh', 'init_scale': .3, 'boundary_gain': 'auto'}
    _keys(obj.get('surrogate', {}), architecture, 'surrogate')
    architecture.update(obj.get('surrogate', {}))
    layers = architecture['layers']
    if not isinstance(layers, list) or not layers:
        raise ProblemError('surrogate.layers must list the hidden-layer widths')
    for width in layers:
        _positive(width, 'hidden-layer width', integer=True)
    activation = architecture['activations']
    if isinstance(activation, str):
        activation = [activation] * len(layers)
    if not isinstance(activation, list) or len(activation) != len(layers) or any(x not in ACTIVATIONS for x in activation):
        raise ProblemError(f'activations must provide one supported name per hidden layer: {", ".join(ACTIVATIONS)}')
    architecture['activations'] = activation
    _positive(architecture['init_scale'], 'surrogate.init_scale')
    if architecture['boundary_gain'] != 'auto':
        _positive(architecture['boundary_gain'], 'surrogate.boundary_gain')
    training = {'epochs': 300, 'steps_per_epoch': 100, 'warmup_steps': 1000, 'learning_rate': .001, 'learning_rate_schedule': 'cosine', 'seed': 0,
                'physics_tolerance': .01, 'data_tolerance': .01}
    if method == 'fixed':
        training['alpha'] = 1.
    _keys(obj.get('training', {}), training, 'training')
    training.update(obj.get('training', {}))
    if training['learning_rate_schedule'] not in ('constant', 'cosine'):
        raise ProblemError('training.learning_rate_schedule must be constant or cosine')
    for key, value in training.items():
        if key == 'learning_rate_schedule':
            continue
        _positive(value, 'training.' + key, integer=key in ('epochs', 'steps_per_epoch', 'warmup_steps', 'seed'), zero=key in ('seed', 'warmup_steps'))
    if training['seed'] > 2**32-1:
        raise ProblemError('seed must fit an unsigned 32-bit integer')
    uncertainty = {'representation_error': False}
    _keys(obj.get('uncertainty', {}), uncertainty, 'uncertainty')
    uncertainty.update(obj.get('uncertainty', {}))
    if type(uncertainty['representation_error']) is not bool:
        raise ProblemError('uncertainty.representation_error must be a boolean')
    if method == 'fixed' and uncertainty['representation_error']:
        raise ProblemError('representation_error requires method mle')
    units = obj.get('units', {'time': 'unspecified', 'bulk': 'unspecified', 'surface': 'fraction'})
    _keys(units, ('time', 'bulk', 'surface'), 'units')
    if any(not isinstance(x, str) or not x.strip() for x in units.values()):
        raise ProblemError('units must contain nonempty text labels; no unit conversion is performed')
    raw_datasets = obj.get('datasets')
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ProblemError('datasets must be a nonempty list')
    datasets = []
    for i, row in enumerate(raw_datasets):
        where = f'datasets[{i}]'
        _keys(row, ('times', 'values', 'data_file', 'initial_state', 'collocation_times'), where)
        row = dict(row)
        if 'data_file' in row:
            if 'times' in row or 'values' in row or not isinstance(row['data_file'], str):
                raise ProblemError(f'{where}: use data_file or inline times/values')
            try:
                with (Path(base) / row['data_file']).open(newline='') as handle:
                    reader = csv.DictReader(handle)
                    expected = ['time', *observed]
                    if reader.fieldnames is None or len(reader.fieldnames) != len(expected) or set(reader.fieldnames) != set(expected):
                        raise ProblemError(f'{where}: CSV columns must be time and the observed species names')
                    records = list(reader)
                    row['times'] = [float(r['time']) for r in records]
                    row['values'] = [[float(r[name]) for name in observed] for r in records]
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise ProblemError(f'{where}: cannot read CSV: {error}') from None
        times = _times(row.get('times'), where + '.times')
        collocation = _times(row.get('collocation_times', row['times']), where + '.collocation_times')
        if collocation[0] < times[0] or collocation[-1] > times[-1]:
            raise ProblemError(f'{where}: collocation_times must lie within the dataset time interval')
        initial = row.get('initial_state')
        if initial is not None:
            initial = _array(initial, where + '.initial_state', 1)
            if initial.shape != (len(species),) or np.any(initial < 0):
                raise ProblemError(f'{where}: initial_state must be nonnegative with one value per species')
            if surface and not np.isclose(initial[[species.index(x) for x in surface]].sum(), 1., atol=1e-10, rtol=0):
                raise ProblemError(f'{where}: initial surface fractions, including vacant sites, must sum to one')
        elif mode == 'forward':
            raise ProblemError(f'{where}: forward mode requires initial_state')
        if mode == 'inverse':
            values = _array(row.get('values'), where + '.values', 2)
            if values.shape != (len(times), len(observed)):
                raise ProblemError(f'{where}: values need one row per time and one column per observed species')
            values = values[:, value_order]
        else:
            if 'values' in row:
                raise ProblemError(f'{where}: forward mode does not use measured values')
            values = None
        datasets.append({'times': times.tolist(), 'collocation_times': collocation.tolist(),
                         'values': None if values is None else values.tolist(),
                         'initial_state': None if initial is None else initial.tolist()})
    return {'schema_version': 1, 'method': method, 'mode': mode, 'species': species, 'surface_species': surface,
            'n_bulk': len(bulk), 'order': order, 'observed_species': output_observed,
            'stoichiometry': matrix.astype(int).tolist(), 'rates': rates.tolist(),
            'datasets': datasets, 'surrogate': architecture, 'training': training,
            'uncertainty': uncertainty, 'units': units,
            'time_scale': max(row['times'][-1] - row['times'][0] for row in datasets)}


def describe(problem):
    """Return interpreted dimensions and conventions without importing JAX."""
    matrix = np.asarray(problem['stoichiometry'])
    u, singular, _ = np.linalg.svd(matrix[problem['order']])
    rank = int(np.sum(singular >= 1e-8))
    output = len(problem['species']) - bool(problem['surface_species'])
    if problem['method'] == 'mle':
        latent = u[:, :rank][problem['n_bulk']:]
        bulk_directions = rank - int(np.sum(np.linalg.svd(latent.T @ latent, compute_uv=False) >= 1e-12))
        output = len(problem['surface_species']) - bool(problem['surface_species']) + bulk_directions
    return {'valid': True, 'method': problem['method'], 'mode': problem['mode'], 'species': problem['species'],
            'internal_species_order': [problem['species'][i] for i in problem['order']],
            'observed_species': problem['observed_species'], 'surface_species': problem['surface_species'],
            'matrix_shape': list(matrix.shape), 'stoichiometric_rank': rank,
            'conservation_dimension': matrix.shape[0] - rank,
            'neural_layers': [1, *problem['surrogate']['layers'], output],
            'initial_condition_enforced': [row['initial_state'] is not None for row in problem['datasets']],
            'surrogate': problem['surrogate'], 'datasets': len(problem['datasets']),
            'time_scale': problem['time_scale'], 'units': problem['units'],
            'execution': 'local', 'variance': 'first-order residual propagation with OAS shrinkage' if problem['method'] == 'mle' else 'not estimated by the fixed-weight objective'}
