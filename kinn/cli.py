"""The local kinn command line interface, including its MLE and SVD extension."""
import argparse
import json
from importlib.resources import files
from pathlib import Path
import sys
import os
import tempfile

from . import __version__


def example(kind='homogeneous', mode='inverse', method='rkinn'):
    import math
    times = [i / 20 for i in range(21)]
    if kind == 'homogeneous':
        species, surface = ['A', 'B'], []
        matrix, initial = [[-1, 1], [1, -1]], [1., 0.]
        observed = species
        values = [[1/3+2/3*math.exp(-3*t), 2/3-2/3*math.exp(-3*t)] for t in times]
    else:
        species, surface = ['A', 'A*', '*'], ['A*', '*']
        matrix, initial = [[-1, 1], [1, -1], [-1, 1]], [.8, .2, .8]
        observed = ['A']
        coverage = [(0.5-math.exp(-3*t)/3)/(1-math.exp(-3*t)/6) for t in times]
        values = [[1-c] for c in coverage]
    row = {'times': times, 'initial_state': initial}
    if mode == 'inverse':
        row['values'] = values
    training = {'epochs': 300, 'steps_per_epoch': 100, 'warmup_steps': 1000, 'learning_rate': .001, 'seed': 0}
    if method == 'kinn' and mode == 'inverse':
        training.update(alpha=100., data_tolerance=.0005)
    return {'schema_version': 1, 'method': method, 'mode': mode, 'species': species, 'surface_species': surface,
            'stoichiometry': matrix, 'observed_species': observed,
            ('rate_constants' if mode == 'forward' else 'initial_rate_constants'): ([2., 1.] if mode == 'forward' else [1.5, .7]),
            'datasets': [row], 'surrogate': {'layers': [16, 16], 'activations': 'tanh'},
            'training': training,
            'units': {'time': 's', 'bulk': 'normalized concentration', 'surface': 'fraction'}}


def _emit(value, output=None):
    text = json.dumps(value, allow_nan=False, indent=2) + '\n'
    if output:
        target = Path(output)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent, prefix='.' + target.name + '.', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        sys.stdout.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description='kinn: local forward and inverse kinetics, with fixed or MLE covariance weighting and SVD.')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('schema', help='Print the JSON problem schema.')
    sub.add_parser('capabilities', help='Print supported modes, activations and input conventions.')
    ex = sub.add_parser('example', help='Print a complete analytic example problem.')
    ex.add_argument('--kind', choices=('homogeneous', 'adsorption'), default='homogeneous')
    ex.add_argument('--mode', choices=('forward', 'inverse'), default='inverse')
    ex.add_argument('--method', choices=('kinn', 'rkinn'), default='rkinn',
                    help='Training formulation within kinn: rkinn = MLE covariance weighting and SVD (default); kinn = original fixed-weight loss.')
    ex.add_argument('--output')
    for command in ('validate', 'run'):
        child = sub.add_parser(command, help='Validate a problem without training.' if command == 'validate' else 'Run the forward or inverse problem locally.')
        child.add_argument('problem')
        child.add_argument('--output')
    args = parser.parse_args(argv)
    try:
        if args.command == 'schema':
            _emit(json.loads(files('kinn').joinpath('schema.json').read_text()))
        elif args.command == 'capabilities':
            from .problem import ACTIVATIONS
            _emit({'package': 'kinn', 'version': __version__, 'methods': {'kinn': 'Original fixed-weight loss within kinn: physics MSE + alpha * data MSE', 'rkinn': 'MLE extension within kinn: automatic covariance weighting, variance propagation and SVD'},
                   'modes': ['forward', 'inverse'], 'reactor': 'closed batch, mass-action kinetics',
                   'matrix_layout': 'species by directed elementary reaction; forward and reverse require separate columns',
                   'surface_constraint': 'one conserved site balance including vacant sites',
                   'observations': 'all nonsurface species, or all species; multiple datasets share kinetic parameters',
                   'surrogate': {'families': {'kinn': 'original nn_combo MLP and surface transform', 'rkinn': 'original nn_npt SVD-constrained MLP'}, 'hidden_layers': 'configurable', 'activations': ACTIVATIONS},
                   'variance': 'first-order residual propagation with automatic OAS covariance weighting; not posterior credible intervals',
                   'execution': 'local', 'network': 'none'})
        elif args.command == 'example':
            _emit(example(args.kind, args.mode, args.method), args.output)
        else:
            from .problem import load, describe
            if args.output and Path(args.output).resolve() == Path(args.problem).resolve():
                raise ValueError('output must differ from the input problem file')
            problem = load(args.problem)
            if args.command == 'validate':
                _emit(describe(problem), args.output)
            else:
                from .numerics import NumericalError
                if problem['method'] == 'kinn':
                    from .pareto import run
                else:
                    from .rkinn import run
                try:
                    result = run(problem)
                except NumericalError as error:
                    _emit({'schema_version': 1, 'status': 'numerical_failure', 'method': problem['method'], 'mode': problem['mode'], 'error': str(error)}, args.output)
                    return 1
                _emit(result, args.output)
                return 0 if result['status'] == 'converged' else 1
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({'error': str(error), 'status': 'invalid_input'}, allow_nan=False), file=sys.stderr)
        return 2
    return 0
