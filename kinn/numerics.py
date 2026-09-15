"""Shared local diagnostics; no training runs or network calls at import."""
import numpy as np
import jax


class NumericalError(RuntimeError):
    """Training produced invalid floating-point values."""


def require_finite(tree, stage):
    if not all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree)):
        raise NumericalError(f'nonfinite values during {stage}; rescale inputs or reduce the learning rate')


def boundary_gain(problem, physical, initial):
    gain = problem['surrogate']['boundary_gain']
    if gain != 'auto':
        return float(gain)
    if initial is None or not problem['surface_species']:
        return 1.
    nbulk = problem['n_bulk']
    derivative = np.asarray(physical.single_eval([np.log(problem['rates'])], [initial]))[nbulk:]
    coverage = np.asarray(initial)[nbulk:]
    capacity = np.where(derivative >= 0, 1. - coverage, coverage)
    # A bounded target cannot meet an initial slope faster than its tanh gate.
    # Give the original gate enough gain for the known initial rate law.
    gain = max(1., 4. * float(np.max(np.abs(derivative) / np.maximum(capacity, 1e-12))))
    require_finite(gain, 'initial-condition scaling')
    return gain


def constraints(problem, predictions):
    surface = [problem['species'].index(name) for name in problem['surface_species']]
    matrices = [np.asarray(x) for x in predictions]
    minimum = min(float(x.min()) for x in matrices)
    balance = max((float(np.abs(x[:, surface].sum(axis=1) - 1.).max()) for x in matrices), default=0.) if surface else 0.
    # Report exact diagnostics separately from convergence; a finite residual
    # tolerance is not a proof of positivity between sampled points.
    return {'minimum_sampled_state': minimum,
            'nonnegative_at_sampled_points': minimum >= -1e-10,
            'maximum_surface_site_balance_error': balance,
            'surface_site_balance_satisfied': balance <= 1e-8,
            'scope': 'Observation and collocation points only; no guarantee between samples.'}


def timing(started, setup, warmup, kernels, epochs):
    from time import perf_counter
    return {'setup_seconds': setup, 'data_warmup_seconds': warmup,
            'first_epoch_seconds': epochs[0], 'warm_epoch_seconds': epochs[1:],
            'first_optimization_kernel_seconds': kernels[0],
            'warm_optimization_kernel_seconds': kernels[1:],
            'wall_seconds': perf_counter() - started}
