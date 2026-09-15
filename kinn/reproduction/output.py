"""Portable numerical exports and headless plots for paper experiments."""
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np


def environment():
    import jax
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'dependencies': {name: version(name) for name in ('jax', 'jaxlib', 'numpy', 'scipy', 'optax')},
            'jax_enable_x64': bool(jax.config.x64_enabled),
            'jax_default_prng_impl':str(jax.config.jax_default_prng_impl),
            'jax_threefry_partitionable':bool(jax.config.jax_threefry_partitionable),
            'devices': [str(d) for d in jax.devices()]}


def save_tree(directory, name, value):
    """Store a parameter/checkpoint tree as numeric arrays and a JSON structure."""
    import jax
    arrays = {}
    def pack(node):
        if isinstance(node, (np.ndarray, jax.Array)):
            key = f'a{len(arrays)}'
            arrays[key] = np.asarray(node)
            if arrays[key].dtype.hasobject:
                raise ValueError('Object arrays cannot be exported')
            return {'array': key}
        if isinstance(node, np.generic):
            return pack(node.item())
        if isinstance(node, dict):
            return {'dict': [[pack(k), pack(v)] for k, v in node.items()]}
        if isinstance(node, (list, tuple)):
            return {type(node).__name__: [pack(v) for v in node]}
        if node is None or isinstance(node, (str, bool, int, float)):
            return {'value': node}
        raise ValueError(f'Unsupported checkpoint value: {type(node).__name__}')
    tree = pack(value)
    directory = Path(directory)
    (directory / (name + '.json')).write_text(json.dumps(tree, allow_nan=False) + '\n')
    np.savez_compressed(directory / (name + '.npz'), **arrays)


def require_plots():
    try:
        import matplotlib
    except ImportError as error:
        raise ValueError('Plotting needs Matplotlib: python -m pip install "matplotlib>=3.8"') from error
    matplotlib.use('Agg')


def export(result, directory, *, plots=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays, datasets = {}, []
    for index, row in enumerate(result['datasets']):
        datasets.append({'metrics': row['metrics']})
        for key, value in row.items():
            if key != 'metrics':
                arrays[f'dataset_{index}_{key}'] = value
        with (directory / f'trajectory-{index + 1}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['time'] + [f'{kind}:{s}' for kind in ('truth', 'surrogate', 'derivative', 'rhs') for s in result['species']])
            writer.writerows(np.column_stack([row[k] for k in ('times', 'truth', 'states', 'derivatives', 'rhs')]))
        if row['observations'].shape[1]:
            with (directory / f'observations-{index+1}.csv').open('w',newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['time'] + result['species'][:row['observations'].shape[1]])
                writer.writerows(np.column_stack((row['times'],row['observations'])))
    np.savez_compressed(directory / 'arrays.npz', **arrays)
    report = {**result, 'datasets': datasets}
    (directory / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (directory / 'history.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'alpha', 'physics_mse', 'data_mse'])
        writer.writerows([r[k] for k in ('epoch', 'alpha', 'physics_mse', 'data_mse')] for r in result['history'])
    if plots:
        plot(result, directory)
    return report


def plot(result, directory):
    require_plots()
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
    for index, row in enumerate(result['datasets']):
        has_surface = len(result['species']) > 3
        fig, axes = plt.subplots(2 if has_surface else 1, 2, squeeze=False, figsize=(8, 6 if has_surface else 3.5), layout='constrained')
        for group, positions in enumerate((range(3), range(3, len(result['species']))) if has_surface else (range(3),)):
            trajectory, parity = axes[group]
            for j in positions:
                line, = trajectory.plot(row['times'], row['truth'][:, j], lw=1, label=result['species'][j])
                trajectory.plot(row['times'], row['states'][:, j], 'o', mfc='none', mec=line.get_color(), ms=2.5, mew=.6, markevery=3)
                if j < row['calibrated_observations'].shape[1]:
                    trajectory.scatter(row['times'],row['calibrated_observations'][:,j],color=line.get_color(),s=3,alpha=.3)
                parity.plot(row['rhs'][:, j], row['derivatives'][:, j], '.', ms=2.5, color=line.get_color())
            lo, hi = np.min([row['rhs'][:, positions], row['derivatives'][:, positions]]), np.max([row['rhs'][:, positions], row['derivatives'][:, positions]])
            parity.plot([lo, hi], [lo, hi], '--', color='gray', lw=.7)
            trajectory.set(xlabel='Time', ylabel='Concentration' if group == 0 else 'Coverage')
            trajectory.legend(ncol=3, fontsize=7)
            parity.set(xlabel='Physical model derivative', ylabel='Surrogate time derivative')
        fig.suptitle(f"{result['source']['model']} · {result['source']['scenario']} · IC {index + 1} · epoch {result['epoch']}")
        for fmt in ('svg', 'pdf'):
            fig.savefig(directory / f'trajectory-{index + 1}.{fmt}')
        plt.close(fig)
    if result['mode'] == 'inverse':
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout='constrained')
        truth, fitted = np.log(result['true_rate_constants']), np.log(result['rate_constants'])
        axes[0].plot(truth, fitted, 'o', ms=4)
        lo, hi = min(truth.min(), fitted.min()), max(truth.max(), fitted.max())
        axes[0].plot([lo, hi], [lo, hi], '--', color='gray', lw=.7)
        axes[0].set(xlabel='True ln(k)', ylabel='Fitted ln(k)')
        history = result['history']
        alpha = np.array([h['alpha'] for h in history])
        for mask, label, marker in ((np.r_[True, np.diff(alpha) >= 0], 'Tightening / refinement', 'o'), (np.r_[False, np.diff(alpha) < 0], 'Relaxation', 's')):
            axes[1].loglog(np.array([h['physics_mse'] for h in history])[mask], np.array([h['data_mse'] for h in history])[mask], marker=marker, ms=3, lw=.6, label=label)
        axes[1].set(xlabel='Physics MSE', ylabel='Data MSE')
        axes[1].legend(fontsize=7)
        for fmt in ('svg', 'pdf'):
            fig.savefig(directory / f'parameters-pareto.{fmt}')
        plt.close(fig)
