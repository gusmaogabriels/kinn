"""Reproduction of the experiments preserved with the KINN papers."""
import json
from importlib.resources import files


def resource(name):
    return files(__package__).joinpath('data', name)


def catalog():
    return json.loads(resource('manifest.json').read_text())


def models():
    return json.loads(resource('models.json').read_text())


def record(identifier):
    for row in catalog()['records']:
        if row['id'] == identifier:
            return row
    raise ValueError(f'Unknown archived experiment: {identifier}; use kinn reproduce list')


def load_archive(identifier):
    """Load a bundled experiment without pickle or executable object arrays."""
    import io
    import numpy as np
    record(identifier)
    tree = json.loads(resource(identifier + '.json').read_text())
    with np.load(io.BytesIO(resource(identifier + '.npz').read_bytes()), allow_pickle=False) as arrays:
        return _restore(tree, arrays)


def _restore(tree, arrays):
    def unpack(node):
        if 'array' in node:
            return arrays[node['array']].copy()
        if 'dict' in node:
            return {unpack(k): unpack(v) for k, v in node['dict']}
        if 'list' in node:
            return [unpack(v) for v in node['list']]
        if 'tuple' in node:
            return tuple(unpack(v) for v in node['tuple'])
        return node['value']
    return unpack(tree)


def load_export(directory, name='parameters'):
    """Read a CLI numerical export without executing historical pickle data."""
    from pathlib import Path
    import numpy as np
    if name not in ('parameters', 'checkpoints', 'checkpoint'):
        raise ValueError('name must be parameters, checkpoints, or checkpoint')
    directory = Path(directory)
    tree = json.loads((directory / (name+'.json')).read_text())
    with np.load(directory / (name+'.npz'), allow_pickle=False) as arrays:
        return _restore(tree, arrays)
