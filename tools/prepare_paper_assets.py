"""Convert trusted repository paper archives to numeric NPZ and JSON.

Run from a Git checkout, not against downloaded/user-supplied pickle files.
No trainer modules are imported. Original research files remain untouched.
"""
import ast
import hashlib
import io
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np


class ArrayReader(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            from numpy._core import multiarray
        except ImportError:
            from numpy.core import multiarray
        allowed = {('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
                   ('numpy.core.multiarray', '_reconstruct'): multiarray._reconstruct,
                   ('numpy.core.multiarray', 'scalar'): multiarray.scalar}
        if (module, name) not in allowed:
            raise ValueError(f'Unsupported archive global: {module}.{name}')
        return allowed[module, name]


def read_original(path):
    with zipfile.ZipFile(path) as archive:
        stream = io.BytesIO(archive.read('data.npy'))
        version = np.lib.format.read_magic(stream)
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        reader(stream)
        return ArrayReader(stream).load().item()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def extract_models(source):
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        while isinstance(target, ast.Subscript):
            target = target.value
        if isinstance(target, ast.Name) and target.id in ('pars', 't0', 'bcv', 'trig'):
            nodes.append(node)
    env = {'jnp': np, **{name: name for name in ('tanh', 'swish', 'sigmoid', 'gauss', 'sin', 'cos')}}
    # Only the trusted repository's model declarations are evaluated.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
    specs = plain(env['pars'])
    for key, spec in specs.items():
        spec['name'] = ['g', 'da', 'dc', 'dcs'][int(key)]
    return specs


def pack(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Nested object array')
        key = f'a{len(arrays)}'
        arrays[key] = value
        return {'array': key}
    if isinstance(value, np.generic):
        return pack(value.item(), arrays)
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [pack(v, arrays) for v in value]}
    if isinstance(value, dict):
        return {'dict': [[pack(k, arrays), pack(v, arrays)] for k, v in value.items()]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {'value': value}
    raise ValueError(type(value))


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'kinn/reproduction/data'
    out.mkdir(parents=True, exist_ok=True)
    source = root / 'paper/trainer_source.py'
    specs = extract_models(source)
    (out / 'models.json').write_text(json.dumps(specs, indent=2) + '\n')
    records = []
    for path in sorted((root / 'paper/database').glob('*.npz')):
        data = read_original(path)
        arrays = {}
        tree = pack(data, arrays)
        name = path.stem
        (out / (name + '.json')).write_text(json.dumps(tree, separators=(',', ':'), allow_nan=False) + '\n')
        np.savez_compressed(out / (name + '.npz'), **arrays)
        _, scenario, index, schedule = name.split('_', 3)
        records.append({'id': name, 'model': specs[index]['name'], 'model_index': int(index),
                        'scenario': scenario, 'schedule': schedule, 'epochs': int(data['epoch']),
                        'saved_states': len(data['iter_data']), 'source': str(path.relative_to(root)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'schema_version': 1, 'paper': '2011.14473v2', 'source_file': 'paper/trainer_source.py',
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'records': records,
                'source_scope': 'Archived training parameters and stage histories from the original repository; recalculating outputs is distinct from training afresh.',
                'missing_mle_archives': ['opt_invvwn_4_alpha107sens_uq_full.npz', 'trainer_invvwn_3_mle.npz']}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Converted {len(records)} archives; extracted {len(specs)} model specifications.')


if __name__ == '__main__':
    main()
