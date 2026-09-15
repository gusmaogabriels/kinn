import copy
import json
from importlib.resources import files
import pytest
from jsonschema import Draft202012Validator
from kinn.cli import example, main
from kinn.problem import ProblemError, describe, load, validate


@pytest.mark.parametrize('method', ['kinn', 'rkinn'])
@pytest.mark.parametrize('mode', ['forward', 'inverse'])
@pytest.mark.parametrize('kind', ['homogeneous', 'adsorption'])
def test_examples_match_schema_and_dimensions(method, mode, kind):
    raw = example(kind, mode, method)
    schema = json.loads(files('kinn').joinpath('schema.json').read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(raw)
    p = validate(raw)
    info = describe(p)
    assert info['method'] == method and info['mode'] == mode
    assert info['matrix_shape'] == [len(raw['species']), 2]
    assert info['stoichiometric_rank'] == 1
    assert info['execution'] == 'local'


@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('method', 'unknown'), ('mode', 'unknown'),
    ('rate_constants', [1, 2]), ('initial_rate_constants', [True, 1.]),
    ('initial_rate_constants', [0, 1]), ('initial_rate_constants', [float('nan'), 1]),
    ('initial_rate_constants', [1]), ('species', ['A', 'A']),
    ('stoichiometry', [[-1e50, 1], [1e50, -1]]),
    ('stoichiometry', [[-.5, .5], [.5, -.5]]),
    ('observed_species', ['A']), ('unknown', 1),
    ('training', {'alpha': 1}), ('training', {'epochs': 0}),
    ('training', {'seed': 2**32}), ('training', {'warmup_steps': -1}),
    ('surrogate', {'layers': [0]}), ('surrogate', {'activations': '__import__("os")'}),
    ('surrogate', {'activations': ['tanh']}), ('uncertainty', {'representation_error': 'yes'})
])
def test_invalid_scientific_input(field, value):
    raw = example()
    raw[field] = value
    with pytest.raises(ProblemError):
        validate(raw)


def test_initial_surface_site_balance_is_checked():
    raw = example('adsorption')
    raw['datasets'][0]['initial_state'] = [.8, .2, .7]
    with pytest.raises(ProblemError, match='sum to one'):
        validate(raw)
    raw = example('adsorption')
    raw['stoichiometry'][-1] = [0, 0]
    with pytest.raises(ProblemError, match='site balance'):
        validate(raw)


def test_missing_forward_initial_and_repeated_times():
    raw = example(mode='forward')
    del raw['datasets'][0]['initial_state']
    with pytest.raises(ProblemError, match='initial_state'):
        validate(raw)
    raw = example()
    raw['datasets'][0]['times'][1] = 0
    with pytest.raises(ProblemError, match='increasing'):
        validate(raw)


def test_csv_paths_and_observation_order(tmp_path):
    (tmp_path/'data.csv').write_text('B,time,A\n0,0,1\n0.3,0.5,0.7\n0.4,1,0.6\n')
    raw = example()
    raw['observed_species'] = ['B', 'A']
    raw['datasets'] = [{'data_file': 'data.csv', 'initial_state': [1, 0]}]
    path = tmp_path/'problem.json'
    path.write_text(json.dumps(raw))
    p = load(path)
    assert p['datasets'][0]['values'] == [[1., 0.], [.7, .3], [.6, .4]]
    assert p['observed_species'] == ['A', 'B']


def test_nonfinite_and_duplicate_json_keys(tmp_path):
    path = tmp_path/'bad.json'
    for text in ('{"mode":"inverse","mode":"forward"}', '{"rate_constants":[NaN]}'):
        path.write_text(text)
        with pytest.raises(ProblemError):
            load(path)


def test_validate_and_cli_do_not_import_jax(tmp_path):
    import subprocess, sys
    path = tmp_path/'input.json'; path.write_text(json.dumps(example()))
    result = subprocess.run([sys.executable, '-c',
        'import sys; from kinn.cli import main; assert main(["validate",sys.argv[1]]) == 0; assert "jax" not in sys.modules', str(path)],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['valid']


def test_cli_cannot_overwrite_input(tmp_path, capsys):
    path = tmp_path/'input.json'; text = json.dumps(example()); path.write_text(text)
    assert main(['run', str(path), '--output', str(path)]) == 2
    assert json.loads(capsys.readouterr().err)['status'] == 'invalid_input'
    assert path.read_text() == text


def test_forward_and_inverse_require_different_rate_fields_in_schema():
    schema = Draft202012Validator(json.loads(files('kinn').joinpath('schema.json').read_text()))
    for mode, field in [('forward','rate_constants'), ('inverse','initial_rate_constants')]:
        raw = example(mode=mode); del raw[field]
        assert list(schema.iter_errors(raw))
        with pytest.raises(ProblemError): validate(raw)
