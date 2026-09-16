import json
import numpy as np
import pytest
from kinn import solve
from kinn.cli import example, main


@pytest.mark.parametrize('method', ['fixed', 'mle'])
@pytest.mark.parametrize('mode', ['forward', 'inverse'])
@pytest.mark.parametrize('kind', ['homogeneous', 'adsorption'])
def test_analytic_forward_and_inverse_problems(method, mode, kind, tmp_path):
    raw = example(kind, mode, method)
    result = solve(raw)
    assert result['method'] == method
    (tmp_path/f'{method}-{kind}-{mode}.json').write_text(json.dumps(result, allow_nan=False))
    assert result['status'] == 'converged', result['history'][-1]
    states = np.array(result['predictions'][0]['states'])
    expected = np.array(example(kind, 'inverse')['datasets'][0]['values'])
    # Independent closed-form trajectories generated in the public examples.
    np.testing.assert_allclose(states[:, :expected.shape[1]], expected, atol=.015, rtol=0)
    np.testing.assert_allclose(states[0], raw['datasets'][0]['initial_state'], atol=1e-12)
    if mode == 'forward':
        np.testing.assert_allclose(result['rate_constants'], [2, 1], rtol=1e-14)
    else:
        np.testing.assert_allclose(result['rate_constants'], [2, 1], rtol=.08 if method == 'fixed' else .04)
    assert result['physical_checks']['nonnegative_at_sampled_points']
    assert result['physical_checks']['surface_site_balance_satisfied']
    times = result['timing']
    assert times['wall_seconds'] >= times['setup_seconds'] + sum(times['warm_epoch_seconds']) + times['first_epoch_seconds']
    assert len(times['warm_epoch_seconds']) == len(result['history']) - 1
    if method == 'mle' and mode == 'inverse':
        uncertainty = result['uncertainty']
        assert uncertainty['full_state_sensitivity_identifiable']
        for key in ('state_residual_covariance', 'log_rate_error_covariance', 'rate_error_covariance'):
            matrix = np.array(uncertainty['datasets'][0][key])
            assert np.isfinite(matrix).all()
            np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
            assert np.linalg.eigvalsh(matrix).min() >= -1e-10
    if method == 'fixed':
        assert result['uncertainty']['method'] == 'not_estimated'


@pytest.mark.parametrize('method', ['fixed', 'mle'])
def test_multiple_datasets_unequal_samples_and_time_units(method):
    raw = example(method=method)
    row = raw['datasets'][0]
    raw['datasets'].append({'times': row['times'][::2], 'values': row['values'][::2], 'initial_state': [1., 0.]})
    # Changing seconds to milliseconds must rescale rates and no stoichiometry.
    for row in raw['datasets']:
        row['times'] = [t*1000 for t in row['times']]
    raw['initial_rate_constants'] = [.0015, .0007]
    raw['training'].update(epochs=1, steps_per_epoch=5, warmup_steps=10, physics_tolerance=1e-14, data_tolerance=1e-14)
    result = solve(raw)
    assert result['status'] == 'max_epochs'
    assert [len(row['states']) for row in result['predictions']] == [21, 11]
    assert result['time_scale'] == 1000
    assert np.isfinite(result['rate_constants']).all()
    assert max(result['rate_constants']) < .01


@pytest.mark.parametrize('method', ['fixed', 'mle'])
def test_inverse_can_infer_without_initial_state(method):
    raw = example(method=method)
    del raw['datasets'][0]['initial_state']
    raw['training'].update(epochs=1, steps_per_epoch=5, warmup_steps=10)
    result = solve(raw)
    assert len(result['predictions'][0]['states'][0]) == 2
    assert np.isfinite(result['predictions'][0]['states']).all()


def test_cli_nonconvergence_and_numerical_failure_are_machine_readable(tmp_path, capsys, monkeypatch):
    raw = example(); raw['training'].update(epochs=1, steps_per_epoch=1, warmup_steps=0, physics_tolerance=1e-14, data_tolerance=1e-14)
    path = tmp_path/'problem.json'; path.write_text(json.dumps(raw))
    output = tmp_path/'result.json'
    assert main(['run', str(path), '--output', str(output)]) == 1
    assert json.loads(output.read_text())['status'] == 'max_epochs'
    from kinn.numerics import NumericalError
    import kinn.rkinn
    def fail(problem):
        raise NumericalError('nonfinite values during optimization')
    monkeypatch.setattr(kinn.rkinn, 'run', fail)
    assert main(['run', str(path)]) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'numerical_failure'
