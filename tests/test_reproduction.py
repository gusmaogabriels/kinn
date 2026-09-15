import hashlib
import io
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from kinn.cli import main
from kinn.reproduction import catalog, load_archive, load_export, resource
from kinn.reproduction.published import verify
from kinn.reproduction.reference import evaluate
from kinn.reproduction.training import plan, train


def test_published_tables_preserve_matches_and_expose_discrepancy():
    report = verify()
    assert report['checks'] == 184
    assert report['matched'] == 183
    assert report['status'] == 'differences_found'
    assert len(report['differences']) == 1
    difference = report['differences'][0]
    assert (difference['table'], difference['experiment'], difference['metric']) == (5,'trainer_invvwn_0_alpha13','log_rate_mae')
    assert difference['published'] == '2.29e-4'
    np.testing.assert_allclose(difference['calculated'], .022917068551180413, rtol=1e-10)


def test_all_archives_are_numerical_and_preserve_original_provenance():
    from tools.prepare_paper_assets import read_original
    root = Path(__file__).resolve().parents[1]
    records = catalog()['records']
    assert len(records) == 21
    for record in records:
        original = root / record['source']
        if original.exists():  # Historical source archives are in Git, not the wheel.
            assert hashlib.sha256(original.read_bytes()).hexdigest() == record['sha256']
        with np.load(io.BytesIO(resource(record['id']+'.npz').read_bytes()), allow_pickle=False) as arrays:
            assert all(not arrays[key].dtype.hasobject for key in arrays.files)
        archive = load_archive(record['id'])
        if original.exists():
            original_data = read_original(original)
            original_leaves, original_tree = jax.tree_util.tree_flatten(original_data)
            converted_leaves, converted_tree = jax.tree_util.tree_flatten(archive)
            assert original_tree == converted_tree
            for expected, actual in zip(original_leaves, converted_leaves):
                np.testing.assert_array_equal(actual, expected)
        assert archive['epoch'] == record['epochs']
        assert len(archive['iter_data']) == record['saved_states']


def test_cli_export_preserves_weights_checkpoints_and_physical_results(tmp_path, capsys):
    target = tmp_path/'results'
    identifier = 'trainer_fwd_0_alpha13'
    assert main(['reproduce','run',identifier,'--output',str(target)]) == 0
    expected = load_archive(identifier)['params']
    actual = load_export(target)
    a, a_tree = jax.tree_util.tree_flatten(expected)
    b, b_tree = jax.tree_util.tree_flatten(actual)
    assert a_tree == b_tree
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x,y)
    assert load_export(target,'checkpoints')['epoch'] == 3000
    with np.load(target/'arrays.npz',allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays['dataset_0_states'][0], [.6,.4,0.], atol=1e-12)
        assert arrays['dataset_0_derivatives'].shape == (100,3)
    assert main(['reproduce','run',identifier,'--output',str(target)]) == 2
    assert 'empty output directory' in capsys.readouterr().err


def test_noisy_paper_selection_uses_original_checkpoint_and_unweighted_history():
    r = evaluate('trainer_invvwn_3_alpha13')
    assert r['selected_state'] == 3 and r['epoch'] == 4000
    assert r['history'][3]['alpha'] > 300
    np.testing.assert_allclose(r['history'][3]['data_mse'], .0001689, rtol=.001)


def test_paper_plan_uses_archive_architecture_and_continuation_schedule():
    recipe = plan('trainer_invsc_1_alpha13')
    assert recipe['neural_layers'][0] == [1,12,12,12,6]
    assert len(recipe['stages']) == 12
    pareto = plan('trainer_invsc_3_alpha17sens')
    assert len(pareto['stages']) == 61
    assert pareto['stages'][20]['alpha'] == pytest.approx(1e6)
    assert pareto['stages'][-1]['alpha'] == pytest.approx(1e-6)
    with pytest.raises(ValueError,match='iteration count'):
        plan('trainer_invsc_3_alpha13sens')


@pytest.mark.parametrize('identifier', ['trainer_fwd_0_alpha13','trainer_invsc_1_alpha13'])
def test_fresh_training_runs_compiled_stages_and_preserves_constraints(identifier):
    result, archive = train(identifier,epochs_per_stage=2,steps_per_epoch=2,max_stages=2)
    assert result['status'] == 'budget_override'
    assert result['origin'] == 'fresh_training'
    assert result['epoch'] == 4
    assert len(archive['iter_data']) == 2
    assert result['timing']['compilation_seconds'] > 0
    assert result['timing']['warm_training_seconds'] > 0
    for row in result['datasets']:
        assert np.isfinite(row['states']).all()
        if result['mode'] == 'inverse':
            np.testing.assert_allclose(row['states'][:,3:].sum(axis=1),1.,atol=1e-12)
        else:
            np.testing.assert_allclose(row['states'][0],row['truth'][0],atol=1e-12)
    first = jax.tree_util.tree_leaves(archive['iter_data'][0][1][0])
    last = jax.tree_util.tree_leaves(archive['params'])
    assert any(not np.array_equal(a,b) for a,b in zip(first,last))


def test_invalid_reproduction_requests_do_not_create_outputs(tmp_path, capsys):
    for identifier, state in [('../trainer_fwd_0_alpha13','paper'),('trainer_fwd_0_alpha13','900'),('all','-1')]:
        target = tmp_path/'output'
        assert main(['reproduce','run',identifier,'--state',state,'--output',str(target)]) == 2
        assert not target.exists()


def test_reproduction_plot_exports(tmp_path):
    pytest.importorskip('matplotlib')
    assert main(['reproduce','run','trainer_invvwn_3_alpha13','--plots','--output',str(tmp_path/'figures')]) == 0
    for name in ('trajectory-1','trajectory-2','parameters-pareto'):
        assert (tmp_path/'figures'/f'{name}.pdf').read_bytes().startswith(b'%PDF')
        assert '<svg' in (tmp_path/'figures'/f'{name}.svg').read_text()


def test_paper_training_failure_is_reported_without_claiming_completion(tmp_path,monkeypatch):
    from kinn.numerics import NumericalError
    def fail(*args,**kwargs):
        raise NumericalError('nonfinite training state')
    monkeypatch.setattr('kinn.reproduction.training.train',fail)
    directory = tmp_path/'failed'
    assert main(['reproduce','train','trainer_fwd_0_alpha13','--output',str(directory)]) == 1
    report = json.loads((directory/'failure.json').read_text())
    assert report['status'] == 'numerical_failure'
    assert not (directory/'result.json').exists()
