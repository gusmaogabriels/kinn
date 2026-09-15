import numpy as np
import jax
import jax.numpy as jnp
import pytest
from kinn.basis import model, nn
from kinn.basis.mle import _cov_oas
from kinn.cli import example
from kinn.problem import validate, describe
from kinn.rkinn import build, refresh, propagate_covariance, uncertainty_report


def test_pointwise_jacobians_match_analytic_mass_action_and_jit():
    matrix = jnp.array([[-1, 1], [1, -1]])
    physical = model(matrix, nobs=2)
    states = jnp.array([[1., 0.], [.2, .8], [.5, .5]])
    params = [jnp.log(jnp.array([2., 1.]))]
    expected = np.broadcast_to([[-2., 1.], [2., -1.]], (3, 2, 2))
    np.testing.assert_allclose(physical.diff_eval(params, [states])[0], expected)
    np.testing.assert_allclose(jax.jit(physical.diff_eval)(params, [states])[0], expected)
    jac = physical.diff_params(params, [states])[0]
    expected_p = np.stack([np.asarray(matrix) * np.array([2., 1.]) * x for x in np.asarray(states)])
    np.testing.assert_allclose(jac, expected_p)
    rate_jac = physical.diff_r(params, [states])[0]
    np.testing.assert_allclose(rate_jac, np.broadcast_to(np.diag([2., 1.]), (3, 2, 2)))


def test_parameter_derivatives_use_parameter_argument():
    net = nn([[1, 3, 2]], [jnp.tanh])
    times = jnp.array([[.2], [.4]])
    derivatives = net.d_pars(net.params[0], times)
    for actual, expected in zip(jax.tree_util.tree_leaves(derivatives), jax.tree_util.tree_leaves(jax.jacfwd(lambda p: net.batched_state(p, times))(net.params[0]))):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_oas_returns_shrunk_covariance_and_finite_isotropic_gradient():
    matrix = jnp.diag(jnp.array([1., 9.]))
    covariance, shrinkage = _cov_oas(matrix, 100)
    expected_shrinkage = 45.5 / 808.
    np.testing.assert_allclose(shrinkage, expected_shrinkage)
    np.testing.assert_allclose(covariance, np.diag([1 + 4*expected_shrinkage, 9 - 4*expected_shrinkage]))
    for c in [jnp.zeros((2, 2)), jnp.eye(2), jnp.array([[3.]])]:
        result, rho = _cov_oas(c, 10)
        assert np.isfinite(result).all() and 0 <= rho <= 1
        gradient = jax.grad(lambda x: jnp.sum(_cov_oas(x, 10)[0]))(c)
        assert np.isfinite(gradient).all()


def test_first_order_covariance_matches_nonlinear_monte_carlo():
    physical = model(jnp.array([[-1, 1], [1, -1]]), nobs=2)
    state = jnp.array([.7, .3]); logs = jnp.log(jnp.array([2., 1.]))
    state_cov = jnp.diag(jnp.array([2e-7, 3e-7])); log_cov = jnp.diag(jnp.array([1e-7, 4e-7]))
    jx = jax.jacfwd(lambda x: physical.single_eval([logs], [x]))(state)
    jp = jax.jacfwd(lambda p: physical.single_eval([p], [state]))(logs)
    propagated = propagate_covariance(jx, state_cov, jp, log_cov)
    rng = np.random.default_rng(123)
    xs = rng.multivariate_normal(state, state_cov, size=60000)
    ps = rng.multivariate_normal(logs, log_cov, size=60000)
    flux = np.exp(ps[:, 0])*xs[:, 0] - np.exp(ps[:, 1])*xs[:, 1]
    empirical = np.cov(np.stack([-flux, flux]))
    np.testing.assert_allclose(propagated, empirical, rtol=.025)
    assert np.linalg.eigvalsh(propagated).min() >= -1e-12


@pytest.mark.parametrize('kind', ['homogeneous', 'adsorption'])
def test_rkinn_initial_condition_and_conservation(kind):
    p = validate(example(kind, 'forward'))
    engine, data = build(p)
    assert engine.nn[0].nns[0].layers_sizes[0] == describe(p)['neural_layers']
    times = jnp.linspace(0, 1, 11)[:, None]
    states = engine.nn[0].batched_state(engine.params['sm'][0], times, engine.zn[0])
    initial = np.array(p['datasets'][0]['initial_state'])[p['order']]
    np.testing.assert_allclose(states[0], initial, atol=1e-12)
    conserved = np.asarray(states @ engine.model.Un)
    np.testing.assert_allclose(conserved, np.broadcast_to(initial @ engine.model.Un, conserved.shape), atol=1e-12)
    if kind == 'adsorption':
        assert np.asarray(states[:, 1:]).min() >= 0
        np.testing.assert_allclose(states[:, 1:].sum(axis=1), 1., atol=1e-12)
    derivative = engine.nn[0].diff_state(engine.params['sm'][0], times, engine.zn[0])
    assert np.isfinite(derivative).all()


def test_rank_deficient_parameters_do_not_report_zero_certainty():
    raw = example(); raw['stoichiometry'] = [[-1, -1], [1, 1]]
    p = validate(raw); engine, data = build(p); refresh(engine, data)
    report = uncertainty_report(engine, p)
    assert report['local_kinetic_sensitivity_rank'] == 1
    assert not report['full_state_sensitivity_identifiable']
    assert report['datasets'][0]['rate_error_covariance'] is None


def test_original_kinn_loss_matches_paper_weighted_residuals():
    from kinn.pareto import build as build_kinn
    raw = example(method='kinn'); raw['training']['alpha'] = 7
    p = validate(raw); engine, data = build_kinn(p)
    params = engine.params
    physics, measured, _ = engine.res_fun(params, [7.], data)
    expected = sum(np.mean(e**2) for e in physics)/len(data) + 7*sum(np.mean(e**2) for e in measured)/len(data)
    np.testing.assert_allclose(engine.loss(params, [7.], data), expected, rtol=1e-12)


def test_representation_error_propagation_is_finite():
    raw = example(); raw['surrogate']['layers'] = [3]
    raw['datasets'][0]['times'] = raw['datasets'][0]['times'][::10]
    raw['datasets'][0]['values'] = raw['datasets'][0]['values'][::10]
    p = validate(raw); engine, data = build(p)
    refresh(engine, data, representation_error=True)
    for x in jax.tree_util.tree_leaves(engine.omegas):
        assert np.isfinite(x).all()


def test_original_dcs_rate_law_and_conservation():
    # The seven forward/reverse pairs in the original README (10 species).
    matrix = jnp.array([
        [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1],
        [0, 0, 0, 0, 0, 0, 2, -2, 0, 0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, -1, 1, -1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
        [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1]])
    physical = model(matrix, nobs=3)
    x = jnp.array([.3, .4, .1, .1, .1, .1, .1, .1, .1, .4])
    rates = jnp.arange(1., 15.)
    a,b,c,aa,bb,cc,d,e,f,site = x
    factors = jnp.array([a*site,aa,b*site,bb,c*site,cc,aa*site,d*d,bb*site,e*e,d*e,f*site,f*e,cc*site])
    expected = matrix @ (rates*factors)
    actual = physical.batched_eval([jnp.log(rates)], [x[None, :]])[0]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert abs(float(actual[3:].sum())) < 1e-12
    np.testing.assert_allclose(physical.Un.T @ actual, 0, atol=1e-12)
    assert physical.diff_eval([jnp.log(rates)], [x[None, :]])[0].shape == (1, 10, 10)
