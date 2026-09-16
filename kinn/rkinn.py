"""Local rKINNs driver built on the original model, nn_npt and MLE optimizer."""
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .basis import model, nn, nn_npt, opt
from .basis.mle import _cov_oas
from .numerics import boundary_gain, constraints, require_finite, timing


ACTIVATIONS = {'tanh': jnp.tanh, 'sigmoid': jax.nn.sigmoid,
               'swish': jax.nn.swish, 'softplus': jax.nn.softplus,
               'sin': jnp.sin, 'gaussian': lambda x: jnp.exp(-x*x)}


@jax.jit
def propagate_covariance(state_jacobian, state_covariance, parameter_jacobian, parameter_covariance):
    """First-order covariance for independent state and log-rate errors."""
    return (state_jacobian @ state_covariance @ jnp.swapaxes(state_jacobian, -1, -2)
            + parameter_jacobian @ parameter_covariance @ jnp.swapaxes(parameter_jacobian, -1, -2))


@jax.jit
def regularize(covariance):
    """Symmetric PSD floor for numerical inversion, separate from OAS shrinkage."""
    if covariance.shape[-1] == 0:
        return covariance
    covariance = (covariance + covariance.T) / 2.
    values, vectors = jnp.linalg.eigh(covariance)
    floor = jnp.maximum(jnp.max(jnp.abs(values)) * 1e-8, 1e-12)
    return (vectors * jnp.maximum(values, floor)) @ vectors.T


class _ScaledModel(model):
    def __init__(self, matrix, nobs, time_scale):
        self.time_scale = time_scale
        super().__init__(matrix, nobs=nobs)

    def single_eval(self, params, batch):
        # Scale the derivative, never the stoichiometry/power-law exponents.
        return self.time_scale * super().single_eval(params, batch)


class _LocalOpt(opt):
    """Repair forward-mode wiring and project propagated covariance consistently."""
    def _res_fun(self, params, batch, zn):
        if self.mode == 'inverse':
            return super()._res_fun(params, batch, zn)
        ts, xs, tsm = [[row[j] for row in batch] for j in range(3)]
        ep, xp, dp, fp = self.res_pm(params, tsm, zn)
        ei, xi, di, fi = self.res_pm(params, ts, zn)
        errors = {'pm': {'pm': ep, 'interp': ei},
                  'interp': [jnp.zeros_like(x) for x in xi]}
        return errors, {'pm': xp, 'interp': xi}, {'pm': dp, 'interp': di}, {'pm': fp, 'interp': fi}, ts, xs, tsm

    def _upd_omegas(self, xs_sm, sigmas, projs, mus, variances):
        weights = {'pm': {'r': [], 'n': []}, 'interp': {'r': [], 'n': []}}
        constants = {'pm': [], 'interp': []}
        for i, x in enumerate(xs_sm['pm']):
            if self.mode == 'forward':
                covariance = sigmas['pm']['pm'][i]
                reduced = self.Ur.T @ covariance @ self.Ur
                reduced = regularize(_cov_oas(reduced, len(x))[0])
                weights['pm']['r'].append(jnp.linalg.inv(reduced))
            else:
                dx = projs['dxs']['xs']['pm'][i] - projs['pm']['xs']['pm'][i]
                dp = projs['pm']['mpars']['pm'][i]
                covariance = propagate_covariance(dx, sigmas['interp'][i], dp, sigmas['mpars'][i])
                reduced = self.quad_nv(self.Ur.T, covariance)
                reduced = jax.vmap(lambda c: regularize(_cov_oas(c, len(x))[0]))(reduced)
                weights['pm']['r'].append(jax.vmap(jnp.linalg.inv)(reduced))
            weights['pm']['n'].append(jnp.zeros((self.Un.shape[1], self.Un.shape[1])))
            if self.mode == 'inverse':
                # A marginal covariance matches the actually observed variables;
                # it does not condition on unobserved residuals as if measured.
                measured = sigmas['interp'][i][:self.ndata, :self.ndata]
                projected = self.proj_r.T @ measured @ self.proj_r
                projected = regularize(_cov_oas(projected, len(xs_sm['interp'][i]))[0])
                weights['interp']['r'].append(jnp.linalg.inv(projected))
                weights['interp']['n'].append(jnp.zeros((self.proj_n.shape[1], self.proj_n.shape[1])))
        return weights, constants

    def _err_fun(self, errors, weights):
        physics = []
        for i, residual in enumerate(errors['pm']['pm']):
            projected = self.proj(self.Ur.T, residual)
            quadratic = self.quad_vv if self.mode == 'inverse' else self.quad_vn
            physics.append(jnp.mean(quadratic(projected, weights['pm']['r'][i])))
        if self.mode == 'forward':
            return [jnp.stack(physics)]
        measured = [jnp.mean(self.quad_vn(self.proj(self.proj_r.T, residual[:, :self.ndata]), weights['interp']['r'][i]))
                    for i, residual in enumerate(errors['interp'])]
        return [jnp.stack(physics), jnp.stack(measured)]

    def _infer_exl(self, pm_dx, pm_dp, edx, ex):
        # Blr already spans site-conserving latent directions; imposing another
        # sum constraint on its coordinates would incorrectly force them to zero.
        dp = self.infer_dp(pm_dp, edx)
        gsl = jnp.dot(pm_dx[:, :, self.nobs:], self.Blr)
        residual = edx - jnp.einsum('tij,tj->ti', pm_dx[:, :, :self.nobs], ex[:, :self.nobs]) - jnp.einsum('tij,j->ti', pm_dp, dp.ravel())
        latent = jax.vmap(lambda a, b: jnp.linalg.pinv(a) @ b)(gsl, residual)
        return latent @ self.Blr.T, dp


def build(problem):
    """Build original rKINNs objects; all numerical state is explicit in params."""
    nbulk = problem['n_bulk']
    order = jnp.asarray(problem['order'], dtype=jnp.int32)
    matrix = jnp.asarray(problem['stoichiometry'])[order]
    time_scale = problem['time_scale']
    physical = _ScaledModel(matrix, nbulk, time_scale)
    rates = jnp.asarray(problem['rates'])
    physical.params = [rates]  # Historical opt constructor accepts positive rates.
    nets, data, nulls = [], [], []
    architecture = problem['surrogate']
    for index, dataset in enumerate(problem['datasets']):
        nstate = len(order)
        raw = nn([[1, *architecture['layers'], nstate - bool(problem['surface_species'])]],
                 [ACTIVATIONS[name] for name in architecture['activations']],
                 nn_scale=architecture['init_scale'])
        # Preserve historical initialization while giving each dataset its own key.
        raw.seed = (problem['training']['seed'] + index) % 2**32
        x0 = jnp.asarray(dataset['initial_state'])[order] if dataset['initial_state'] is not None else None
        if x0 is None and not problem['surface_species']:
            x0 = jnp.asarray(dataset['values']).mean(axis=0)
        null = x0 @ physical.Un if x0 is not None else jnp.empty(0)
        net = nn_npt([raw], physical, nobs=nbulk,
                     trig=bool(problem['surface_species']), damp=False,
                     gain=boundary_gain(problem, physical, x0),
                     mode='forward' if dataset['initial_state'] is not None else 'inverse', bc=(jnp.array([0.]), x0),
                     out_n=null, nn_scale=architecture['init_scale'])
        if len(null):
            net.set_nullspace(null)
        nets.append(net)
        nulls.append(null)
        start = dataset['times'][0]
        times = (jnp.asarray(dataset['times']) - start)[:, None] / time_scale
        collocation = (jnp.asarray(dataset['collocation_times']) - start)[:, None] / time_scale
        values = jnp.asarray(dataset['values']) if problem['mode'] == 'inverse' else jnp.zeros((len(times), nstate))
        data.append((times, values, collocation))
    inference = problem['mode'] == 'inverse' and len(problem['observed_species']) < len(order)
    trainer = _LocalOpt(nets, physical, mode=problem['mode'], nobs=nbulk, inference=inference,
                       num_iter=problem['training']['steps_per_epoch'], num_epochs=problem['training']['epochs'])
    trainer.ndata = len(problem['observed_species']) if problem['mode'] == 'inverse' else nbulk
    trainer.params['pm'] = [jnp.log(rates)]
    trainer.params['scales'] = jnp.ones(len(order) - nbulk) if not inference else jnp.empty(0)
    trainer.zn = nulls
    learning_rate = problem['training']['learning_rate']
    if problem['training']['learning_rate_schedule'] == 'cosine':
        learning_rate = optax.cosine_decay_schedule(learning_rate, problem['training']['epochs'] * problem['training']['steps_per_epoch'], alpha=.01)
    optimizer = {'optimizer': optax.adam, 'kwargs': {'learning_rate': learning_rate}}
    frozen = {'optimizer': optax.set_to_zero, 'kwargs': {}}
    trainer.set_optimizer({'sm': optimizer, 'pm': optimizer if problem['mode'] == 'inverse' else frozen, 'scales': frozen})
    trainer.initialize()
    trainer.grads = jax.jit(jax.grad(trainer._loss))
    trainer.subproj = jnp.eye(len(rates))
    return trainer, data


def refresh(trainer, data, *, representation_error=False):
    if trainer.mode == 'inverse':
        trainer.upd(trainer.params, data, trainer.zn, reperror=representation_error, mureg=False, center=False)
    else:
        (trainer.errs, trainer.xs_sm, trainer.xs_sm_dt, trainer.xs_pm_dt,
         trainer.ts, trainer.xs, trainer.tsm) = trainer.res_fun(trainer.params, data, trainer.zn)
        trainer.sigmas, trainer.mus, trainer.vars = trainer.get_stats(trainer.errs, False)
        trainer.omegas, trainer.lcte = trainer.upd_omegas(trainer.xs_sm, trainer.sigmas, {}, trainer.mus, trainer.vars)
    return trainer.omegas


def uncertainty_report(trainer, problem):
    """Expose the original first-order error propagation with explicit scope."""
    inverse_order = np.argsort(problem['order'])
    scales = problem['time_scale']
    details = {'method': 'first_order_residual_propagation', 'covariance_estimator': 'empirical_residual_second_moments',
               'weight_regularization': 'OAS shrinkage and a numerical eigenvalue floor',
               'scope': 'Empirical residual/error covariances, not posterior or repeated-sample estimator confidence intervals.',
               'assumptions': 'Local linearization; state and log-rate errors are treated as independent in the propagated covariance.',
               'numerical_covariance_floor': 'max(1e-12, 1e-8 * largest absolute eigenvalue) when inverting weights',
               'datasets': []}
    if trainer.mode == 'forward':
        details['parameter_uncertainty'] = 'not_estimated: rate constants are fixed inputs'
        for covariance in trainer.sigmas['pm']['pm']:
            array = np.asarray(covariance)[np.ix_(inverse_order, inverse_order)] / scales**2
            details['datasets'].append({'physics_defect_covariance': array.tolist()})
        return details
    sensitivity = np.concatenate([np.asarray(j) for j in trainer.projs['pm']['mpars']['interp']], axis=0).reshape(-1, len(problem['rates']))
    details['parameter_covariance_estimator'] = 'OAS'
    norm = np.linalg.norm(sensitivity, axis=0)
    scaled = sensitivity / np.where(norm > 0, norm, 1.)
    singular = np.linalg.svd(scaled, compute_uv=False)
    threshold = (singular[0] if len(singular) else 0.) * 1e-8
    rank = int(np.sum(singular > threshold))
    identifiable = rank == len(problem['rates'])
    details['local_kinetic_sensitivity_rank'] = rank
    details['kinetic_parameters'] = len(problem['rates'])
    details['full_state_sensitivity_identifiable'] = identifiable
    details['identifiability_scope'] = 'Kinetic RHS sensitivity along the inferred full-state trajectory; this does not prove identifiability from partial observations.'
    rates = np.asarray(jnp.exp(trainer.params['pm'][0]))
    for i, covariance in enumerate(trainer.sigmas['interp']):
        cp = trainer.sigmas['mpars'][i]
        jx = trainer.projs['pm']['xs']['interp'][i]
        jp = trainer.projs['pm']['mpars']['interp'][i]
        propagated = propagate_covariance(jx, covariance, jp, cp)
        state = np.asarray(covariance)[np.ix_(inverse_order, inverse_order)]
        derivative = np.asarray(propagated)[:, inverse_order][:, :, inverse_order] / scales**2
        param = np.asarray(cp)
        details['datasets'].append({'state_residual_covariance': state.tolist(),
            'log_rate_error_covariance': param.tolist() if identifiable else None,
            'rate_error_covariance': (rates[:, None] * param * rates[None, :]).tolist() if identifiable else None,
            'rhs_error_covariance': derivative.tolist(),
            'rate_error_status': 'local_linearized_estimate' if identifiable else 'rank_deficient; unidentifiable directions must not be interpreted as zero uncertainty'})
    return details


def run(problem):
    started = perf_counter()
    trainer, data = build(problem)
    warmup_started = perf_counter()
    if trainer.mode == 'inverse' and problem['training']['warmup_steps']:
        warmup_optimizer = optax.adam(problem['training']['learning_rate'])
        warmup_state = warmup_optimizer.init(trainer.params)
        def data_loss(params):
            predictions = trainer.pred_sm(params, [row[0] for row in data], trainer.zn)
            return sum(jnp.mean((x[:, :trainer.ndata] - row[1]) ** 2) for x, row in zip(predictions, data)) / len(data)

        @jax.jit
        def warmup(params, state):
            def update(carry, _):
                gradients = jax.grad(data_loss)(carry[0])
                updates, state = warmup_optimizer.update(gradients, carry[1], carry[0])
                return (optax.apply_updates(carry[0], updates), state), None
            return jax.lax.scan(update, (params, state), None, length=problem['training']['warmup_steps'])[0]
        trainer.params, _ = warmup(trainer.params, warmup_state)
        jax.block_until_ready(trainer.params)
        trainer.opt_state = trainer.optimizer.init(trainer.params)
    warmup_seconds = perf_counter() - warmup_started
    refresh(trainer, data, representation_error=problem['uncertainty']['representation_error'])
    require_finite((trainer.params, trainer.omegas), 'initial covariance estimation')
    steps = problem['training']['steps_per_epoch']

    @jax.jit
    def epoch(params, state, weights):
        def update(carry, _):
            params, state = trainer.step(carry[1], carry[0], weights, data, trainer.zn, trainer.subproj)
            return (params, state), None
        return jax.lax.scan(update, (params, state), None, length=steps)[0]

    jax.block_until_ready(trainer.omegas)
    setup_seconds = perf_counter() - started
    history = []
    durations = []
    epoch_durations = []
    for iteration in range(problem['training']['epochs']):
        before = perf_counter()
        trainer.params, trainer.opt_state = epoch(trainer.params, trainer.opt_state, trainer.omegas)
        jax.block_until_ready(trainer.params)
        durations.append(perf_counter() - before)
        require_finite(trainer.params, 'optimization')
        refresh(trainer, data, representation_error=problem['uncertainty']['representation_error'])
        require_finite((trainer.errs, trainer.omegas, trainer.sigmas), 'covariance update')
        state_rmse = float(jnp.sqrt(jnp.mean(jnp.concatenate([jnp.ravel(e[:, :trainer.ndata]) for e in trainer.errs['interp']]) ** 2)))
        physics_rmse = float(jnp.sqrt(jnp.mean(jnp.concatenate([jnp.ravel(e) for e in trainer.errs['pm']['pm']]) ** 2))) / problem['time_scale']
        history.append({'epoch': iteration + 1, 'state_rmse': state_rmse, 'physics_rmse': physics_rmse})
        epoch_durations.append(perf_counter() - before)
        if physics_rmse <= problem['training']['physics_tolerance'] and (trainer.mode == 'forward' or state_rmse <= problem['training']['data_tolerance']):
            break
    last = history[-1]
    success = last['physics_rmse'] <= problem['training']['physics_tolerance'] and (trainer.mode == 'forward' or last['state_rmse'] <= problem['training']['data_tolerance'])
    inverse_order = np.argsort(problem['order'])
    predictions = [np.asarray(x)[:, inverse_order].tolist() for x in trainer.xs_sm['interp']]
    uncertainty = uncertainty_report(trainer, problem)
    checks = constraints(problem, [np.asarray(x)[:, inverse_order] for phase in ('interp', 'pm') for x in trainer.xs_sm[phase]])
    status = 'converged' if success else 'max_epochs'
    if success and not (checks['nonnegative_at_sampled_points'] and checks['surface_site_balance_satisfied']):
        status = 'physical_constraint_violation'
    return {'schema_version': 1, 'method': 'mle', 'status': status,
            'mode': problem['mode'], 'species': problem['species'],
            'rate_constants': np.asarray(jnp.exp(trainer.params['pm'][0])).tolist(),
            'log_rate_constants': np.asarray(trainer.params['pm'][0]).tolist(),
            'predictions': [{'times': d['times'], 'states': values} for d, values in zip(problem['datasets'], predictions)],
            'history': history, 'uncertainty': uncertainty, 'units': problem.get('units', {}),
            'physical_checks': checks,
            'surrogate': problem['surrogate'], 'time_scale': problem['time_scale'],
            'timing': timing(started, setup_seconds, warmup_seconds, durations, epoch_durations)}
