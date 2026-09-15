# The local `kinn` CLI

Install the repository with Python 3.11 or newer (`python -m pip install .`). This installs JAX, NumPy, SciPy and Optax, and the `kinn` command. Training and all input/output files stay on the machine running the command. There is no server, telemetry, or account requirement.

## Choose the training formulation and problem

`kinn` is the package, Python import and CLI command for every configuration below. The `rkinns` research branch introduced the MLE variance propagation and SVD extension to this package. The `method` field selects its training formulation: `rkinn` enables that extension (the default), while `kinn` selects the original fixed-weight objective for Pareto studies. The scientific names KINNs and rKINNs refer to these formulations.

| `method` | `mode` | Estimated quantities | Training objective |
| --- | --- | --- | --- |
| `kinn` | `forward` | Neural trajectory | Original physics mean-square residual |
| `kinn` | `inverse` | Neural trajectory and positive rate constants | Original physics MSE + `alpha` × data MSE |
| `rkinn` | `forward` | Neural trajectory | Physics residual with estimated covariance weights |
| `rkinn` | `inverse` | Neural trajectory and positive rate constants | Data and physics residuals with automatic covariance weighting |

KINNs uses the `nn_combo` parameterization and `TrainerCV` weighted objective from the original paper code. rKINNs uses `nn_npt`, the original SVD decomposition, and the covariance-update routines in `basis/mle.py`. Both use the original mass-action model and JAX differentiation. They fit trajectories with neural networks; the CLI does not delegate the forward problem to an ODE solver.

```sh
kinn example --method rkinn --mode inverse --kind adsorption --output problem.json
kinn validate problem.json
kinn run problem.json --output result.json
```

Use `--method kinn` for the original formulation, `--mode forward` for known rates, or `--kind homogeneous` for the A ⇌ B example. `python -m kinn` is equivalent to `kinn`. The generated examples are analytic reference problems with true rates `[2, 1]`; inverse inputs start at `[1.5, 0.7]`.

`kinn capabilities` describes the supported model, methods and commands. `kinn schema` prints the bundled JSON Schema. `kinn validate` additionally checks matrix dimensions, site conservation, observations and time ordering without loading JAX or training a model.

The additional [`rkinn-dcs-forward.json`](../examples/rkinn-dcs-forward.json) preserves the ten-species, fourteen-rate mechanism and rate constants from the original paper. This stiff benchmark is more demanding than the analytic examples: the default budget reached `max_epochs` in local testing. Inspect residuals and physical checks when adjusting its architecture, collocation grid or training budget.

## Input contract

A minimal forward problem for A ⇌ B is:

```json
{
  "schema_version": 1,
  "method": "kinn",
  "mode": "forward",
  "species": ["A", "B"],
  "stoichiometry": [[-1, 1], [1, -1]],
  "rate_constants": [2, 1],
  "datasets": [{
    "times": [0, 0.1, 0.2, 0.4, 0.7, 1],
    "initial_state": [1, 0]
  }],
  "surrogate": {"layers": [16, 16], "activations": "tanh"},
  "units": {"time": "s", "bulk": "normalized concentration"}
}
```

The stoichiometric matrix has one **species row** and one **directed elementary reaction column**. Negative entries are reactants; positive entries are products. A reversible reaction has two separate columns and two rate constants. Reaction orders are inferred from negative stoichiometric entries. This contract cannot represent an unchanged catalyst or a species on both sides of an elementary reaction whose order cannot be recovered from the net column. Such mechanisms need an explicit rate-law extension rather than a misleading net matrix.

The current CLI describes closed, isothermal, well-mixed mass-action systems. It supports homogeneous reactions and one surface site balance with equal site occupancy. Include adsorbates and the vacant site in `species`, and list their names in `surface_species`. Their matrix rows must sum to zero for every reaction, and their initial fractions must sum to one. At least one nonsurface species is required. Multiple site types, temperature-dependent rates, flow terms, PDEs, arbitrary custom functions and arbitrary missing bulk channels are outside this CLI contract.

For A + * ⇌ A*, use:

```json
{
  "species": ["A", "A*", "*"],
  "surface_species": ["A*", "*"],
  "stoichiometry": [[-1, 1], [1, -1], [-1, 1]],
  "observed_species": ["A"]
}
```

This fragment belongs inside a complete problem such as `kinn example --kind adsorption`. The bulk state and surface coverage must use compatible normalization for the supplied stoichiometry; the CLI does not infer concentration-to-site-density factors. `units` records labels without performing conversions.

In forward mode, supply `rate_constants` and a full `initial_state` for every dataset. These rates remain fixed throughout optimization. In inverse mode, supply `initial_rate_constants`, `observed_species` and measured `values`; the rates are optimized in log space. `values` rows follow `times`, and columns follow `observed_species`. Measurements must be calibrated. Observations can cover all nonsurface species, or all species. Species and observation order are normalized internally, then predictions are returned in the input `species` order.

An inverse `initial_state` is optional. When supplied, it is an exact initial condition for the surrogate, including latent species. Without it, original KINNs learns an unconstrained initial state; homogeneous rKINNs fixes conserved quantities from the mean observed state, and surface rKINNs uses its original latent conserved-coordinate parameterization. No unobserved initial state is filled with an assumed zero.

Each dataset needs at least three finite, strictly increasing times. `collocation_times` optionally chooses the physics-residual grid within that dataset's time interval; it defaults to `times`. Datasets can have different numbers of samples. They have separate neural trajectories and share one set of kinetic parameters. Each dataset contributes equally to the training objective; reported RMSE pools samples.

Instead of inline `times` and `values`, inverse datasets can use `"data_file": "measurements.csv"`. The path is relative to the problem JSON. The CSV must have a `time` column and exactly the observed-species columns (in any order). The numeric data and JSON are parsed, never evaluated as Python.

## Architecture and training

`surrogate.layers` lists hidden-layer widths. A single activation name is broadcast to all hidden layers, or a list specifies each layer individually. Available activations are `tanh`, `sigmoid`, `swish`, `softplus`, `sin` and `gaussian`. The time input has width one; output dimensions follow the selected method, stoichiometric rank and surface constraint. The default is two hidden layers of width 16, tanh activations, and `init_scale: 0.3`.

Known initial conditions use the original tanh boundary gate. `surrogate.boundary_gain` defaults to `"auto"`, which scales the gate using the initial rate law for surface systems. A positive numeric gain overrides it. For inverse problems the automatic choice uses the initial rate guesses; large changes in inferred time scales may require a larger gain.

| Training field | Default | Meaning |
| --- | --- | --- |
| `epochs` | 300 | Maximum covariance/training cycles |
| `steps_per_epoch` | 100 | Compiled optimizer steps per cycle |
| `warmup_steps` | 1000 | Data-fit initialization for inverse problems; zero disables it |
| `learning_rate` | 0.001 | Adam initial learning rate |
| `learning_rate_schedule` | `cosine` | Decay to 1% of the initial rate; `constant` is also supported |
| `seed` | 0 | Reproducible initialization; each dataset gets a separate seed |
| `physics_tolerance` | 0.01 | Absolute RMS state-derivative residual in input time units |
| `data_tolerance` | 0.01 | Absolute RMS residual over measured species |
| `alpha` | 1 | Data MSE weight for original KINNs only |

The supplied KINNs inverse examples explicitly use `alpha: 100` and `data_tolerance: 0.0005`. Their weighting is illustrative; data scales, noise and identifiability determine appropriate settings. For a Pareto study, repeat the same KINNs input with several positive `alpha` values. rKINNs replaces this manual weight with covariance updates. Both methods still require architecture and optimizer choices.

Time is shifted by each dataset's first observation and divided by a common maximum duration. The physical RHS is multiplied by that duration; the stoichiometric matrix and mass-action exponents are unchanged. Reported times, derivatives, rate constants and propagated RHS covariance use the input time scale.

## Results and variance propagation

Successful runs return JSON with the selected method and mode, rates, log rates, predicted states, residual history, physical checks, uncertainty details and timings. Exit code `0` means the requested residual tolerances were met and sampled physical checks passed. Code `1` means `max_epochs`, `physical_constraint_violation` or `numerical_failure`; inspect the output before using it. Code `2` means an input or file error. Convergence is a stopping criterion, not a guarantee of unique parameter recovery.

`physical_checks` reports the minimum sampled state and surface-balance error at observation and collocation points. No positivity or approximation guarantee is made between those points. KINNs retains its original unconstrained bulk outputs; rKINNs additionally enforces its SVD conservation structure.

Inverse rKINNs estimates state residual second moments and local log-rate error covariances, applies oracle approximating shrinkage (OAS) when constructing covariance weights, and propagates error through model Jacobians. The result includes state, log-rate, rate and time-dependent RHS error covariance. Rate covariance is the first-order log-to-rate transformation. The propagated covariance assumes independent state and log-rate errors; cross covariance is not included.

These are **empirical local error estimates, not posterior credible intervals or calibrated confidence intervals for the fitted parameter estimator**. The CLI does not currently expose Hessian/profile-likelihood confidence intervals or bootstrap coverage estimates. A rank-deficient kinetic sensitivity matrix is reported explicitly and rate-error covariance is `null` rather than presenting unidentified directions as zero uncertainty. Full-state sensitivity rank does not establish identifiability from partial measurements.

`uncertainty.representation_error: true` also includes the neural derivative's state linearization when forming residual weights. It is available only for rKINNs and increases derivative/compilation work. Numerical inversion uses an eigenvalue floor of `max(1e-12, 1e-8 * largest absolute eigenvalue)` separately from OAS shrinkage. Forward rKINNs reports the surrogate's physics-defect covariance; fixed input rates have no estimated uncertainty. Original KINNs does not estimate covariance.

## JAX execution and timings

Neural evaluations, time derivatives, kinetic Jacobians and optimizer kernels use JAX. A compiled `lax.scan` executes each block of optimizer steps. Time derivatives use forward-mode differentiation; kinetic state Jacobians are computed pointwise and batched, avoiding differentiation of an entire batch into a quadratic-size Jacobian. Validation, object construction, stopping decisions and file output remain in Python.

`setup_seconds` includes construction, inverse initialization and initial compilation. The first epoch and its optimizer-kernel timing include compilation on first use; subsequent `warm_epoch_seconds` include covariance refresh and diagnostic work, while `warm_optimization_kernel_seconds` isolates optimization. Timings synchronize JAX results. `wall_seconds` includes setup and final uncertainty reporting but excludes interpreter/import startup. `python benchmarks/cli_timing.py` measures fresh-process wall time as well. Warm timings describe reuse inside one solve, not zero-compilation startup for a new CLI invocation.

CI checks CPU execution on Linux, macOS and Windows, a minimum supported dependency set, numerical examples, covariance propagation, and an installed wheel outside the checkout. Timing artifacts are informational; hardware-dependent times are not pass/fail thresholds.

## Python and original research material

```python
from kinn import solve

result = solve("problem.json")  # or supply the same problem as a dictionary
print(result["status"], result["rate_constants"])
```

The original kinetic, neural-network and training classes remain under `kinn/basis`. The CLI adapters live in `kinn/pareto.py` and `kinn/rkinn.py`. `paper/` retains the research notebooks and source scripts, including their original experiment definitions. Those historical experiment launchers are reference material, not the CLI's tested entry points. Install `.[notebooks]` for plotting and notebook dependencies; explicit historical plotting imports from `kinn.basis` remain available lazily.

The numerical adapters correct forward residual wiring, apply the computed OAS covariance, preserve site-conserving latent directions, and project covariance matrices into the same SVD coordinates as the residuals. The original research files and training animation are retained.

References: [KINNs](https://arxiv.org/abs/2011.14473) and [rKINNs / maximum-likelihood estimators](https://arxiv.org/abs/2304.05991).
