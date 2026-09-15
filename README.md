# kinn: Kinetics-Informed Neural Networks

`kinn` fits neural-network trajectories to kinetic models and measurements using JAX. It supports forward problems with known rate constants and inverse problems that estimate rates from measurements. Both modes support fixed residual weighting or MLE adaptive covariance weighting. The MLE formulation includes SVD conservation constraints and, for inverse problems, automatic variance propagation.

## Run locally

Python 3.11 or newer is required. Install the CLI branch:

```sh
git clone --branch kinn-cli https://github.com/gusmaogabriels/kinn.git
cd kinn
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. Installation provides the `kinn` Python package and the `kinn` command. `python -m kinn` is an equivalent entry point. Training runs locally on the machine where you invoke it.

### Estimate rate constants from measurements

Generate an adsorption example, check its inputs, and solve the inverse problem:

```sh
kinn example --method mle --mode inverse --kind adsorption --output inverse.json
kinn validate inverse.json
kinn run inverse.json --output inverse-result.json
```

The example describes A + * ⇌ A*, observes the bulk species A, and supplies a known initial state. Its synthetic measurements use rates `[2, 1]`; inference starts from `[1.5, 0.7]`. Replace the measurements and mechanism with your own, or provide a CSV through `data_file` as described in the [input guide](./docs/cli.md#input-contract).

### Fit a trajectory with known rate constants

```sh
kinn example --method mle --mode forward --kind homogeneous --output forward.json
kinn validate forward.json
kinn run forward.json --output forward-result.json
```

This example fits a neural trajectory for A ⇌ B with known rates `[2, 1]` and initial state `[1, 0]`. Forward training matches the network's time derivative to the kinetic model while enforcing the supplied initial state. Rates remain unchanged under either training method; forward inputs contain no measured `values`.

### Choose the training objective

`mode` selects what is fitted; `method` selects the residual weighting and its associated original surrogate parameterization. These are independent choices in one package, `kinn`. When `method` is omitted, it defaults to `mle`.

| `mode` | `method` | Fitted quantities | Objective |
| --- | --- | --- | --- |
| `forward` | `fixed` | Neural trajectory; supplied rates stay fixed | Physics MSE |
| `forward` | `mle` | Neural trajectory; supplied rates stay fixed | Physics residual with adaptive covariance weighting |
| `inverse` | `fixed` | Neural trajectory and rate constants | Physics MSE + `alpha` × data MSE |
| `inverse` | `mle` | Neural trajectory and rate constants | Data and physics residuals with adaptive covariance weighting and local variance propagation |

To generate inputs for the original fixed-weight formulation:

```sh
kinn example --method fixed --mode forward --kind homogeneous --output fixed-forward.json
kinn example --method fixed --mode inverse --kind adsorption --output fixed-inverse.json
kinn validate fixed-inverse.json
kinn run fixed-inverse.json --output fixed-inverse-result.json
```

For inverse `fixed` problems, `training.alpha` is a positive scalar held constant throughout one `kinn run`. It defaults to `1`; the generated fixed inverse examples explicitly choose `100`. Repeat the same problem with different alpha values to study the data/physics tradeoff. Forward `fixed` problems have no data term, so alpha has no effect.

MLE updates covariance matrices during training, rather than tuning a scalar alpha. Omit `training.alpha` for `mle`; validation rejects it. Architecture, learning rate and training budget remain configurable for both methods.

`--method` and `--mode` are flags for `kinn example`. To solve your own input, set `method` and `mode` inside the JSON; `kinn run` reads them from that file.

## Reproduce the paper experiments

The CLI includes the original four mechanisms, both initial conditions, and 21 saved fixed-weight training archives. Re-evaluate a reference experiment with its trajectories, derivatives, parameters, calibration scales and Pareto history:

```sh
python -m pip install ".[reproduce]"
kinn reproduce list
kinn reproduce run trainer_invvwn_3_alpha13 --output paper-example --plots
```

Fresh fixed-weight training is also available with the saved architectures and stage schedules:

```sh
kinn reproduce plan trainer_fwd_0_alpha13 --output training-plan.json
kinn reproduce train trainer_fwd_0_alpha13 --output fresh-forward --plots
```

Evaluating a saved archive and training afresh are separate operations. These reference commands currently cover the fixed-weight paper experiments; full reproduction of both papers remains incomplete. The [reproduction guide](./docs/reproduction.md) describes the available experiments, checkpoint selection and exported files. General forward and inverse problems with MLE weighting use `kinn run` as shown above.

## Define a kinetic problem

Inputs describe the mechanism, datasets and surrogate architecture. A complete forward input for A ⇌ B with MLE covariance weighting is:

```json
{
  "schema_version": 1,
  "method": "mle",
  "mode": "forward",
  "species": ["A", "B"],
  "stoichiometry": [[-1, 1], [1, -1]],
  "rate_constants": [2, 1],
  "datasets": [{
    "times": [0, 0.1, 0.2, 0.4, 0.7, 1],
    "initial_state": [1, 0]
  }],
  "surrogate": {
    "layers": [16, 16],
    "activations": "tanh"
  },
  "units": {"time": "s", "bulk": "normalized concentration"}
}
```

The stoichiometric matrix has **species rows and directed reaction columns**. Negative entries are reactants and positive entries are products. Each reversible reaction needs two columns and two rate constants, in matching order. Reaction orders follow the negative entries of the matrix.

For surface mechanisms, include adsorbates and vacant sites in `species` and identify them in `surface_species`. The current CLI supports one conserved site balance with equal site occupancy: the surface rows must sum to zero in every reaction column, and initial surface fractions must sum to one. Bulk concentrations and surface fractions must use compatible normalization for the supplied matrix.

Forward inputs supply `rate_constants` and a full `initial_state` for every dataset. Inverse inputs instead supply `initial_rate_constants`, `observed_species`, and `values` or a CSV `data_file` for each dataset. Observations may cover all bulk species or all species. Multiple datasets can have different time grids and separate neural trajectories while sharing kinetic parameters. A supplied `initial_state` is enforced exactly; it is required for forward problems and optional for inverse problems.

Set hidden-layer widths with `surrogate.layers`. Activations may be one name for all layers or a list with one name per layer: `tanh`, `sigmoid`, `swish`, `softplus`, `sin` or `gaussian`. Training settings include epochs, optimizer steps per epoch, learning rate, seed and residual tolerances. `training.warmup_steps` initializes the data fit in inverse mode. Forward convergence uses `physics_tolerance`; inverse convergence also requires `data_tolerance`.

The input contract currently covers closed, isothermal, well-mixed mass-action systems. Arbitrary rate laws, multiple surface site types and partial observations of bulk species need extensions to that contract. See the [CLI and input guide](./docs/cli.md) for the full settings, units and constraints; the [examples](./examples) contain complete inputs.

## Use results from Python or another program

The Python API accepts the same JSON file or an equivalent dictionary:

```python
from kinn import solve

result = solve("inverse.json")
print(result["status"])
print(result["rate_constants"])
print(result["uncertainty"])
```

| Result field | Contents |
| --- | --- |
| `mode`, `method` | The problem mode and training method used |
| `status` | Whether residual tolerances and sampled physical checks passed |
| `rate_constants`, `log_rate_constants` | Supplied rates in forward mode, fitted rates in inverse mode; reaction-column order |
| `predictions` | Times and predicted states for each dataset, in input species order |
| `history` | Epoch, data RMSE and physics RMSE throughout training |
| `uncertainty` | Final empirical covariance estimates for MLE; `method: "not_estimated"` for fixed weighting |
| `physical_checks` | Sampled nonnegativity and surface-site balance checks |
| `timing` | Setup, first-epoch, warm execution and total solve timings |

Inverse MLE results include state-residual, log-rate, rate and time-dependent kinetic-RHS error covariance. Rank-deficient kinetic sensitivities produce `null` rate-error covariances with a diagnostic. Forward MLE results report `physics_defect_covariance`; they do not estimate uncertainty in the supplied rates. Fixed weighting does not estimate covariance in either mode.

For general problems, `kinn run` stores final results and RMSE history, but does not yet export neural weights, reloadable checkpoints or parameter/covariance histories. The separate `kinn reproduce` commands export the fixed-weight reference experiment parameters and saved stages; see the [reproduction guide](./docs/reproduction.md). These paper-specific exports have a different scope from a general solve.

For programs and agents, `kinn capabilities` describes supported operations and `kinn schema` prints the JSON Schema. `kinn validate` checks the input without importing JAX or training. A run exits with code `0` on convergence with sampled physical checks, `1` on nonconvergence or numerical failure, and `2` on an input or file error. Inspect `status` before using a result.

The [original ten-species, fourteen-rate DCS example](./examples/mle-dcs-forward.json) is a more demanding benchmark. Its current default budget reaches `max_epochs`; it is useful for evaluating architectures and convergence, and is not a demonstrated converged CLI example.

## JAX execution and timing

Neural evaluations, time derivatives, kinetic Jacobians and optimizer steps use JAX. Compiled blocks of optimizer steps reuse their compiled program during a solve. Validation, convergence decisions and file handling run in Python.

The first-epoch timing includes compilation on first use. Warm epoch timings include covariance refresh and diagnostics; warm optimizer-kernel timings isolate the optimizer steps. JAX results are synchronized before timings are recorded. `wall_seconds` includes setup and uncertainty reporting but excludes Python/import startup; [the timing benchmark](./benchmarks/cli_timing.py) also measures fresh-process wall time. See the [timing guide](./docs/cli.md#jax-execution-and-timings) for details.

CI covers Linux, macOS and Windows, supported dependency combinations, numerical examples, covariance propagation and installation of the built wheel.

## Adaptive covariance weighting during training

During MLE training, KINNs estimates covariance from sampled residuals and propagates state and parameter uncertainty through the kinetic model. The resulting inverse covariance matrices weight the data and physics residuals. They are refreshed between training cycles and held fixed within each block of parameter updates, as described in the [MLE paper](https://arxiv.org/abs/2304.05991).

The comparison below shows this adaptive trajectory alongside the regularization sweep from the [original fixed-alpha formulation](https://doi.org/10.48550/arXiv.2011.14473), evaluated in the same likelihood coordinates.

<p align="center">
  <img src="./misc/gifs/pareto-sweep.gif" alt="Overlaid fixed-alpha and MLE convergence paths in shared likelihood coordinates, with a close-up of the MLE stable point" width="800"/>
</p>

*Animated redraw of [Figure 2c of the MLE paper](https://arxiv.org/html/2304.05991v2#S2.F2).* Fixed-alpha and MLE paths share the paper's likelihood coordinates, with the MLE stable point enlarged in the inset. Frames reveal published points; playback does not represent training time.

## MLE training example

The [MLE paper](https://arxiv.org/abs/2304.05991) extends KINNs with adaptive covariance weighting, SVD coordinates and automatic variance propagation. The animation below illustrates inverse fitting for the DCS mechanism.

**Try it live:** an interactive, in-browser version of the KINNs inverse-kinetics solver runs at **[gabrielgusmao.com/blog/kinns-playground](https://www.gabrielgusmao.com/blog/kinns-playground/)**: recover rate constants from noisy transient data with JAX-style autodiff and MLE uncertainty, no install. See also the [Lotka-Volterra Neural ODE](https://www.gabrielgusmao.com/blog/node-lv-playground/) and [pharmacokinetics Neural ODE](https://www.gabrielgusmao.com/blog/pkpd-playground/) playgrounds.

***KINNs with MLE adaptive covariance*** for the *dcs* reaction type

<p align="center">
  <img src="./misc/gifs/kinn4.gif" alt="KINNs MLE training: state interpolation, derivative matching, and recovery of kinetic parameters for the dcs reaction system" width="800"/>
</p>

$`\dot{\mathbf{x}}(t)`$ is obtained by automatic differentiation of the neural-network trajectory $`\mathbf{x}(t)`$ with respect to time $`t`$. The physical model (microkinetic model) is denoted by $`f_{\mathbf{p}}(\cdot)`$, with parameters $`\mathbf{p}=\ln(\mathbf{k})`$.

The kinetic model represents the following fully reversible chemical reactions. Type *d* involves adsorption/desorption steps. Type *c* introduces surface intermediates, such as $`D^*`$, without a corresponding gas-phase species. Reactions between radicals $`D^*`$, $`E^*`$, and $`F^*`$ add further complexity, type *s*.

Here $`*`$ denotes a vacant surface site, and $`k_i`$ and $`k_{-i}`$ denote the forward and reverse rate constants for each reaction pair. The notebook stores the fourteen constants in interleaved forward/reverse order.

```math
\begin{aligned}
A + * &\underset{k_{-1}}{\overset{k_1}{\rightleftharpoons}} A^* \\
B + * &\underset{k_{-2}}{\overset{k_2}{\rightleftharpoons}} B^* \\
C + * &\underset{k_{-3}}{\overset{k_3}{\rightleftharpoons}} C^* \\
A^* + * &\underset{k_{-4}}{\overset{k_4}{\rightleftharpoons}} 2D^* \\
B^* + * &\underset{k_{-5}}{\overset{k_5}{\rightleftharpoons}} 2E^* \\
D^* + E^* &\underset{k_{-6}}{\overset{k_6}{\rightleftharpoons}} F^* + * \\
F^* + E^* &\underset{k_{-7}}{\overset{k_7}{\rightleftharpoons}} C^* + *
\end{aligned}
```

The [JAX](https://github.com/jax-ml/jax)-based kinetic models, neural networks and trainers are under [kinn](./kinn).

## Reference Jupyter Notebooks

The original fixed-alpha Pareto studies are retained in:

1. [Data generation and KINNs training](./paper/kinn_datagen_reg.ipynb).
2. [Data processing and plot generation](./paper/kinn_plotsgen_reg.ipynb).

The corresponding Python sources are [kinn_datagen_reg.py](./paper/kinn_datagen_reg.py) and [kinn_plotsgen_reg.py](./paper/kinn_plotsgen_reg.py). The original kinetic model, neural network, and training loop live in [kinn/basis](./kinn/basis); the paper-specific constraints, loss, and benchmark systems are in [trainer_source.py](./paper/trainer_source.py).


## MLE, SVD and automatic variance propagation

The [rKINNs paper](https://arxiv.org/abs/2304.05991) describes the MLE and SVD extension implemented in [basis/mle.py](./kinn/basis/mle.py). For a stoichiometric matrix $`M`$, the kinetic model is $`\dot{x}=M r(x,p)`$, where $`p=\ln(k)`$. An SVD separates the reactive coordinates from conserved coordinates:

```math
M = U_r D_r V_r^T, \qquad U_n^T M = 0,
\qquad \frac{d}{dt}(U_n^T x)=0.
```

The SVD-constrained surrogate represents this conservation structure. In inverse mode, MLE training estimates covariance from the current residuals and uses inverse covariance matrices to weight agreement with the observations and the kinetic model. This replaces the fixed scalar data/physics weight used in the original Pareto studies. In forward mode, covariance weighting applies to the physics residual alone.

For inverse MLE problems with independent state and log-rate errors, the local RHS covariance is propagated through $`J_x=\partial f/\partial x`$ and $`J_p=\partial f/\partial p`$:

```math
\Sigma_f(t) = J_x(t)\Sigma_x J_x(t)^T + J_p(t)\Sigma_p J_p(t)^T.
```

General `kinn run` MLE results report empirical error covariances, with explicit numerical-regularization diagnostics and, for inverse problems, sensitivity-rank diagnostics. These are not posterior credible intervals or calibrated parameter confidence intervals. Both forward methods keep rates fixed; `method: "fixed"` retains the original objective without covariance estimation. See the [variance and result guide](./docs/cli.md#results-and-variance-propagation) for the assumptions and scope.

The original rKINNs [notebook](./paper/rkinn.ipynb) and [Python source](./paper/rkinn.py) are retained alongside the KINNs reference material.
