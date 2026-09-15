# kinn: Kinetics-Informed Neural Networks

`kinn` fits neural-network trajectories to kinetic models and measurements. Given rate constants, it solves a forward problem; given measurements, it estimates the rate constants and trajectories together. The package includes MLE covariance weighting, automatic variance propagation and SVD conservation constraints, as well as the original fixed-weight loss for Pareto studies.

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
kinn example --mode inverse --kind adsorption --output inverse.json
kinn validate inverse.json
kinn run inverse.json --output inverse-result.json
```

The example describes A + * ⇌ A*, observes the bulk species A, and supplies a known initial state. Its synthetic measurements use rates `[2, 1]`; inference starts from `[1.5, 0.7]`. Replace the measurements and mechanism with your own, or provide a CSV through `data_file` as described in the [input guide](./docs/cli.md#input-contract).

### Fit a trajectory with known rate constants

```sh
kinn example --mode forward --kind homogeneous --output forward.json
kinn validate forward.json
kinn run forward.json --output forward-result.json
```

This example fits a neural trajectory for A ⇌ B with fixed rates `[2, 1]` and initial state `[1, 0]`. Forward training matches the network's time derivative to the kinetic model while enforcing the supplied initial state.

### Choose the training objective

Both problem modes use the same `kinn` package and command. The default uses MLE covariance weighting and SVD. The `method` field selects the training objective and its associated original surrogate parameterization:

| Training option | Example flag / JSON value | Behavior |
| --- | --- | --- |
| MLE adaptive covariance (default) | `--method mle` / `"method": "mle"` | Update covariance weights during training, propagate local errors through kinetic Jacobians, and use the SVD-constrained surrogate |
| Fixed weighting | `--method fixed` / `"method": "fixed"` | Minimize physics MSE + `alpha` × data MSE; vary `training.alpha` for a Pareto study |

For example, `kinn example --method fixed --mode inverse --output weighted.json` generates an input for the original fixed-weight objective. With `fixed`, `training.alpha` is your chosen weight; with `mle`, the residual covariances determine the weighting automatically.

## Define a kinetic problem

Inputs describe the mechanism, datasets and surrogate architecture. A complete forward input for A ⇌ B is:

```json
{
  "schema_version": 1,
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

Inverse inputs supply `initial_rate_constants`, `observed_species`, and `values` or a CSV `data_file` for each dataset. Observations may cover all bulk species or all species. Multiple datasets can have different time grids and separate neural trajectories while sharing kinetic parameters. A supplied `initial_state` is enforced exactly; it is required for forward problems and optional for inverse problems.

Set hidden-layer widths with `surrogate.layers`. Activations may be one name for all layers or a list with one name per layer: `tanh`, `sigmoid`, `swish`, `softplus`, `sin` or `gaussian`. Training settings include epochs, optimizer steps per epoch, learning rate, seed and residual tolerances.

The input contract currently covers closed, isothermal, well-mixed mass-action systems. Arbitrary rate laws, multiple surface site types and partial observations of bulk species need extensions to that contract. See the [CLI and input guide](./docs/cli.md) for the full settings, units and constraints; the [examples](./examples) contain complete inputs.

## Use results from Python or another program

The Python API accepts the same JSON file or an equivalent dictionary:

```python
from kinn import solve

result = solve("inverse.json")
print(result["status"])
print(result["rate_constants"])
print(result["uncertainty"]["datasets"][0]["rate_error_covariance"])
```

| Result field | Contents |
| --- | --- |
| `status` | Whether residual tolerances and sampled physical checks passed |
| `rate_constants`, `log_rate_constants` | Final kinetic parameters in reaction-column order |
| `predictions` | Times and predicted states for each dataset, in input species order |
| `history` | Epoch, data RMSE and physics RMSE throughout training |
| `uncertainty` | Final empirical covariance estimates and their scope; available for MLE training |
| `physical_checks` | Sampled nonnegativity and surface-site balance checks |
| `timing` | Setup, first-epoch, warm execution and total solve timings |

Inverse MLE results include state-residual, log-rate, rate and time-dependent kinetic-RHS error covariance. Covariance weights are recalculated during training. The current JSON output stores the final covariance matrices and RMSE history; NN weights, reloadable checkpoints, parameter/covariance histories and the notebook's Hessian uncertainty analysis are not yet exposed by the CLI.

For programs and agents, `kinn capabilities` describes supported operations and `kinn schema` prints the JSON Schema. `kinn validate` checks the input without importing JAX or training. A run exits with code `0` on convergence with sampled physical checks, `1` on nonconvergence or numerical failure, and `2` on an input or file error. Inspect `status` before using a result.

The [original ten-species, fourteen-rate DCS example](./examples/mle-dcs-forward.json) is a more demanding benchmark. Its current default budget reaches `max_epochs`; it is useful for evaluating architectures and convergence, and is not a demonstrated converged CLI example.

## JAX execution and timing

Neural evaluations, time derivatives, kinetic Jacobians and optimizer steps use JAX. Compiled blocks of optimizer steps reuse their compiled program during a solve. Validation, convergence decisions and file handling run in Python.

The first-epoch timing includes compilation on first use. Warm epoch timings include covariance refresh and diagnostics; warm optimizer-kernel timings isolate the optimizer steps. JAX results are synchronized before timings are recorded. `wall_seconds` includes setup and uncertainty reporting but excludes Python/import startup; [the timing benchmark](./benchmarks/cli_timing.py) also measures fresh-process wall time. See the [timing guide](./docs/cli.md#jax-execution-and-timings) for details.

CI covers Linux, macOS and Windows, supported dependency combinations, numerical examples, covariance propagation and installation of the built wheel.

## Pareto + Regularization Approach

arXiv: https://doi.org/10.48550/arXiv.2011.14473

**Try it live:** an interactive, in-browser version of the KINNs inverse-kinetics solver runs at **[gabrielgusmao.com/blog/kinns-playground](https://www.gabrielgusmao.com/blog/kinns-playground/)**: recover rate constants from noisy transient data with JAX-style autodiff and MLE uncertainty, no install. See also the [Lotka-Volterra Neural ODE](https://www.gabrielgusmao.com/blog/node-lv-playground/) and [pharmacokinetics Neural ODE](https://www.gabrielgusmao.com/blog/pkpd-playground/) playgrounds.

***KINNs training example*** for the *dcs* reaction type

<p align="center">
  <img src="./misc/gifs/kinn4.gif" alt="KINNs training: state interpolation, derivative matching, and recovery of kinetic parameters for the dcs reaction system" width="800"/>
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

1. [Data generation and KINNs training](./paper/kinn_datagen_reg.ipynb).
2. [Data processing and plot generation](./paper/kinn_plotsgen_reg.ipynb).

The corresponding Python sources are [kinn_datagen_reg.py](./paper/kinn_datagen_reg.py) and [kinn_plotsgen_reg.py](./paper/kinn_plotsgen_reg.py). The original kinetic model, neural network, and training loop live in [kinn/basis](./kinn/basis); the paper-specific constraints, loss, and benchmark systems are in [trainer_source.py](./paper/trainer_source.py).


## MLE, SVD and automatic variance propagation

The [rKINNs paper](https://arxiv.org/abs/2304.05991) describes the MLE and SVD extension implemented in [basis/mle.py](./kinn/basis/mle.py). For a stoichiometric matrix $`M`$, the kinetic model is $`\dot{x}=M r(x,p)`$, where $`p=\ln(k)`$. An SVD separates the reactive coordinates from conserved coordinates:

```math
M = U_r D_r V_r^T, \qquad U_n^T M = 0,
\qquad \frac{d}{dt}(U_n^T x)=0.
```

The SVD-constrained surrogate represents this conservation structure. MLE training estimates covariance from the current residuals and uses inverse covariance matrices to weight agreement with the observations and the kinetic model. This replaces the fixed scalar data/physics weight used in the original Pareto studies.

For independent state and log-rate errors, the local RHS covariance is propagated through $`J_x=\partial f/\partial x`$ and $`J_p=\partial f/\partial p`$:

```math
\Sigma_f(t) = J_x(t)\Sigma_x J_x(t)^T + J_p(t)\Sigma_p J_p(t)^T.
```

The CLI reports empirical first-order error covariances, with explicit rank and numerical-regularization diagnostics. These are not posterior credible intervals or calibrated parameter confidence intervals. Forward runs keep rates fixed; original KINNs retains its weighted objective without covariance estimation. See the [variance and result guide](./docs/cli.md#results-and-variance-propagation) for the assumptions and scope.

The original rKINNs [notebook](./paper/rkinn.ipynb) and [Python source](./paper/rkinn.py) are retained alongside the KINNs reference material.
