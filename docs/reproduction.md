# Reproducing the KINN papers

All commands run locally through the single `kinn` package. Install the branch as shown in the README, then `python -m pip install ".[reproduce]"` for optional Matplotlib plots. Numerical commands need only the base installation.

## Coverage and acceptance criteria

The reference papers are [Kinetics-Informed Neural Networks, v2](https://arxiv.org/html/2011.14473v2) and [Maximum-likelihood Estimators in Physics-Informed Neural Networks for High-dimensional Inverse Problems, v2](https://arxiv.org/abs/2304.05991).

An experiment is fully reproduced only when its inputs, initialization, training schedule, saved-state selection and exported results can be regenerated and checked. Re-evaluating an archived network verifies the current model and surrogate implementation; it does not establish that fresh training reaches the same optimum.

| Result | CLI coverage | Remaining verification |
| --- | --- | --- |
| Original paper, four mechanisms and two initial conditions | All 21 archives evaluate in current JAX | Fresh training equivalence across the full matrix |
| Tables 3 and 4 | Every numerical entry matches within rounding | None for archived evaluation |
| Table 5 | All but the noisy homogeneous MAE match | Resolve the printed MAE versus saved parameters |
| Numerical content behind trajectory, derivative, parameter and Pareto figures | CSV/NPZ plus SVG/PDF redraws | Original layout and selections beyond the documented table/trajectory choices |
| Original conceptual Figures 1 and 2 | Existing figures retained in `paper/` | No numerical experiment to rerun |
| MLE paper Figures 1–3, covariance relaxation and Hessian uncertainty | Source notebook retained; general MLE CLI available | Faithful notebook settings, history exports and reference comparisons |

`kinn reproduce list` returns this coverage in machine-readable form, including `full_paper_reproduction: false`. A successful archive export or ordinary demo does not certify reproduction of both papers.

## Evaluate saved experiments

```sh
kinn reproduce list
kinn reproduce run trainer_fwd_3_alpha13 --output forward-reference --plots
kinn reproduce run trainer_invvwn_3_alpha13 --output inverse-reference --plots
kinn reproduce run trainer_invsc_3_alpha17sens --output pareto-reference --plots
kinn reproduce run all --output all-references --plots
```

Choose a new or empty output directory. IDs preserve historical filenames. Indices `0`, `1`, `2`, `3` denote `g`, `da`, `dc`, `dcs`. Scenarios are `fwd` (forward), `inv` (Q), `invsc` (SQ), and `invvwn` (SQ+n). `alpha17sens` is the saved 61-stage tightening/relaxation sweep. These are experiment identifiers, not packages or method names.

Each experiment exports:

- `result.json`: selected checkpoint, rate constants, calibration scales, stage history and metrics.
- `arrays.npz` and `trajectory-1.csv`, `trajectory-2.csv`: times, ground-truth integration, surrogate states, automatic time derivatives and physical-model derivatives. NPZ also includes raw and calibrated observations; `observations-1.csv` and `observations-2.csv` contain reconstructed raw inverse measurements.
- `parameters.json` + `parameters.npz`: selected neural weights, log rates and log calibration factors in the original tree structure.
- `checkpoints.json` + `checkpoints.npz`: every original saved stage, including parameters, alpha, errors and optimizer settings.
- With `--plots`, trajectory/parity and parameter/Pareto plots in SVG and PDF. Lines show ground truth, open circles the surrogate, and faint dots reconstructed inverse measurements. These redraws are not copies of the paper's typesetting.

The output manifest records experiment selection, dependency versions, JAX precision/devices and total elapsed time; each result includes the original archive SHA256. `run all` also writes the published-table `verification.json`, including discrepancies, alongside the results. Archived histories contain stage checkpoints, not every optimizer step or epoch.

Retrieve arrays and weights from Python without pickle:

```python
import numpy as np
from kinn.reproduction import load_export

arrays = np.load("inverse-reference/arrays.npz", allow_pickle=False)
states = arrays["dataset_0_states"]
parameters = load_export("inverse-reference")
history = load_export("inverse-reference", "checkpoints")["iter_data"]
```

Bundled archives are numeric-only NPZ plus a JSON tree. Original research archives remain untouched. Developers can regenerate the conversion from a Git checkout with `python tools/prepare_paper_assets.py`; its restricted reader is only for the trusted repository files. The runtime CLI never deserializes historical pickle objects.

## Verify the numerical tables

```sh
kinn reproduce verify --output verification.json
```

This checks 184 values against the printed precision in Tables 3–5. It exits 0 only if every value matches; differences produce a complete report and exit 1. Currently 183 match. The Table 5 homogeneous noisy MAE is `0.022917068551180413` from the final saved log rates, versus the published `2.29e-4`. The roughly hundredfold difference is unresolved; the verifier does not substitute a corrected value or widen its tolerance.

The paper's custom `r2_score` computes mean squared Pearson correlation over species. The export labels that value `r2` and also supplies the usual `coefficient_of_determination`; these are different statistics. Derivative parity compares the surrogate derivative with the physical RHS evaluated at surrogate states and fitted rates, rather than with ground-truth ODE derivatives.

Default `--state paper` follows the plotting script's noisy-case checkpoint heuristic; noiseless and forward evaluations use the final model. DCS SQ+n selects state index 3, epoch 4000. Use `--state final` or a nonnegative zero-based index to inspect another state. Results always record the selected state and epoch.

Table 5 uses different states from the noisy trajectory table. Its noiseless parameter values match the penultimate saved states and its heterogeneous noisy values match the final states. This correspondence is inferred from the numerical values; the surviving plotting script contains a one-state lag in its parameter-history construction. Each verification row records its state. No single checkpoint is represented as reproducing all figures and tables.

## Train afresh

```sh
kinn reproduce plan trainer_fwd_0_alpha13 --output recipe.json
kinn reproduce train trainer_fwd_0_alpha13 --output fresh-forward --plots
kinn reproduce train trainer_invsc_3_alpha17sens --output fresh-pareto --plots
```

The recipe describes stoichiometry, species, true rates for data generation, initial conditions, actual archived network widths, activations, noise, initialization and every recorded alpha/Adam stage. Forward training uses the original auxiliary time-scale network and exact initial-condition transformation. Inverse SQ cases jointly train log calibration factors, network weights and log rate constants. Optimizer moments reset between stages while learned parameters continue.

Archived architectures and stage schedules take precedence over conflicting settings in the surviving script. For example, archived `da` networks have three hidden layers of width 12, while `trainer_source.py` declares 10. Some archives include extra continuation/refinement stages. The earlier `alpha13sens` archive has no unambiguous iteration-count recipe, so its `train` command is unavailable.

Full runs can take hours. For an execution check:

```sh
kinn reproduce train trainer_invsc_1_alpha13 --epochs-per-stage 2 --steps-per-epoch 2 --max-stages 2 --output training-smoke
```

Budget overrides are recorded as `budget_override`. Full runs report `schedule_completed`; neither asserts convergence or agreement with the paper. Stage-boundary checkpoints contain neural/kinetic/calibration parameters and history, but do not yet implement automatic resume or within-stage optimizer-state recovery.

Epochs use a compiled JAX scan. Timings separate explicit epoch compilation from synchronized warm training and include total runtime for setup, diagnostics and evaluation. The CLI additionally reports elapsed time including checkpoint writes and final exports. Results and performance may vary with JAX/XLA, hardware and random-number implementation; the environment is recorded rather than promising bitwise identical retraining.

Reference ODE integration uses LSODA with `rtol=1e-11`, `atol=1e-13`. The old script requests `1e-20`, below double precision, over a longer interval while sampling only `[0,1]`. The current integration uses that sampled interval and achievable tolerances. Archived table comparisons verify that this change stays within published rounding. SQ normalization uses standard deviation across both datasets; SQ+n recreates the original same-key-per-dataset convention. Exact historical random streams across JAX releases are not yet verified.

## MLE references still needed

`paper/rkinn.ipynb` refers to `opt_invvwn_4_alpha107sens_uq_full.npz` and `trainer_invvwn_3_mle.npz`; neither is present in the available checkouts. The notebook differs from the fixed-weight paper: reversed initial-condition order, max-normalized surface signals, a different noise key for each dataset, SVD calibration, centered covariance updates and additional likelihood/Hessian calculations.

The general MLE solver does not yet certify this notebook workflow. Those settings, covariance relaxation, common-likelihood comparison of Pareto and MLE histories, kinetic Hessian uncertainty and architecture variants must be ported and tested against reference outputs before the full-reproduction flag can become true. A missing archive is reported as missing, never replaced by a demo or synthesized reference.
