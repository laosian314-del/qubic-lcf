# QUBIC-LCF

**QUBO-based combinatorial linear-combination fitting for XAS/XANES spectra.**

QUBIC-LCF solves the linear combination fitting (LCF) problem for X-ray
absorption spectroscopy (XAS/XANES): given a target spectrum `y` and a
reference library `A` (one reference spectrum per column), find *which*
references participate in the mixture and *what* their concentrations are.
Instead of relaxing the discrete selection, QUBIC-LCF maps subset selection
onto a QUBO (quadratic unconstrained binary optimization) problem and attacks
it with off-the-shelf combinatorial solvers.

The full method is a hybrid pipeline:

```
QUBO encoding  ->  combinatorial solver  ->  top-K candidate subsets
               ->  NNLS concentration refine  ->  concentration threshold
               ->  BIC model selection  ->  final support + concentrations
```

The QUBO stage explores the combinatorial subset space; the NNLS refine
stage re-fits non-negative concentrations continuously; the threshold drops
spurious components; and the Bayesian information criterion (BIC) picks the
final model among the top-K candidates.

## Installation

Core dependencies are `numpy`, `scipy` and `scikit-learn`. Solver backends
are optional extras. To install everything (editable install, all solvers):

```bash
pip install -e '.[all]'
```

Selective extras:

```bash
pip install -e '.[milp]'    # PuLP / CBC MILP solver
pip install -e '.[cpsat]'   # OR-Tools CP-SAT solver
pip install -e '.[sa]'      # simulated annealing (dimod)
pip install -e '.[tabu]'    # tabu search (dimod + dwave-tabu)
pip install -e '.[gpu]'     # GPU simulated annealing (torch)
```

Pinned versions of the reference environment are listed in
[`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

## Quickstart

A fully synthetic, self-contained example
([`examples/quickstart.py`](examples/quickstart.py)):

```python
import numpy as np
import qubic

# Synthetic reference library: 10 energy points, 5 candidate references.
np.random.seed(42)
A = np.random.rand(10, 5)              # non-negative reference spectra

# Ground-truth mixture: references 1 and 3 with concentrations 0.4 / 0.6.
true_idx = [1, 3]
true_conc = np.array([0.4, 0.6])
target = A[:, true_idx] @ true_conc + 0.01 * np.random.randn(10)

# 1. Build the QUBO for the B1 encoding (1 binary variable per reference).
Q, q, var_map, decode = qubic.build_qubo("B1", A, target, k=2)

# 2. Solve it with the MILP backend (CBC via PuLP).
solutions = qubic.solve("milp", Q, q, var_map, top_k=10)

# 3. Or run the full hybrid pipeline in one call.
result = qubic.run_pipeline(A, target, k_true=2, gt_indices=true_idx,
                            scheme="B1", solver="milp", top_k=10)
print("f1 =", result["f1"], " k_pred =", result["k_pred"])
print("concentrations =", result["concentrations"])

# 4. Compare against the classical baselines.
baselines = qubic.run_all_baselines(A, target, gt_indices=true_idx)
for name, res in baselines.items():
    print(name, "f1 =", res["f1"], "k_pred =", res["k_pred"])
```

## QUBO encodings

Each scheme encodes the subset-selection problem with a different number of
binary variables per reference component and different penalty terms.

| Scheme      | Variables per component | Description                                                    |
|-------------|-------------------------|----------------------------------------------------------------|
| `B1`        | 1                       | binary selection only                                          |
| `B2`        | 3                       | selector + 2 quantization bits                                 |
| `3bit`      | 3                       | Rubtsov anti-mixing penalty, not linked                        |
| `4bit`      | 4                       | Rubtsov anti-mixing penalty, not linked                        |
| `linked_B2` | 3                       | selector + 2 quantization bits (selector linked to the bits)   |
| `linked_B3` | 4                       | selector + 3 quantization bits (selector linked to the bits)   |
| `linked_B4` | 5                       | selector + 4 quantization bits (selector linked to the bits)   |

All schemes are available through the single dispatcher
`qubic.build_qubo(scheme, A, target, k, **kwargs)`, which returns
`(Q, q, var_map, decode)`.

## Solvers

All solvers are accessed through `qubic.solve(name, Q, q, var_map, ...)`.
Backends are imported lazily; only the solver you request needs its
dependency installed.

| Solver   | Backend                            | Dependency (optional)   | Install extra              |
|----------|------------------------------------|-------------------------|----------------------------|
| `milp`   | MILP / CBC via PuLP                | `pulp`                  | `pip install 'qubic-lcf[milp]'`  |
| `cp_sat` | OR-Tools CP-SAT                    | `ortools`               | `pip install 'qubic-lcf[cpsat]'` |
| `scip`   | SCIP via PySCIPOpt                 | `pyscipopt`             | `pip install pyscipopt`         |
| `sa`     | simulated annealing                | `dimod`                 | `pip install 'qubic-lcf[sa]'`   |
| `tabu`   | tabu search                        | `dimod`, `dwave-tabu`   | `pip install 'qubic-lcf[tabu]'` |
| `gpu_sa` | GPU simulated annealing            | `torch`                 | `pip install 'qubic-lcf[gpu]'`  |
| `qaoa`   | QAOA (statevector simulation)      | none (numpy/scipy only) | — (core install)               |

## Baselines

Classical non-combinatorial baselines for comparison, all available via
`qubic.run_all_baselines(A, target, gt_indices=...)` or individually:

| Name                  | Method                                                             |
|-----------------------|--------------------------------------------------------------------|
| `omp`                 | orthogonal matching pursuit + BIC                                  |
| `mp`                  | matching pursuit + BIC                                             |
| `lasso+`              | non-negative Lasso (`lasso_positive`) + BIC                        |
| `topk_corr`           | top-K references by correlation + BIC                              |
| `oracle_exact`        | brute-force enumeration of all subsets (oracle upper bound)        |

## PyFitIt integration

QUBIC-LCF can be used as a drop-in mixture-search backend for
[PyFitIt](https://github.com/steam-mock/PyFitIt)-style workflows. The
`qubic.pyfitit_api` module exposes an analogue of PyFitIt's
`tryAllMixtures`: it enumerates candidate mixtures with the QUBO pipeline
instead of PyFitIt's built-in search.

```python
from qubic.pyfitit_api import QubicBackend, try_all_mixtures

backend = QubicBackend(scheme="B2", solver="milp", top_k=10)
result = try_all_mixtures(A, target, backend=backend)
print(result["f1"], result["k_pred"], result["concentrations"])
```

See [`examples/pyfitit_integration.py`](examples/pyfitit_integration.py) for
a complete runnable example. **PyFitIt itself is optional** and is not a
dependency of `qubic-lcf`; install it separately only if you need the full
PyFitIt workflow. The core pipeline works without it.

## Data

**Experimental data are NOT included in this repository.** No measured
spectra, reference libraries, or benchmark `.npy`/`.csv` files are shipped
here. The only data in the repository are *synthetic*: the generator in
[`examples/quickstart.py`](examples/quickstart.py) creates a small random
reference library and a ground-truth mixture so that every part of the
pipeline can be exercised end to end without any external files.

To reproduce the benchmarks of the accompanying article:

1. Obtain the experimental XAS/XANES reference library and target spectra
   (see the data availability statement of the article).
2. Build the reference matrix `A` (energy points x references) and target
   vectors `y`, e.g. as NumPy arrays saved to `.npy` files outside this
   repository (the `data/` and `results/` directories are git-ignored by
   design).
3. Run the pipeline over the mixtures with `qubic.run_pipeline` and the
   baselines with `qubic.run_all_baselines`, sweeping the encoding schemes
   and solvers listed above.
4. Aggregate the per-mixture metrics (`f1`, `k_pred`, concentration errors)
   as reported in the article.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 QUBIC-LCF
authors.

## Citation

If you use this code, please cite the accompanying article (in preparation).
