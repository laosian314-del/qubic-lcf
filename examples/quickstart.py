"""QUBIC-LCF quickstart: fully synthetic end-to-end example.

Generates a small random reference library and a known ground-truth mixture,
then exercises the whole hybrid pipeline:

    QUBO -> solver -> top-K candidates -> NNLS refine -> threshold -> BIC

Runs without installing the package: if `qubic` is not importable, `../src`
is added to sys.path. Requires only numpy/scipy/scikit-learn plus the PuLP
package for the `milp` solver (pip install pulp).
"""

import pathlib
import sys

try:
    import qubic  # noqa: F401
except ImportError:
    sys.path.insert(
        0, str(pathlib.Path(__file__).resolve().parents[1] / "src")
    )
    import qubic

import numpy as np


def best_of_pipeline(res):
    """Pick the min-BIC entry from a run_pipeline result dictionary."""
    pool = res["results"]
    return pool[min(pool, key=lambda tk: pool[tk]["bic"])]


def main():
    # --- Synthetic data -------------------------------------------------
    np.random.seed(42)
    n_energy, n_refs = 10, 5
    A = np.random.rand(n_energy, n_refs)  # non-negative reference spectra

    true_idx = [1, 3]
    true_conc = np.array([0.4, 0.6])
    target = A[:, true_idx] @ true_conc
    target = target + 0.01 * np.random.randn(n_energy)

    print("A:", A.shape, " target:", target.shape)
    print("ground truth: references", true_idx, "concentrations", true_conc)

    # --- 1. Build the QUBO (B1: one binary variable per reference) ------
    Q, q, var_map, decode = qubic.build_qubo("B1", A, target, k=2)
    print("QUBO built: Q", Q.shape, "| variables:", var_map["M"])

    # --- 2. Solve the QUBO with the MILP backend -----------------------
    solutions, wall_time, gap = qubic.solve("milp", Q, q, var_map, top_k=10)
    print(f"solver returned {len(solutions)} candidate sample(s) "
          f"in {wall_time:.3f}s")

    # --- 3. Full hybrid pipeline ---------------------------------------
    out = qubic.run_pipeline(
        A,
        target,
        k_true=2,
        gt_indices=true_idx,
        scheme="B1",
        solver="milp",
        top_k_values=(1, 10),
    )
    result = best_of_pipeline(out)
    print("--- QUBIC-LCF pipeline ---")
    print("f1           =", result.get("f1"))
    print("k_pred       =", result.get("k_pred"))
    print("selected     =", result.get("selected"))
    print("concentrations =", result.get("concentrations"))

    # --- 4. Classical baselines ----------------------------------------
    baselines = qubic.run_all_baselines(A, target)
    print("--- baselines ---")
    for name, res in sorted(baselines.items()):
        f1 = qubic.compute_f1(res["selected"], true_idx)
        print(
            f"{name:12s} f1={f1:4.2f} "
            f"k_pred={res.get('k_pred')!s:4s} "
            f"concentrations={res.get('concentrations')}"
        )


if __name__ == "__main__":
    main()
