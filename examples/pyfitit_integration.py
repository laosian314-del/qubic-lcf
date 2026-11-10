"""QUBIC-LCF as a PyFitIt-style mixture-search backend.

`qubic.pyfitit_api` exposes an analogue of PyFitIt's `tryAllMixtures`:
instead of PyFitIt's built-in mixture enumeration, candidate subsets are
found with the QUBO -> top-K -> NNLS -> BIC pipeline.

NOTE: PyFitIt itself is OPTIONAL. `qubic-lcf` does not depend on it; the
full PyFitIt workflow is only needed if you integrate with PyFitIt's own
fitting/validation machinery. The core QUBIC-LCF pipeline works without it.

Runs without installing the package: if `qubic` is not importable, `../src`
is added to sys.path.
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


def main():
    # --- Synthetic data (same generator as examples/quickstart.py) ------
    np.random.seed(42)
    A = np.random.rand(10, 5)  # non-negative reference spectra
    true_idx = [1, 3]
    target = A[:, true_idx] @ np.array([0.4, 0.6]) + 0.01 * np.random.randn(10)

    # --- Use qubic.pyfitit_api as the mixture-search backend ------------
    try:
        from qubic.pyfitit_api import QubicBackend, try_all_mixtures
    except ImportError as exc:
        print(
            "qubic.pyfitit_api is unavailable: %s\n"
            "This example is optional. Install the package with\n"
            "    pip install -e '.[all]'\n"
            "and, for the full PyFitIt workflow, install PyFitIt separately\n"
            "(pip install pyfitit). The core QUBIC-LCF pipeline does not\n"
            "require PyFitIt." % exc
        )
        return

    backend = QubicBackend(scheme="B2", solver="milp", top_k=10)
    result = try_all_mixtures(A, target, backend=backend)

    print("--- qubic.pyfitit_api / try_all_mixtures ---")
    print("f1             =", result.get("f1"))
    print("k_pred         =", result.get("k_pred"))
    print("selected refs  =", result.get("selected"))
    print("concentrations =", result.get("concentrations"))


if __name__ == "__main__":
    main()
