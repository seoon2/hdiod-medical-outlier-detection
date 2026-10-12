"""Compare HDIOD with RDOS, LOF, KNN and Isolation Forest at k = 20.

Usage:
    python scripts/run_benchmark.py                      # all datasets
    python scripts/run_benchmark.py --datasets Pima WPBC
    python scripts/run_benchmark.py --legacy-rdos        # reproduce dissertation RDOS
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hdiod.datasets import LOADERS  # noqa: E402
from hdiod.evaluate import build_models, metrics, run_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--legacy-rdos", action="store_true")
    parser.add_argument("--out", default="results/benchmark.csv")
    args = parser.parse_args()

    rows = []
    for name in args.datasets:
        X, y = LOADERS[name]()
        print(f"\n{name}: {X.shape[0]} samples, {X.shape[1]} features, {y.sum()} anomalies ({y.mean():.1%})")
        for model_name, model in build_models(name, k=args.k, legacy_rdos=args.legacy_rdos).items():
            labels, scores = run_model(model, X)
            m = metrics(y, labels, scores)
            rows.append({"Dataset": name, "Model": model_name, **m})
            print(f"  {model_name:8s} AUC {m['AUC-ROC']:.4f}  F1 {m['F1-score']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
