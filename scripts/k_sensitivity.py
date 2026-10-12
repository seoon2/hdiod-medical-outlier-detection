"""AUC-ROC of every detector as the neighbourhood size k varies from 10 to 100.

Usage:
    python scripts/k_sensitivity.py --dataset Pima
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hdiod.datasets import LOADERS  # noqa: E402
from hdiod.evaluate import build_models, metrics, run_model, stratified_subsample  # noqa: E402

COLOURS = {"HDIOD": "#e41a1c", "RDOS": "#984ea3", "LOF": "#ff7f00", "KNN": "#4daf4a", "IForest": "#377eb8"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(LOADERS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(range(10, 101, 10)))
    parser.add_argument("--max-samples", type=int, default=2000, help="stratified subsample for large datasets")
    parser.add_argument("--legacy-rdos", action="store_true")
    args = parser.parse_args()

    X, y = stratified_subsample(*LOADERS[args.dataset](), n=args.max_samples)

    rows = []
    for k in args.ks:
        for name, model in build_models(args.dataset, k=k, legacy_rdos=args.legacy_rdos).items():
            labels, scores = run_model(model, X)
            rows.append({"k": k, "Model": name, "AUC-ROC": metrics(y, labels, scores)["AUC-ROC"]})
        print(f"k = {k} done")

    df = pd.DataFrame(rows)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    df.round(4).to_csv(out_dir / f"k_sensitivity_{args.dataset}.csv", index=False)

    plt.figure(figsize=(8, 5))
    for name, group in df.groupby("Model"):
        plt.plot(group["k"], group["AUC-ROC"], marker="o", label=name, color=COLOURS[name])
    plt.xlabel("Number of neighbours k")
    plt.ylabel("AUC-ROC")
    plt.title(f"Sensitivity to k on {args.dataset}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"k_sensitivity_{args.dataset}.png", dpi=150)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
