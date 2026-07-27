"""
Train the DMD linear model across several ranks and report each one.

For every rank r requested, the model uses the first r modes of the dataset
(the modes are ordered by energy), evaluates with a 20% holdout and k-fold
selection, and writes a folder reports/dmd_r{r}/ holding its metrics and two
figures. A summary table across all ranks is written to reports/dmd_summary.txt.

Run from the project root:
    py scripts/train_dmd_model.py --dataset features/dmd_dataset.csv --ranks 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable so this script can call the shared model logic
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dmd_model import (
    cross_validate_dmd,
    format_report,
    load_dataset,
    plot_coefficients,
    plot_predicted_vs_actual,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the DMD linear model across several ranks."
    )
    parser.add_argument("--dataset", default="features/dmd_dataset.csv",
                        help="Joined DMD dataset from build_dataset.py.")
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 3, 4],
                        help="Ranks r to evaluate (default: 2 3 4).")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of folds on the 80 percent pool.")
    parser.add_argument("--holdout-size", type=float, default=0.20,
                        help="Fraction held out as the final test set.")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Seed for a reproducible holdout and fold split.")
    parser.add_argument("--output-dir", default="reports",
                        help="Parent folder for the per-rank result folders.")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    summary_rows = []

    for n_modes in args.ranks:
        # Each rank uses only the rows that have all of its features present
        df = load_dataset(args.dataset, n_modes=n_modes)

        cv = cross_validate_dmd(
            df,
            n_modes=n_modes,
            n_splits=args.n_splits,
            holdout_size=args.holdout_size,
            random_state=args.random_state,
        )

        # Write this rank's folder: metrics plus the two figures
        rank_dir = output_root / f"dmd_r{n_modes}"
        rank_dir.mkdir(parents=True, exist_ok=True)

        report = format_report(cv)
        (rank_dir / "metrics.txt").write_text(report + "\n", encoding="utf-8")
        plot_predicted_vs_actual(cv, rank_dir / "predicted_vs_actual.png")
        plot_coefficients(cv, rank_dir / "coefficients.png")

        print(report)
        print(f"\nSaved to {rank_dir}\n")
        print("-" * 56)

        summary_rows.append(cv)

    # Summary table across the ranks
    lines = [
        "DMD model : results across ranks",
        "=" * 56,
        f"dataset: {args.dataset}",
        f"holdout: {int(args.holdout_size * 100)} percent, "
        f"{args.n_splits} folds on the rest",
        "",
        f"{'rank':>4}  {'features':>8}  {'holdout RMSE':>12}  "
        f"{'holdout R2':>10}  {'baseline RMSE':>13}",
    ]
    for cv in summary_rows:
        lines.append(
            f"{cv['n_modes']:>4}  {cv['n_features']:>8}  "
            f"{cv['holdout_rmse']:>12.4f}  {cv['holdout_r2']:>10.4f}  "
            f"{cv['holdout_baseline_rmse']:>13.4f}"
        )

    summary = "\n".join(lines)
    (output_root / "dmd_summary.txt").write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"\nSaved summary to {output_root / 'dmd_summary.txt'}")


# Only run when executed directly, not when imported
if __name__ == "__main__":
    main()