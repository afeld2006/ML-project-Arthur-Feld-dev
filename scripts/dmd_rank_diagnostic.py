"""
Singular-value diagnostic across the spectrogram set.

The rank r is the one hyperparameter of the DMD feature extraction. This script
runs the decomposition on every .mat file, records the singular-value spectrum
of each, and writes two figures plus a summary, so that r is fixed from the data
rather than chosen arbitrarily.

Loading and key detection are reused from mat_spectrogram_features.py, and the
dB conversion follows the same convention as the mean-frequency feature,
P = 10^(dB/10).

Run from the project root:
    py scripts/dmd_rank_diagnostic.py data/raw --freq-key freq
        --spectrogram-key psdx_dB --values-are-db true
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: write files, no window
import matplotlib.pyplot as plt
import numpy as np

# Make src/ importable so this script can call the shared logic there
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mat_spectrogram_features import (
    detect_db_values,
    find_frequency_vector,
    find_spectrogram_matrix,
    load_any_mat,
)
from dmd import dmd

# Energy levels reported in the summary and drawn on the cumulative plot
ENERGY_LEVELS = (0.90, 0.95, 0.99)


def load_linear_spectrogram(
    mat_path: Path,
    freq_key: str | None,
    spectrogram_key: str | None,
    values_are_db: bool | str,
) -> tuple[np.ndarray, bool]:
    """Load one .mat file and return its spectrogram as linear power.

    DMD assumes a linear relation between successive snapshots, so decibels
    are converted first, exactly as the mean-frequency feature does.
    """
    mat = load_any_mat(mat_path)

    # Reuse the existing detection so the keys match the rest of the project
    _, freq = find_frequency_vector(mat, freq_key=freq_key)
    detected_key, spectrogram = find_spectrogram_matrix(
        mat, freq=freq, spectrogram_key=spectrogram_key
    )

    # find_spectrogram_matrix returns frequency x time, so the snapshots are
    # already the columns, which is the layout DMD expects
    if values_are_db == "auto":
        is_db = detect_db_values(spectrogram, detected_key)
    else:
        is_db = bool(values_are_db)

    if is_db:
        spectrogram = 10.0 ** (spectrogram / 10.0)

    # Clean any non-finite entries before the decomposition
    spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
    return spectrogram, is_db


def plot_spectra(normalised: np.ndarray, median: np.ndarray,
                 out_path: Path) -> None:
    """Overlay every singular-value spectrum, with the median on top."""
    idx = np.arange(1, normalised.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    # One faint line per signal; the shared shape is what matters here
    for row in normalised:
        ax.semilogy(idx, row, color="tab:blue", alpha=0.15, linewidth=1)
    ax.semilogy(idx, median, color="crimson", linewidth=2, label="median")

    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Singular value, normalised by the first")
    ax.set_title(f"Singular-value spectra of {normalised.shape[0]} spectrograms")
    ax.set_xticks(idx)
    ax.grid(True, which="both", color="0.92", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cumulative(cumulative: np.ndarray, out_path: Path) -> None:
    """Show how much energy is captured as more modes are kept."""
    idx = np.arange(1, cumulative.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in cumulative:
        ax.plot(idx, row, color="tab:blue", alpha=0.15, linewidth=1)
    ax.plot(idx, np.median(cumulative, axis=0), color="crimson",
            linewidth=2, label="median")

    # Reference lines at the reported energy levels
    for level in ENERGY_LEVELS:
        ax.axhline(level, color="0.55", linestyle=":", linewidth=1.0)
        ax.text(idx[-1], level, f" {level:.2f}", va="center",
                fontsize=8, color="0.45")

    ax.set_xlabel("Number of modes kept")
    ax.set_ylabel("Fraction of singular energy captured")
    ax.set_title("Cumulative energy against rank")
    ax.set_xticks(idx)
    ax.grid(True, color="0.92", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_report(files, spectra, errors, db_flags, median,
                 cumulative, dt) -> str:
    """Assemble the text summary printed and saved by the script."""
    lines = [
        "DMD rank diagnostic",
        "=" * 52,
        f"files found:     {len(files)}",
        f"files processed: {len(spectra)}",
        f"errors:          {len(errors)}",
        f"snapshots per file: {len(spectra[0]) + 1}",
        f"snapshot spacing:   {dt} s",
        f"treated as dB:      {sorted(set(db_flags))}",
        "",
        "Median normalised singular value by index:",
    ]
    for i, value in enumerate(median, start=1):
        lines.append(f"  index {i:>2}: {value:.6f}")

    lines += ["", "Modes needed to reach a given energy fraction:"]
    for level in ENERGY_LEVELS:
        counts = [int(np.searchsorted(row, level) + 1) for row in cumulative]
        lines.append(
            f"  {level:.2f}: median {int(np.median(counts))}, "
            f"min {min(counts)}, max {max(counts)}"
        )

    # The largest drop between consecutive values marks the boundary between
    # structure and noise; that is where the rank should be cut
    ratios = median[:-1] / median[1:]
    lines += ["", "Ratio of consecutive median singular values:"]
    for i, ratio in enumerate(ratios, start=1):
        lines.append(f"  index {i:>2} to {i + 1:>2}: {ratio:.3f}")

    drop = int(np.argmax(ratios))
    lines += [
        "",
        f"Largest drop: index {drop + 1} to {drop + 2} (ratio {ratios[drop]:.3f})",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Singular-value diagnostic used to fix the DMD rank r."
    )
    parser.add_argument("input_dir", nargs="?", default="data/raw",
                        help="Folder containing .mat files (default: data/raw).")
    parser.add_argument("--freq-key", default=None,
                        help="Manual frequency key, for example freq.")
    parser.add_argument("--spectrogram-key", default=None,
                        help="Manual spectrogram key, for example psdx_dB.")
    parser.add_argument("--values-are-db", choices=["auto", "true", "false"],
                        default="auto",
                        help="Whether spectrogram values are in dB.")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Snapshot spacing in seconds (default: 0.1).")
    parser.add_argument("--output-dir", default="reports/dmd_rank",
                        help="Where to save the figures and the summary.")
    parser.add_argument("--max-modes", type=int, default=20,
                        help="How many singular values to plot and tabulate.")
    args = parser.parse_args()

    # Convert the dB text choice into "auto" / True / False
    if args.values_are_db == "auto":
        values_are_db = "auto"
    elif args.values_are_db == "true":
        values_are_db = True
    else:
        values_are_db = False

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.mat"))
    if not files:
        print(f"No .mat files found in {in_dir}")
        return 1
    print(f"Found {len(files)} files")

    spectra = []
    db_flags = []
    errors = []

    for mat_path in files:
        try:
            spectrogram, is_db = load_linear_spectrogram(
                mat_path, args.freq_key, args.spectrogram_key, values_are_db
            )
            # Full-rank run; the whole spectrum is the point of the diagnostic
            res = dmd(spectrogram, mean_centre=True, dt=args.dt)
            spectra.append(res["singular_values"])
            db_flags.append(is_db)
        except Exception as exc:
            # Record the failure so one bad file does not stop the batch
            errors.append((mat_path.name, str(exc)))

    if not spectra:
        print("No file could be processed:")
        for name, message in errors:
            print(f"  {name}: {message}")
        return 1

    print(f"Processed {len(spectra)} files with {len(errors)} errors")
    for name, message in errors:
        print(f"  error {name}: {message}")

    # Each spectrum is divided by its own first value, so signals recorded at
    # different overall levels can be compared on one axis
    n_modes = min(args.max_modes, min(len(s) for s in spectra))
    normalised = np.array([s[:n_modes] / s[0] for s in spectra])
    median = np.median(normalised, axis=0)

    # Energy is carried by the squared singular values
    energy = normalised ** 2
    cumulative = np.cumsum(energy, axis=1) / np.sum(energy, axis=1, keepdims=True)

    plot_spectra(normalised, median, out_dir / "singular_value_spectra.png")
    plot_cumulative(cumulative, out_dir / "cumulative_energy.png")

    report = build_report(files, spectra, errors, db_flags,
                          median, cumulative, args.dt)
    (out_dir / "rank_diagnostic.txt").write_text(report + "\n", encoding="utf-8")

    print()
    print(report)
    print("\nSaved:")
    for name in ("rank_diagnostic.txt", "singular_value_spectra.png",
                 "cumulative_energy.png"):
        print(f"  {out_dir / name}")
    return 0


# Only run when executed directly, not when imported
if __name__ == "__main__":
    raise SystemExit(main())