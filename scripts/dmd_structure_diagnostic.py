"""
Cross-signal structure of the DMD modes, and the effect of mean-centring.

The rank diagnostic asked how much structure each spectrogram contains. This
script asks two further questions across the whole set:

  1. Do the modes recur? Every signal is decomposed separately, so its modes
     carry no relation to any other signal's. Pooling the descriptors of all
     signals shows whether the same dynamics appear repeatedly, which would
     be visible as dense regions in the frequency against growth-rate plane.

  2. Does mean-centring behave as assumed? Each spectrogram is decomposed
     twice, with and without the time-mean removed. Without centring a mode
     is expected to be spent representing the stationary background, giving
     a frequency and a growth rate near zero.

Modes are compared by their descriptors, never by their position in the
returned list: the ordering carries no meaning across separate decompositions.

Run from the project root:
    py scripts/dmd_structure_diagnostic.py data/raw --freq-key freq
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
from dmd import dmd, mode_energies, mode_frequencies, mode_growth_rates

# A mode counts as stationary when both descriptors sit this close to zero
STATIONARY_F_TOL = 1e-6   # Hz
STATIONARY_G_TOL = 1e-6   # 1/s

# Conjugate pairs describe one oscillation; two modes are treated as the same
# physical component when their descriptors agree to this precision
PAIR_DECIMALS = 6


def load_linear_spectrogram(
    mat_path: Path,
    freq_key: str | None,
    spectrogram_key: str | None,
    values_are_db: bool | str,
) -> np.ndarray:
    """Load one .mat file and return its spectrogram as linear power.

    Identical handling to the rank diagnostic, so the two scripts describe
    the same data.
    """
    mat = load_any_mat(mat_path)

    _, freq = find_frequency_vector(mat, freq_key=freq_key)
    detected_key, spectrogram = find_spectrogram_matrix(
        mat, freq=freq, spectrogram_key=spectrogram_key
    )

    if values_are_db == "auto":
        is_db = detect_db_values(spectrogram, detected_key)
    else:
        is_db = bool(values_are_db)

    # DMD assumes a linear relation between snapshots; dB is logarithmic
    if is_db:
        spectrogram = 10.0 ** (spectrogram / 10.0)

    return np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)


def deduplicate_pairs(freqs, growths, energies):
    """Keep one representative per complex-conjugate pair.

    A real oscillation is carried by two modes with identical descriptors.
    Counting both would double every point in the pooled plots and inflate
    any apparent density, so only the first of each pair is kept.
    """
    seen = set()
    keep = []

    for i, (f, g) in enumerate(zip(freqs, growths)):
        # rounding groups the two members of a pair onto the same key
        key = (round(float(f), PAIR_DECIMALS), round(float(g), PAIR_DECIMALS))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)

    keep = np.array(keep, dtype=int)
    return freqs[keep], growths[keep], energies[keep]


def describe_modes(spectrogram, rank, dt, mean_centre):
    """Decompose one spectrogram and return its per-mode descriptors."""
    res = dmd(spectrogram, rank=rank, mean_centre=mean_centre, dt=dt)

    freqs = mode_frequencies(res["eigenvalues"], dt)
    growths = mode_growth_rates(res["eigenvalues"], dt)
    energies = mode_energies(
        res["modes"], res["amplitudes"],
        res["eigenvalues"], spectrogram.shape[1],
    )
    return freqs, growths, energies


def plot_frequency_growth(freqs, growths, energies, out_path, n_signals):
    """Scatter every retained mode in the frequency against growth-rate plane.

    Dense regions would mean the same dynamics recur across different
    aircraft and phases; a formless spread would mean each signal is
    decomposed into its own unrelated set of modes.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # marker area follows energy, so the dominant modes are visible as such
    relative = energies / energies.max() if energies.max() > 0 else energies
    sizes = 8.0 + 60.0 * relative

    ax.scatter(freqs, growths, s=sizes, color="tab:blue", alpha=0.45,
               edgecolors="none")
    # a mode on this line neither grows nor decays across the pass
    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=1.0)

    ax.set_xlabel("Mode frequency (Hz)")
    ax.set_ylabel("Mode growth rate (1/s)")
    ax.set_title(f"Pooled DMD modes of {n_signals} spectrograms "
                 f"({len(freqs)} modes)")
    ax.grid(True, color="0.92", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_descriptor_histograms(freqs, growths, out_path):
    """Show how the two dynamic descriptors are distributed across the set.

    A recurring dynamic would appear as a peak; a broad spread would mean
    the modes take no preferred value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(freqs, bins=40, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("Mode frequency (Hz)")
    axes[0].set_ylabel("Number of modes")
    axes[0].set_title("Distribution of mode frequency")
    axes[0].grid(True, color="0.92", linewidth=0.6)

    axes[1].hist(growths, bins=40, color="tab:blue", alpha=0.8)
    axes[1].set_xlabel("Mode growth rate (1/s)")
    axes[1].set_ylabel("Number of modes")
    axes[1].set_title("Distribution of mode growth rate")
    axes[1].grid(True, color="0.92", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_centring_comparison(stationary_counts, out_path):
    """Compare how many stationary modes appear with and without centring."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = ["without centring", "with centring"]
    values = [stationary_counts["uncentred"], stationary_counts["centred"]]

    ax.bar(labels, values, color="tab:blue", alpha=0.85)
    ax.set_ylabel("Signals with a stationary mode")
    ax.set_title("Effect of mean-centring on the returned modes")
    ax.grid(True, axis="y", color="0.92", linewidth=0.6)

    # print the count above each bar so the figure is readable on its own
    for i, value in enumerate(values):
        ax.text(i, value, f" {value}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarise(values, name, unit):
    """Return summary lines for one descriptor distribution."""
    return [
        f"{name} ({unit}):",
        f"  minimum:  {np.min(values):.4f}",
        f"  median:   {np.median(values):.4f}",
        f"  maximum:  {np.max(values):.4f}",
        f"  mean:     {np.mean(values):.4f}",
        f"  st. dev.: {np.std(values):.4f}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-signal structure of the DMD modes."
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
    parser.add_argument("--rank", type=int, default=8,
                        help="Modes kept per signal before pooling (default: 8).")
    parser.add_argument("--output-dir", default="reports/dmd_structure",
                        help="Where to save the figures and the summary.")
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

    pooled_f = []       # descriptors of every retained mode, all signals
    pooled_g = []
    pooled_a = []
    modes_per_signal = []
    stationary_counts = {"centred": 0, "uncentred": 0}
    errors = []

    for mat_path in files:
        try:
            spectrogram = load_linear_spectrogram(
                mat_path, args.freq_key, args.spectrogram_key, values_are_db
            )

            # centred run: this is the configuration the features will use
            f_c, g_c, a_c = describe_modes(
                spectrogram, args.rank, args.dt, mean_centre=True
            )
            f_c, g_c, a_c = deduplicate_pairs(f_c, g_c, a_c)

            pooled_f.append(f_c)
            pooled_g.append(g_c)
            pooled_a.append(a_c)
            modes_per_signal.append(len(f_c))

            # a stationary mode neither oscillates nor changes level
            if np.any((np.abs(f_c) < STATIONARY_F_TOL)
                      & (np.abs(g_c) < STATIONARY_G_TOL)):
                stationary_counts["centred"] += 1

            # uncentred run: kept only to measure what centring removes
            f_u, g_u, _ = describe_modes(
                spectrogram, args.rank, args.dt, mean_centre=False
            )
            if np.any((np.abs(f_u) < STATIONARY_F_TOL)
                      & (np.abs(g_u) < STATIONARY_G_TOL)):
                stationary_counts["uncentred"] += 1

        except Exception as exc:
            # Record the failure so one bad file does not stop the batch
            errors.append((mat_path.name, str(exc)))

    if not pooled_f:
        print("No file could be processed:")
        for name, message in errors:
            print(f"  {name}: {message}")
        return 1

    n_signals = len(pooled_f)
    print(f"Processed {n_signals} files with {len(errors)} errors")
    for name, message in errors:
        print(f"  error {name}: {message}")

    freqs = np.concatenate(pooled_f)
    growths = np.concatenate(pooled_g)
    energies = np.concatenate(pooled_a)

    plot_frequency_growth(freqs, growths, energies,
                          out_dir / "mode_frequency_growth.png", n_signals)
    plot_descriptor_histograms(freqs, growths,
                               out_dir / "descriptor_histograms.png")
    plot_centring_comparison(stationary_counts,
                             out_dir / "centring_comparison.png")

    # ---- summary ----
    lines = [
        "DMD cross-signal structure",
        "=" * 52,
        f"signals processed: {n_signals}",
        f"errors:            {len(errors)}",
        f"modes kept per signal before pairing: {args.rank}",
        f"modes retained after pairing: median "
        f"{int(np.median(modes_per_signal))}, "
        f"min {min(modes_per_signal)}, max {max(modes_per_signal)}",
        f"total modes pooled: {len(freqs)}",
        "",
    ]
    lines += summarise(freqs, "Mode frequency", "Hz")
    lines += [""]
    lines += summarise(growths, "Mode growth rate", "1/s")
    lines += [
        "",
        "Growth rate sign:",
        f"  growing (g > 0): {int(np.sum(growths > 0))}",
        f"  decaying (g < 0): {int(np.sum(growths < 0))}",
        "",
        "Stationary modes (frequency and growth rate both near zero):",
        f"  without centring: {stationary_counts['uncentred']} of {n_signals} signals",
        f"  with centring:    {stationary_counts['centred']} of {n_signals} signals",
    ]

    report = "\n".join(lines)
    (out_dir / "structure_diagnostic.txt").write_text(report + "\n",
                                                      encoding="utf-8")

    print()
    print(report)
    print("\nSaved:")
    for name in ("structure_diagnostic.txt", "mode_frequency_growth.png",
                 "descriptor_histograms.png", "centring_comparison.png"):
        print(f"  {out_dir / name}")
    return 0


# Only run when executed directly, not when imported
if __name__ == "__main__":
    raise SystemExit(main())