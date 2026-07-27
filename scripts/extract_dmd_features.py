"""
Extract DMD features from every .mat spectrogram into one feature table.

Each signal is reduced to n_modes retained modes, and each mode contributes
three descriptors, so the table holds d = 3 * n_modes feature columns. Modes
are numbered by descending energy, which means a smaller model can simply use
the first columns without the extraction being run again.

Run from the project root:
    py scripts/extract_dmd_features.py data/raw --freq-key freq
        --spectrogram-key psdx_dB --values-are-db true --n-modes 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable so this script can call the shared extraction logic
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dmd_features import descriptor_column_names, extract_dmd_features_from_folder


def main() -> None:
    # Every option below is exposed as a flag, so behaviour changes without
    # editing code
    parser = argparse.ArgumentParser(
        description="Extract DMD descriptors from .mat spectrograms."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",              # optional positional: falls back to the default
        default="data/raw",
        help="Folder containing .mat files (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        default="features/dmd_features.csv",
        help="Where to save the output CSV.",
    )
    parser.add_argument(
        "--pattern",
        default="*.mat",
        help="File pattern to search for.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",    # flag: search sub-folders when present
        help="Search subfolders too.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=4,
        help="Modes retained per signal; gives 3 x n_modes features.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Snapshot spacing in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--no-mean-centre",
        action="store_true",    # flag: present means centring is switched off
        help="Skip removing the time-mean snapshot before decomposing.",
    )
    parser.add_argument(
        "--freq-key",
        default=None,           # None: auto-detect the frequency variable
        help="Manual frequency key, for example freq.",
    )
    parser.add_argument(
        "--spectrogram-key",
        default=None,           # None: auto-detect the spectrogram variable
        help="Manual spectrogram key, for example psdx_dB.",
    )
    parser.add_argument(
        "--values-are-db",
        choices=["auto", "true", "false"],  # restrict to the supported values
        default="auto",
        help="Whether spectrogram values are in dB.",
    )
    args = parser.parse_args()

    # Convert the dB text choice into "auto" / True / False for the function
    if args.values_are_db == "auto":
        values_are_db = "auto"
    elif args.values_are_db == "true":
        values_are_db = True
    else:
        values_are_db = False

    # Run the extraction over every matching file; writes the CSV internally
    df = extract_dmd_features_from_folder(
        input_dir=args.input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
        output_csv=args.output,
        n_modes=args.n_modes,
        dt=args.dt,
        mean_centre=not args.no_mean_centre,
        freq_key=args.freq_key,
        spectrogram_key=args.spectrogram_key,
        values_are_db=values_are_db,
    )

    # Print the strongest mode per file plus the filtering counts, so the
    # result is visible in the terminal without opening the CSV
    summary_cols = ["filename", "dmd_f_1_hz", "dmd_g_1_per_s",
                    "n_nyquist_dropped", "status"]
    print(df[summary_cols].to_string(index=False))

    n_ok = int((df["status"] == "ok").sum())
    n_error = int((df["status"] == "error").sum())

    print()
    print(f"Saved CSV to: {args.output}")
    print(f"Successful files: {n_ok}")
    print(f"Files with errors: {n_error}")
    print(f"Feature columns:   {len(descriptor_column_names(args.n_modes))}"
          f"  (3 descriptors x {args.n_modes} modes)")

    # Modes at the sampling limit are discarded and replaced; report how often
    if n_ok:
        dropped = df.loc[df["status"] == "ok", "n_nyquist_dropped"]
        padded = df.loc[df["status"] == "ok", "n_modes_padded"]
        print(f"Nyquist modes dropped: {int(dropped.sum())} across "
              f"{int((dropped > 0).sum())} files")
        # padding means a signal held fewer components than were requested
        if padded.sum() > 0:
            print(f"Signals padded with empty modes: {int((padded > 0).sum())}"
                  f"  (n_modes exceeds what those signals contain)")

    # List any files that failed, so problems are easy to spot
    if n_error:
        print("\nThese files could not be processed:")
        print(df.loc[df["status"] == "error", ["filename", "error"]]
              .to_string(index=False))


# Only run when executed directly, not when imported
if __name__ == "__main__":
    main()