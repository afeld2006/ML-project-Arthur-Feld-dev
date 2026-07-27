from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# The three flight phases each signal is recorded in
PHASES = ("Approach", "Flyover", "Recede")
# Columns that describe how a feature row was produced rather than being
# features themselves; everything else in a feature CSV is taken as a feature,
# so the join works for any extraction without naming its columns here
METADATA_COLS = {
    "filename", "filepath", "status", "error",
    # written by the mean-frequency extraction
    "method", "fmin_hz", "fmax_hz", "values_are_db", "freq_key",
    "spectrogram_key", "n_frequency_bins", "n_time_bins",
    # written by the DMD extraction
    "n_modes", "dt_s", "mean_centred", "n_snapshots",
    "n_nyquist_dropped", "n_modes_available", "n_modes_padded",
    # join keys, added during the merge
    "AudioNum", "Phase",
}
# Regex to pull the signal number and phase out of a filename; case-insensitive
_FILENAME_RE = re.compile(r"SignalNum(\d+).*?(Approach|Flyover|Recede)", re.IGNORECASE)


def parse_signal_and_phase(filename: str) -> tuple[int | None, str | None]:
    """Pull (SignalNum, Phase) out of a .mat filename like
    'SignalNum107_A350-941_Approach.mat'. Returns (None, None) if no match."""
    m = _FILENAME_RE.search(str(filename))  # search anywhere in the name
    if not m:
        return None, None  # no signal/phase found
    phase = m.group(2).capitalize()  # normalise case, e.g. "APPROACH" : "Approach"
    return int(m.group(1)), phase  # signal number as int, phase as text


def load_and_aggregate_labels(
    xlsx_path: str | Path,
    sheet_name: str,
    include_zero: bool = True,
) -> pd.DataFrame:
    """Load the psychoacoustic sheet, compute the per-row mean annoyance across
    participant columns, then average any repeated (AudioNum, Phase) presentations."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    # Participant rating columns are the integer-named ones (1, 2, 3, ...)
    participant_cols = [c for c in df.columns if isinstance(c, int)]

    ratings = df[participant_cols]
    if not include_zero:
        ratings = ratings.replace(0, pd.NA)  # treat 0 as missing so it is skipped

    # Per-row label: mean annoyance across the participants
    df = df.assign(annoyance_mean=ratings.mean(axis=1))

    # Some (signal, phase) stimuli appear twice (test-retest): average them together
    agg = (
        df.groupby(["AudioNum", "Phase"], as_index=False)
        .agg(
            annoyance_mean=("annoyance_mean", "mean"),      # combined mean rating
            n_presentations=("annoyance_mean", "size"),     # how many rows were merged
        )
    )
    return agg


def build_modeling_dataset(
    features_csv: str | Path,
    xlsx_path: str | Path,
    sheet_name: str,
    include_zero: bool = True,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join mean-frequency features to aggregated annoyance labels on
    (SignalNum, Phase). Returns (merged_ok, unmatched_features)."""
    feats = pd.read_csv(features_csv)
    feats = feats[feats["status"] == "ok"].copy()  # keep only successfully extracted files

    # Derive the join keys (signal number, phase) from each filename
    parsed = feats["filename"].apply(parse_signal_and_phase)
    feats["AudioNum"] = parsed.apply(lambda t: t[0])  # signal number
    feats["Phase"] = parsed.apply(lambda t: t[1])     # phase

    labels = load_and_aggregate_labels(xlsx_path, sheet_name, include_zero=include_zero)

    # Left join: every feature row tries to find its matching label
    merged = feats.merge(labels, on=["AudioNum", "Phase"], how="left")

    # Split into matched vs. unmatched (unmatched have no annoyance label)
    ok = merged[merged["annoyance_mean"].notna()].copy()
    unmatched = merged[merged["annoyance_mean"].isna()].copy()

    # Keep only the modelling columns, ordered and sorted for a clean output.
    # Anything that is not metadata counts as a feature, so one mean-frequency
    # column and a block of DMD descriptors are both handled without change.
    if feature_cols is None:
        feature_cols = [c for c in feats.columns if c not in METADATA_COLS]

    cols = (["filename", "AudioNum", "Phase"]
            + list(feature_cols)
            + ["annoyance_mean", "n_presentations"])
    ok = ok[cols].sort_values(["AudioNum", "Phase"]).reset_index(drop=True)
    return ok, unmatched
