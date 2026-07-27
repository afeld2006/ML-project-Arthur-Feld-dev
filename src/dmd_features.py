"""Turn each spectrogram into a fixed-length DMD feature vector.

One spectrogram gives a set of modes, and each retained mode contributes three
descriptors : oscillation frequency, growth rate and energy. Keeping r modes
therefore gives d = 3r features, the same count for every signal, which is what
a regression model requires.

Three rules decide which modes are retained, applied in this order:

  1. Conjugate pairs are collapsed. A real oscillation is carried by two modes
     with identical descriptors ; keeping both would duplicate every column of
     the feature vector and leave the design matrix rank deficient.

  2. Modes at the Nyquist limit are dropped. An eigenvalue on the negative real
     axis reports a frequency of exactly 1 / (2 dt), the fastest the sampling
     grid allows, rather than a resolved oscillation. Measured on the present
     set these were 12 of 387 pooled modes, affecting 10 of 77 signals.

  3. The r highest-energy survivors are kept, ranked by the energy descriptor.
     Because a dropped mode is replaced by the next one down, every signal
     yields exactly r modes and the feature vector is always the same length.

Loading and key detection are reused from mat_spectrogram_features.py, and the
dB conversion follows the same convention as the mean-frequency feature,
P = 10^(dB/10).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dmd import dmd, mode_energies, mode_frequencies, mode_growth_rates
from mat_spectrogram_features import (
    detect_db_values,
    find_frequency_vector,
    find_spectrogram_matrix,
    load_any_mat,
)

# Two modes are treated as one conjugate pair when their descriptors agree to
# this many decimals; the pair members are identical up to rounding
PAIR_DECIMALS = 6

# A frequency within this distance of the Nyquist limit is taken to be the
# sampling artefact rather than a resolved oscillation
NYQUIST_TOL_HZ = 1e-6

# How many raw modes to request per retained mode. A conjugate pair costs two
# raw modes, and dropped modes need replacements, so the decomposition is asked
# for more than r and the surplus is discarded after filtering.
RAW_MODE_FACTOR = 4


def load_linear_spectrogram(
    mat_path: str | Path,
    freq_key: str | None = None,
    spectrogram_key: str | None = None,
    values_are_db: bool | str = "auto",
) -> tuple[np.ndarray, bool, str]:
    """Load one .mat file and return its spectrogram as linear power.

    DMD assumes a linear relation between successive snapshots, so decibel
    values are converted first ; averaging or advancing dB directly would be
    physically meaningless.

    Returns the matrix (frequency x time), whether dB was assumed, and the
    spectrogram key that was used.
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
    return spectrogram, is_db, detected_key


def deduplicate_pairs(
    freqs: np.ndarray,
    growths: np.ndarray,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep one representative of each complex-conjugate pair.

    Both members of a pair describe the same physical oscillation and carry
    identical descriptors, so keeping both would place duplicate columns in
    the feature vector. Order is preserved, so the higher-energy member of
    each pair is the one kept.
    """
    seen: set[tuple[float, float]] = set()
    keep: list[int] = []

    for i, (f, g) in enumerate(zip(freqs, growths)):
        # rounding maps the two members of a pair onto the same key
        key = (round(float(f), PAIR_DECIMALS), round(float(g), PAIR_DECIMALS))
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)

    index = np.array(keep, dtype=int)
    return freqs[index], growths[index], energies[index]


def drop_nyquist_modes(
    freqs: np.ndarray,
    growths: np.ndarray,
    energies: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Remove modes sitting at the Nyquist limit.

    An eigenvalue on the negative real axis has phase pi, which maps to the
    frequency 1 / (2 dt). Nothing faster can be represented at this sampling
    rate, so the value reports the limit of the grid rather than a measured
    oscillation. These modes also carry extreme growth rates, which would act
    as outliers in a linear model fitted on few signals.

    Returns the surviving descriptors and how many modes were removed.
    """
    nyquist = 1.0 / (2.0 * dt)
    at_limit = np.abs(freqs - nyquist) < NYQUIST_TOL_HZ
    keep = ~at_limit
    return freqs[keep], growths[keep], energies[keep], int(np.sum(at_limit))


def extract_dmd_descriptors(
    spectrogram: np.ndarray,
    n_modes: int,
    dt: float,
    mean_centre: bool = True,
) -> dict[str, Any]:
    """Reduce one spectrogram to n_modes sets of descriptors.

    The decomposition is asked for more raw modes than are needed, because
    conjugate pairs are then collapsed and Nyquist modes discarded ; the
    surplus supplies replacements so that exactly n_modes survive.
    """
    n_snapshots = spectrogram.shape[1]

    # ask for enough raw modes that filtering still leaves n_modes behind
    raw_rank = min(n_modes * RAW_MODE_FACTOR, n_snapshots - 1)

    res = dmd(spectrogram, rank=raw_rank, mean_centre=mean_centre, dt=dt)

    freqs = mode_frequencies(res["eigenvalues"], dt)
    growths = mode_growth_rates(res["eigenvalues"], dt)
    energies = mode_energies(
        res["modes"], res["amplitudes"], res["eigenvalues"], n_snapshots
    )

    # rule 1 : one representative per conjugate pair
    freqs, growths, energies = deduplicate_pairs(freqs, growths, energies)

    # rule 2 : discard the sampling artefacts
    freqs, growths, energies, n_dropped = drop_nyquist_modes(
        freqs, growths, energies, dt
    )

    n_available = len(freqs)

    # rule 3 : keep the strongest survivors, ranked by energy
    order = np.argsort(energies)[::-1][:n_modes]
    kept_f = freqs[order]
    kept_g = growths[order]
    kept_a = energies[order]

    # A signal genuinely containing fewer than n_modes components cannot supply
    # more ; the vector is padded with zero-energy entries so that its length
    # stays fixed, and the shortfall is recorded so it is visible in the table
    # rather than silently absorbed.
    n_padded = max(n_modes - n_available, 0)
    if n_padded > 0:
        pad = np.zeros(n_padded)
        kept_f = np.concatenate([kept_f, pad])
        kept_g = np.concatenate([kept_g, pad])
        kept_a = np.concatenate([kept_a, pad])

    return {
        "frequencies": kept_f,
        "growth_rates": kept_g,
        "energies": kept_a,
        "n_nyquist_dropped": n_dropped,
        "n_modes_available": n_available,
        "n_modes_padded": n_padded,
        "raw_rank": res["rank"],
        "n_snapshots": n_snapshots,
    }


def descriptor_column_names(n_modes: int) -> list[str]:
    """Column names for the feature block, ordered mode by mode.

    Modes are numbered by descending energy, so mode 1 is the strongest.
    """
    names: list[str] = []
    for i in range(1, n_modes + 1):
        names.extend([f"dmd_f_{i}_hz", f"dmd_g_{i}_per_s", f"dmd_a_{i}"])
    return names


def extract_dmd_features_from_mat(
    mat_path: str | Path,
    n_modes: int = 4,
    dt: float = 0.1,
    mean_centre: bool = True,
    freq_key: str | None = None,
    spectrogram_key: str | None = None,
    values_are_db: bool | str = "auto",
) -> dict[str, Any]:
    """Extract one row of DMD features from one .mat file."""
    mat_path = Path(mat_path)

    spectrogram, db_used, detected_key = load_linear_spectrogram(
        mat_path,
        freq_key=freq_key,
        spectrogram_key=spectrogram_key,
        values_are_db=values_are_db,
    )

    descriptors = extract_dmd_descriptors(
        spectrogram, n_modes=n_modes, dt=dt, mean_centre=mean_centre
    )

    # Metadata first, so each row explains how it was produced
    row: dict[str, Any] = {
        "filename": mat_path.name,
        "filepath": str(mat_path),
        "n_modes": n_modes,
        "dt_s": dt,
        "mean_centred": mean_centre,
        "values_are_db": db_used,
        "spectrogram_key": detected_key,
        "n_snapshots": descriptors["n_snapshots"],
        "n_nyquist_dropped": descriptors["n_nyquist_dropped"],
        "n_modes_available": descriptors["n_modes_available"],
        "n_modes_padded": descriptors["n_modes_padded"],
        "status": "ok",
        "error": "",
    }

    # Then the descriptors themselves, three per mode
    for i in range(n_modes):
        row[f"dmd_f_{i + 1}_hz"] = float(descriptors["frequencies"][i])
        row[f"dmd_g_{i + 1}_per_s"] = float(descriptors["growth_rates"][i])
        row[f"dmd_a_{i + 1}"] = float(descriptors["energies"][i])

    return row


def extract_dmd_features_from_folder(
    input_dir: str | Path,
    pattern: str = "*.mat",
    recursive: bool = False,
    output_csv: str | Path | None = None,
    n_modes: int = 4,
    dt: float = 0.1,
    mean_centre: bool = True,
    freq_key: str | None = None,
    spectrogram_key: str | None = None,
    values_are_db: bool | str = "auto",
) -> pd.DataFrame:
    """Extract DMD features from every .mat file in a folder."""
    input_dir = Path(input_dir)

    # Collect the files; recurse into sub-folders only if asked
    if recursive:
        mat_files = sorted(input_dir.rglob(pattern))
    else:
        mat_files = sorted(input_dir.glob(pattern))

    if not mat_files:
        raise FileNotFoundError(f"No files matching {pattern} found in {input_dir}")

    feature_names = descriptor_column_names(n_modes)
    rows = []

    for mat_file in mat_files:
        try:
            row = extract_dmd_features_from_mat(
                mat_path=mat_file,
                n_modes=n_modes,
                dt=dt,
                mean_centre=mean_centre,
                freq_key=freq_key,
                spectrogram_key=spectrogram_key,
                values_are_db=values_are_db,
            )
        except Exception as exc:
            # Record an error row so one bad file does not stop the batch;
            # the same columns are present, filled with NaN
            row = {
                "filename": mat_file.name,
                "filepath": str(mat_file),
                "n_modes": n_modes,
                "dt_s": dt,
                "mean_centred": mean_centre,
                "values_are_db": values_are_db,
                "spectrogram_key": spectrogram_key,
                "n_snapshots": np.nan,
                "n_nyquist_dropped": np.nan,
                "n_modes_available": np.nan,
                "n_modes_padded": np.nan,
                "status": "error",
                "error": str(exc),
            }
            for name in feature_names:
                row[name] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)  # one row per file

    # Optionally write the feature table to disk
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df