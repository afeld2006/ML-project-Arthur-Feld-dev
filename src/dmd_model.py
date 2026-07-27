"""Multi-feature linear model on the DMD descriptors.

The DMD dataset holds 3r features per signal : an oscillation frequency, a
growth rate and an energy for each of r modes. Two preparation steps are
needed before a linear model can use them.

  1. Energy is log-transformed. The raw energy spans about ten orders of
     magnitude across the set, with a few near-stationary growing modes
     reaching values hundreds of thousands of times the median. A logarithm
     turns this multiplicative spread into an additive one, the same reason
     the mean-frequency work is done in dB. log is a fixed function, so it is
     applied once and carries no information between signals.

  2. Every feature is standardised. Frequency, growth rate and log-energy sit
     on different scales and in different units, so without standardisation
     the fit and its coefficients would be dominated by whichever feature
     happens to be largest. The mean and standard deviation used are computed
     from the training rows only, and then applied to the held-out rows ; a
     scaler fitted on the full set would let the held-out signals influence
     the scale, which is a form of leakage. This is why standardisation is
     performed inside the cross-validation loop rather than once on the file.

The evaluation protocol matches the mean-frequency model : a 20% holdout is
set aside, the remaining 80% is split into folds, the fold with the lowest
validation RMSE is selected, and that model is scored on the untouched holdout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

TARGET_COL = "annoyance_mean"       # model target y

# The mean-frequency feature from the MVP, joined back in when requested so its
# contribution can be compared against the DMD descriptors on equal footing
CENTROID_COL = "mean_frequency_hz"

# Energy floor used before the logarithm ; a padded mode has energy exactly 0,
# and log(0) is undefined. No padding occurs at the ranks used here, but the
# floor keeps the transform safe for any rank or dataset.
ENERGY_FLOOR = 1e-12


def feature_columns(n_modes: int, with_centroid: bool = False) -> list[str]:
    """Names of the feature columns, ordered mode by mode.

    Modes are numbered by descending energy, so selecting a smaller n_modes
    keeps the strongest modes and drops the weakest. When with_centroid is set
    the mean-frequency feature is appended as one extra column.
    """
    names: list[str] = []
    for i in range(1, n_modes + 1):
        names.extend([f"dmd_f_{i}_hz", f"dmd_g_{i}_per_s", f"dmd_a_{i}"])
    if with_centroid:
        names.append(CENTROID_COL)
    return names


def energy_columns(n_modes: int) -> list[str]:
    """Names of the energy columns among the features, which are logged."""
    return [f"dmd_a_{i}" for i in range(1, n_modes + 1)]


def load_dataset(
    dataset_csv: str | Path,
    n_modes: int,
    with_centroid: bool = False,
    centroid_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load the DMD dataset and keep the rows and columns this rank needs.

    When with_centroid is set, the mean-frequency feature is merged in from
    centroid_csv on the filename, so a single frame carries both the DMD
    descriptors and the MVP feature.
    """
    df = pd.read_csv(dataset_csv)

    if with_centroid:
        if centroid_csv is None:
            raise ValueError("centroid_csv is required when with_centroid is set")
        mvp = pd.read_csv(centroid_csv)
        # bring only the filename key and the one feature, to avoid duplicating
        # the label or the join columns already present in the DMD dataset
        df = df.merge(mvp[["filename", CENTROID_COL]], on="filename", how="left")

    needed = feature_columns(n_modes, with_centroid=with_centroid) + [TARGET_COL]
    # Drop any row missing a feature or the label, then reset for a clean frame
    return df.dropna(subset=needed).reset_index(drop=True)


def build_feature_matrix(
    df: pd.DataFrame,
    n_modes: int,
    with_centroid: bool = False,
) -> np.ndarray:
    """Assemble the feature matrix, with energy columns log-transformed.

    log is applied here because it is a fixed function that uses nothing from
    other rows ; only the later standardisation must be confined to training
    rows. The centroid, when present, is left in Hz and only standardised, as
    it is not skewed the way the energy columns are.
    """
    cols = feature_columns(n_modes, with_centroid=with_centroid)
    X = df[cols].to_numpy(dtype=float)

    # take the base-10 logarithm of the energy columns only
    energy_names = energy_columns(n_modes)
    energy_idx = [cols.index(name) for name in energy_names]
    X[:, energy_idx] = np.log10(np.maximum(X[:, energy_idx], ENERGY_FLOOR))

    return X


def standardise(train: np.ndarray, other: np.ndarray):
    """Standardise both arrays using statistics from the training rows only.

    Returns the standardised training array, the standardised other array, and
    the mean and scale, so a caller can report or reuse them.
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    # a constant feature has zero spread ; dividing by 1 leaves it at zero
    scale = np.where(std > 0, std, 1.0)

    train_s = (train - mean) / scale
    other_s = (other - mean) / scale
    return train_s, other_s, mean, scale


def cross_validate_dmd(
    df: pd.DataFrame,
    n_modes: int,
    n_splits: int = 5,
    holdout_size: float = 0.20,
    random_state: int = 42,
    with_centroid: bool = False,
) -> dict:
    """Hold out a final test set, run k-fold on the rest, keep the best fold.

    Mirrors the mean-frequency procedure, with standardisation refitted inside
    every fold on that fold's training rows.
    """
    X = build_feature_matrix(df, n_modes, with_centroid=with_centroid)
    y = df[TARGET_COL].to_numpy(dtype=float)

    # Step 1 : separate the final test set from the pool used for the folds
    X_pool, X_holdout, y_pool, y_holdout = train_test_split(
        X, y, test_size=holdout_size, random_state=random_state
    )

    # Step 2 : shuffle the pool then divide it into folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rmse = []      # validation RMSE per fold
    fold_r2 = []        # validation R2 per fold
    fold_n_val = []     # size of each validation fold
    fold_state = []     # the fitted pieces needed to score the holdout later

    # Step 3 : each pass trains on the other folds and validates on this one
    for train_idx, val_idx in kf.split(X_pool):
        # standardisation is fitted on this fold's training rows only
        X_tr, X_val, mean, scale = standardise(
            X_pool[train_idx], X_pool[val_idx]
        )

        model = LinearRegression().fit(X_tr, y_pool[train_idx])
        y_hat = model.predict(X_val)

        fold_rmse.append(float(np.sqrt(mean_squared_error(y_pool[val_idx], y_hat))))
        fold_r2.append(float(r2_score(y_pool[val_idx], y_hat)))
        fold_n_val.append(int(len(val_idx)))
        fold_state.append((model, mean, scale))

    # Step 4 : pick the fold with the lowest validation RMSE
    best_index = int(np.argmin(fold_rmse))
    best_model, best_mean, best_scale = fold_state[best_index]

    # Step 5 : score the selected model on the untouched holdout, standardised
    # with that fold's training statistics
    X_holdout_s = (X_holdout - best_mean) / best_scale
    y_pred_holdout = best_model.predict(X_holdout_s)
    holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, y_pred_holdout)))
    holdout_r2 = float(r2_score(y_holdout, y_pred_holdout))

    # Baseline : always predict the training-pool mean annoyance
    baseline_pred = np.full_like(y_holdout, y_pool.mean(), dtype=float)
    holdout_baseline_rmse = float(
        np.sqrt(mean_squared_error(y_holdout, baseline_pred))
    )

    return {
        "n_modes": n_modes,
        "n_features": X.shape[1],
        "n_splits": n_splits,
        "n_pool": int(len(y_pool)),
        "n_holdout": int(len(y_holdout)),
        "fold_rmse": fold_rmse,
        "fold_r2": fold_r2,
        "fold_n_val": fold_n_val,
        "best_fold": best_index + 1,            # 1-based for reporting
        "best_fold_rmse": fold_rmse[best_index],
        "best_fold_r2": fold_r2[best_index],
        "coefficients": best_model.coef_.copy(),   # standardised, so comparable
        "intercept": float(best_model.intercept_),
        "feature_names": feature_columns(n_modes, with_centroid=with_centroid),
        "y_holdout": y_holdout,
        "y_pred_holdout": y_pred_holdout,
        "holdout_rmse": holdout_rmse,             # reported result
        "holdout_r2": holdout_r2,                 # reported result
        "holdout_baseline_rmse": holdout_baseline_rmse,
    }


def plot_predicted_vs_actual(cv: dict, out_path: str | Path) -> None:
    """Predicted against actual annoyance on the holdout signals."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend: write a file, no window
    import matplotlib.pyplot as plt

    y_true = cv["y_holdout"]
    y_pred = cv["y_pred_holdout"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, color="tab:blue", s=45, alpha=0.85, zorder=3)

    # the diagonal y = x is the perfect-prediction reference, not a fit
    lims = [min(y_true.min(), y_pred.min()) - 0.2,
            max(y_true.max(), y_pred.max()) + 0.2]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect prediction", zorder=2)

    ax.set_xlabel("Actual annoyance")
    ax.set_ylabel("Predicted annoyance")
    ax.set_title(f"Predicted vs actual : holdout set (r = {cv['n_modes']})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(frameon=False)
    ax.grid(True, color="0.92", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coefficients(cv: dict, out_path: str | Path) -> None:
    """Standardised coefficients, one bar per feature.

    Because the features were standardised, the bars are directly comparable :
    a taller bar means the feature moves the prediction more per standard
    deviation of its own change.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = cv["feature_names"]
    coefs = cv["coefficients"]

    # colour by descriptor type so the families are easy to tell apart
    colours = []
    for name in names:
        if name == CENTROID_COL:
            colours.append("tab:red")
        elif "_f_" in name:
            colours.append("tab:blue")
        elif "_g_" in name:
            colours.append("tab:orange")
        else:
            colours.append("tab:green")

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(names))
    ax.bar(positions, coefs, color=colours, alpha=0.85)
    ax.axhline(0.0, color="0.4", linewidth=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Standardised coefficient")
    ax.set_title(f"Model coefficients (r = {cv['n_modes']}, "
                 f"{cv['n_features']} features)")
    ax.grid(True, axis="y", color="0.92", linewidth=0.6)

    # a small legend mapping colour to descriptor family
    from matplotlib.patches import Patch
    legend = [
        Patch(color="tab:blue", label="frequency"),
        Patch(color="tab:orange", label="growth rate"),
        Patch(color="tab:green", label="log energy"),
    ]
    if CENTROID_COL in names:
        legend.append(Patch(color="tab:red", label="mean frequency"))
    ax.legend(handles=legend, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def format_report(cv: dict) -> str:
    """Assemble the per-rank metrics report."""
    lines = [
        f"DMD linear model  (r = {cv['n_modes']}, {cv['n_features']} features)",
        "=" * 56,
        f"pool used for folds: {cv['n_pool']} signals",
        f"final holdout set:   {cv['n_holdout']} signals (never seen)",
        "",
        "Validation results per fold:",
    ]
    for i, (rmse, r2, n_val) in enumerate(
        zip(cv["fold_rmse"], cv["fold_r2"], cv["fold_n_val"]), start=1
    ):
        marker = "  <= best" if i == cv["best_fold"] else ""
        lines.append(f"  fold {i} (val n={n_val}):  "
                     f"RMSE = {rmse:.4f}   R2 = {r2:.4f}{marker}")

    lines += [
        "",
        f"Selected fold: {cv['best_fold']} (lowest validation RMSE)",
        "",
        "FINAL RESULTS on the held-out 20 percent:",
        f"  RMSE:          {cv['holdout_rmse']:.4f}",
        f"  R2:            {cv['holdout_r2']:.4f}",
        f"  baseline RMSE: {cv['holdout_baseline_rmse']:.4f}  "
        f"(predicting the pool mean)",
        "",
        "Standardised coefficients:",
    ]
    for name, coef in zip(cv["feature_names"], cv["coefficients"]):
        lines.append(f"  {name:<16}: {coef:+.4f}")
    lines.append(f"  {'intercept':<16}: {cv['intercept']:+.4f}")

    return "\n".join(lines)