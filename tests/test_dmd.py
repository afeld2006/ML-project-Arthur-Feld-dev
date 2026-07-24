"""
Verification of the DMD implementation on synthetic data with known answers.

Same approach as test_sinusoid_recovery.py : the input is built from modes
whose frequency and growth rate are set in advance, and the test checks the
values DMD returns against those inputs.

Six tests:
  A) Single decaying oscillation - one planted mode must be recovered.
  B) Two superposed modes        - both must be separated and recovered.
  C) Pure growth                 - a real eigenvalue must give frequency 0.
  D) Rank truncation             - a weak mode must be dropped when r is small.
  E) Mean-centring               - a constant background costs one mode.
  F) Helper functions            - snapshot pairing and rank selection.

Run from the project root:
    py tests/test_dmd.py
"""

import sys
from pathlib import Path

import numpy as np

# Make src/ importable regardless of where we run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dmd import (
    dmd,
    mode_frequencies,
    mode_growth_rates,
    mode_energies,
    build_snapshot_pairs,
    choose_rank,
)

DT = 0.1        # snapshot spacing in seconds; matches the real spectrograms
TOL = 1e-6      # tolerance for recovered frequencies and growth rates


def check(name, got, expected, tol=TOL):
    # Compare within a tolerance to allow for floating-point rounding
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got {got:.6f}, expected {expected:.6f}")
    if not ok:
        raise AssertionError(f"{name} failed (tolerance {tol})")


def report(name, ok, detail=""):
    # For checks whose result is a condition rather than a number
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{detail}")
    if not ok:
        raise AssertionError(f"{name} failed")


def make_signal(freqs, growths, shapes_cos, n_snapshots,
                amplitudes=None, shapes_sin=None):
    """Build a real snapshot matrix from modes with known properties.

    A real oscillation at frequency f is produced by a complex-conjugate PAIR
    of DMD modes, and a pair needs TWO independent spatial directions. So each
    oscillatory mode is built as
        amp * exp(g t) * [shape_cos * cos(2 pi f t) + shape_sin * sin(2 pi f t)]
    with the two shapes linearly independent. Building it from a single shape
    would make the matrix rank 1, and no algorithm could recover a pair from it.

    A mode with f = 0 is pure growth or decay and needs only one direction.
    """
    t = np.arange(n_snapshots) * DT
    n_states = len(shapes_cos[0])
    data = np.zeros((n_states, n_snapshots), dtype=float)

    if amplitudes is None:
        amplitudes = [1.0] * len(freqs)
    if shapes_sin is None:
        shapes_sin = [None] * len(freqs)

    for f, g, shape_c, shape_s, amp in zip(
        freqs, growths, shapes_cos, shapes_sin, amplitudes
    ):
        shape_c = np.asarray(shape_c, dtype=float).reshape(-1, 1)
        envelope = np.exp(g * t).reshape(1, -1)   # growth or decay over time

        if f == 0.0:
            data += amp * (shape_c @ envelope)    # one direction is enough
            continue

        if shape_s is None:
            raise ValueError("an oscillatory mode needs a second shape vector")

        shape_s = np.asarray(shape_s, dtype=float).reshape(-1, 1)
        cos_t = np.cos(2.0 * np.pi * f * t).reshape(1, -1)
        sin_t = np.sin(2.0 * np.pi * f * t).reshape(1, -1)
        data += amp * (shape_c @ (envelope * cos_t)
                       + shape_s @ (envelope * sin_t))

    return data


def test_a_single_oscillation():
    """One decaying oscillation; frequency and growth rate must both return."""
    print("A) Single decaying oscillation")
    f_true, g_true = 1.25, -0.8   # planted values; below Nyquist at 5 Hz

    shape_c = np.array([1.0, 0.5, -0.3, 0.2, 0.9])
    shape_s = np.array([0.2, -0.7, 0.4, 1.0, -0.1])
    data = make_signal([f_true], [g_true], [shape_c], 40, shapes_sin=[shape_s])

    res = dmd(data, rank=2, mean_centre=False, dt=DT)
    freqs = mode_frequencies(res["eigenvalues"], DT)
    growths = mode_growth_rates(res["eigenvalues"], DT)

    # The conjugate pair shares one frequency and one growth rate
    check("  planted frequency", float(freqs[0]), f_true)
    check("  planted growth rate", float(growths[0]), g_true)
    print()


def test_b_two_modes():
    """Two oscillations; each growth rate must return with its own frequency."""
    print("B) Two superposed modes")
    f1, g1 = 0.5, -0.3
    f2, g2 = 2.0, -1.5

    # Each oscillation needs its own pair of directions, so four in total
    shape1c = np.array([1.0, 0.0, 0.0, 0.5, 0.2])
    shape1s = np.array([0.3, 0.8, -0.2, 0.0, 1.0])
    shape2c = np.array([0.0, 1.0, 0.4, 0.0, -0.6])
    shape2s = np.array([-0.5, 0.1, 1.0, 0.7, 0.0])

    data = make_signal([f1, f2], [g1, g2], [shape1c, shape2c], 60,
                       shapes_sin=[shape1s, shape2s])

    res = dmd(data, rank=4, mean_centre=False, dt=DT)
    all_f = mode_frequencies(res["eigenvalues"], DT)
    all_g = mode_growth_rates(res["eigenvalues"], DT)

    unique_f = np.sort(np.unique(np.round(all_f, 6)))
    check("  lower frequency", float(unique_f[0]), min(f1, f2), 1e-5)
    check("  upper frequency", float(unique_f[-1]), max(f1, f2), 1e-5)

    # Pair each recovered frequency with its own growth rate
    i1 = int(np.argmin(np.abs(all_f - f1)))
    i2 = int(np.argmin(np.abs(all_f - f2)))
    check("  growth rate of mode 1", float(all_g[i1]), g1, 1e-5)
    check("  growth rate of mode 2", float(all_g[i2]), g2, 1e-5)
    print()


def test_c_pure_growth():
    """A real eigenvalue has no imaginary part, so the frequency must be 0."""
    print("C) Pure growth, no oscillation")
    g_pure = 0.4
    shape = np.array([0.7, -0.2, 0.5, 1.0, 0.1])
    data = make_signal([0.0], [g_pure], [shape], 30)

    res = dmd(data, rank=1, mean_centre=False, dt=DT)
    freqs = mode_frequencies(res["eigenvalues"], DT)
    growths = mode_growth_rates(res["eigenvalues"], DT)

    check("  frequency is zero", float(freqs[0]), 0.0, 1e-9)
    check("  growth rate", float(growths[0]), g_pure)
    print()


def test_d_rank_truncation():
    """A mode planted far below the other must be dropped when r is reduced."""
    print("D) Rank truncation")
    f_strong, g_strong = 1.0, -0.2
    f_weak, g_weak = 3.0, -0.5

    shape_sc = np.array([1.0, 0.2, 0.0, 0.3, 0.1])
    shape_ss = np.array([0.4, -0.6, 0.9, 0.0, 0.2])
    shape_wc = np.array([0.0, 1.0, 0.5, 0.0, -0.2])
    shape_ws = np.array([0.1, 0.0, -0.3, 1.0, 0.5])

    # The second mode carries 1/1000 of the amplitude of the first
    data = make_signal([f_strong, f_weak], [g_strong, g_weak],
                       [shape_sc, shape_wc], 60,
                       amplitudes=[1.0, 0.001],
                       shapes_sin=[shape_ss, shape_ws])

    res_trunc = dmd(data, rank=2, mean_centre=False, dt=DT)
    f_trunc = mode_frequencies(res_trunc["eigenvalues"], DT)

    check("  truncation keeps the strong mode", float(f_trunc[0]), f_strong, 1e-4)
    report("  requested rank is returned", res_trunc["rank"] == 2,
           f": got {res_trunc['rank']}")

    # At full rank the strong mode must still carry the larger energy
    res_full = dmd(data, rank=4, mean_centre=False, dt=DT)
    f_full = mode_frequencies(res_full["eigenvalues"], DT)
    e_full = mode_energies(res_full["modes"], res_full["amplitudes"], res_full["eigenvalues"], 60)
    i_strong = int(np.argmin(np.abs(f_full - f_strong)))
    i_weak = int(np.argmin(np.abs(f_full - f_weak)))

    report("  strong mode carries more energy",
           e_full[i_strong] > e_full[i_weak],
           f": {e_full[i_strong]:.4e} > {e_full[i_weak]:.4e}")
    print()


def test_e_mean_centring():
    """A constant background is a stationary mode; centring removes it."""
    print("E) Mean-centring")
    f_osc, g_osc = 1.5, -0.4
    shape_c = np.array([1.0, -0.5, 0.3, 0.0, 0.6])
    shape_s = np.array([0.2, 0.9, -0.4, 0.7, 0.0])

    osc = make_signal([f_osc], [g_osc], [shape_c], 50, shapes_sin=[shape_s])
    # A large stationary offset, as in the average spectral shape of a pass
    background = np.array([50.0, 48.0, 52.0, 49.0, 51.0]).reshape(-1, 1)
    data = osc + background

    # Without centring, one mode is spent on the constant: f = 0 and g = 0
    res_raw = dmd(data, rank=3, mean_centre=False, dt=DT)
    f_raw = mode_frequencies(res_raw["eigenvalues"], DT)
    g_raw = mode_growth_rates(res_raw["eigenvalues"], DT)
    i_stat = int(np.argmin(np.abs(f_raw) + np.abs(g_raw)))

    print(f"       uncentred stationary mode: f = {f_raw[i_stat]:.6f} Hz, "
          f"g = {g_raw[i_stat]:.6f} 1/s")
    report("  uncentred data produces a stationary mode",
           abs(f_raw[i_stat]) < 1e-6 and abs(g_raw[i_stat]) < 1e-6)

    # With centring, the planted oscillation is recovered instead.
    # The subtracted time-mean only estimates the background; a decaying
    # oscillation has a small nonzero mean over a finite window, so the
    # tolerance here is looser than elsewhere.
    res_centred = dmd(data, rank=2, mean_centre=True, dt=DT)
    f_c = mode_frequencies(res_centred["eigenvalues"], DT)
    g_c = mode_growth_rates(res_centred["eigenvalues"], DT)

    check("  centred: planted frequency", float(f_c[0]), f_osc, 5e-3)
    check("  centred: planted growth rate", float(g_c[0]), g_osc, 1e-2)
    print()


def test_f_helpers():
    """Snapshot pairing and rank selection must behave as documented."""
    print("F) Helper functions")

    # The two matrices must be the same sequence shifted by one column
    d = np.arange(12, dtype=float).reshape(3, 4)
    p, p1 = build_snapshot_pairs(d)
    report("  snapshot pairs are shifted by one column",
           p.shape == (3, 3) and p1.shape == (3, 3)
           and np.allclose(p[:, 1:], p1[:, :-1]))

    # A higher energy threshold can never keep fewer modes
    sig = np.array([10.0, 5.0, 1.0, 0.01, 0.0])
    r90 = choose_rank(sig, energy_threshold=0.90)
    r99 = choose_rank(sig, energy_threshold=0.999)
    report("  energy threshold is monotonic", r90 <= r99,
           f": {r90} <= {r99}")

    # Asking for more modes than the data supports must be capped
    r_capped = choose_rank(sig, rank=99)
    report("  requested rank is capped at the numerical rank", r_capped <= 4,
           f": got {r_capped}")
    print()


def main():
    # Run all six test groups in order
    test_a_single_oscillation()
    test_b_two_modes()
    test_c_pure_growth()
    test_d_rank_truncation()
    test_e_mean_centring()
    test_f_helpers()
    print("All DMD verification tests passed.")


if __name__ == "__main__":
    main()