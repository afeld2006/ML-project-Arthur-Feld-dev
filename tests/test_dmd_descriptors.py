"""
Verification of the three per-mode descriptors f_i, g_i and a_i.

test_dmd.py checks that DMD recovers planted eigenvalues. This file checks the
descriptors computed from them, and in particular that the energy descriptor
accounts for how a mode evolves in time rather than for its starting size only.

Four tests:
  A) Descriptor formulas  - values computed from eigenvalues built by hand.
  B) Time weighting       - a growing mode must outrank a decaying one of the
                            same initial amplitude. This is what proves a_i
                            uses the whole window and not just snapshot 1.
  C) Conjugate pairs      - both members of a pair describe one oscillation,
                            so their descriptors must agree.
  D) Scale invariance     - rescaling the data must leave f_i and g_i
                            unchanged, and scale a_i by the same factor.

Run from the project root:
    py tests/test_dmd_descriptors.py
"""

import sys
from pathlib import Path

import numpy as np

# Make src/ importable regardless of where we run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dmd import dmd, mode_frequencies, mode_growth_rates, mode_energies

DT = 0.1        # snapshot spacing in seconds; matches the real spectrograms
N_SNAP = 40     # snapshots in the synthetic signals used here
TOL = 1e-9      # descriptors are closed-form, so the tolerance is tight


def check(name, got, expected, tol=TOL):
    # Compare within a tolerance to allow for floating-point rounding
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got {got:.9f}, expected {expected:.9f}")
    if not ok:
        raise AssertionError(f"{name} failed (tolerance {tol})")


def report(name, ok, detail=""):
    # For checks whose result is a condition rather than a number
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{detail}")
    if not ok:
        raise AssertionError(f"{name} failed")


def make_two_mode_signal(f1, g1, f2, g2, n_states=6, seed=0):
    """Build a signal from two oscillations of equal initial amplitude.

    Each oscillation needs two independent spatial directions to carry its
    complex-conjugate pair, so four random shapes are drawn in total.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_SNAP) * DT
    data = np.zeros((n_states, N_SNAP))

    for f, g in [(f1, g1), (f2, g2)]:
        shape_cos = rng.standard_normal(n_states)
        shape_sin = rng.standard_normal(n_states)
        envelope = np.exp(g * t)          # the only difference between them
        data += np.outer(shape_cos, envelope * np.cos(2 * np.pi * f * t))
        data += np.outer(shape_sin, envelope * np.sin(2 * np.pi * f * t))

    return data


def test_a_descriptor_formulas():
    """f_i and g_i must invert the eigenvalue exactly."""
    print("A) Descriptor formulas")

    # A mode with growth g and frequency f becomes the discrete eigenvalue
    # exp((g + 2i pi f) dt); the descriptors must return f and g from it
    for f_true, g_true in [(1.0, -0.5), (2.5, 0.3), (0.25, -2.0)]:
        lam = np.array([np.exp((g_true + 2.0j * np.pi * f_true) * DT)])
        check(f"  frequency f = {f_true}",
              float(mode_frequencies(lam, DT)[0]), f_true)
        check(f"  growth rate g = {g_true}",
              float(mode_growth_rates(lam, DT)[0]), g_true)

    # A real positive eigenvalue has no imaginary part, so no oscillation
    lam_real = np.array([np.exp(0.7 * DT)])
    check("  real eigenvalue gives zero frequency",
          float(mode_frequencies(lam_real, DT)[0]), 0.0, 1e-12)
    print()


def test_b_time_weighting():
    """A growing mode must outrank a decaying one of equal initial size."""
    print("B) Time weighting")
    f_grow, g_grow = 1.0, 0.5      # gains strength across the window
    f_decay, g_decay = 2.0, -1.2   # fades across the window

    data = make_two_mode_signal(f_grow, g_grow, f_decay, g_decay)
    res = dmd(data, rank=4, mean_centre=False, dt=DT)

    f_all = mode_frequencies(res["eigenvalues"], DT)
    a_all = mode_energies(res["modes"], res["amplitudes"],
                          res["eigenvalues"], N_SNAP)

    i_grow = int(np.argmin(np.abs(f_all - f_grow)))
    i_decay = int(np.argmin(np.abs(f_all - f_decay)))

    report("  growing mode carries more integrated energy",
           a_all[i_grow] > a_all[i_decay],
           f": {a_all[i_grow]:.4e} > {a_all[i_decay]:.4e}")

    # modes are returned in rank order, so the growing one must come first
    f_first = float(f_all[0])
    report("  growing mode is ranked first",
           abs(f_first - f_grow) < abs(f_first - f_decay),
           f": first mode at {f_first:.4f} Hz, growing mode at {f_grow} Hz")
    print()


def test_c_conjugate_pairs():
    """Both members of a conjugate pair describe the same oscillation."""
    print("C) Conjugate pairs")
    data = make_two_mode_signal(1.0, -0.3, 2.0, -0.8)
    res = dmd(data, rank=4, mean_centre=False, dt=DT)

    f_all = np.sort(mode_frequencies(res["eigenvalues"], DT))

    # Four modes carrying two oscillations must give two equal pairs
    pairs_equal = all(
        abs(f_all[i] - f_all[i + 1]) < 1e-6 for i in range(0, 4, 2)
    )
    report("  frequencies occur in equal pairs", pairs_equal)

    n_distinct = len(np.unique(np.round(f_all, 6)))
    report("  four modes reduce to two distinct frequencies",
           n_distinct == 2, f": found {n_distinct}")
    print()


def test_d_scale_invariance():
    """Rescaling the data must not move f_i or g_i, and must scale a_i."""
    print("D) Scale invariance")
    factor = 7.0
    data = make_two_mode_signal(1.0, -0.3, 2.0, -0.8)

    res = dmd(data, rank=4, mean_centre=False, dt=DT)
    res_scaled = dmd(data * factor, rank=4, mean_centre=False, dt=DT)

    f_ref = np.sort(mode_frequencies(res["eigenvalues"], DT))
    g_ref = np.sort(mode_growth_rates(res["eigenvalues"], DT))
    f_new = np.sort(mode_frequencies(res_scaled["eigenvalues"], DT))
    g_new = np.sort(mode_growth_rates(res_scaled["eigenvalues"], DT))

    # The dynamics do not depend on how loud the signal is
    report("  frequencies unchanged", np.allclose(f_ref, f_new, atol=1e-8))
    report("  growth rates unchanged", np.allclose(g_ref, g_new, atol=1e-8))

    # Energy is a size, so it must follow the data by the same factor
    a_ref = mode_energies(res["modes"], res["amplitudes"],
                          res["eigenvalues"], N_SNAP)
    a_new = mode_energies(res_scaled["modes"], res_scaled["amplitudes"],
                          res_scaled["eigenvalues"], N_SNAP)

    check("  total energy scales with the data",
          float(np.sum(a_new) / np.sum(a_ref)), factor, 1e-6)
    print()


def main():
    # Run all four test groups in order
    test_a_descriptor_formulas()
    test_b_time_weighting()
    test_c_conjugate_pairs()
    test_d_scale_invariance()
    print("All descriptor tests passed.")


if __name__ == "__main__":
    main()