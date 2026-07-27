"""
Verification of the DMD feature extraction rules.

test_dmd.py checks the decomposition and test_dmd_descriptors.py checks the
three descriptors. This file checks the step between them and the feature
table : which modes are retained, and whether every signal yields a vector of
the same length.

Five tests:
  A) Pair removal      - conjugate pairs collapse to one representative.
  B) Nyquist removal   - modes at the sampling limit are discarded and the
                         next mode by energy takes the place of each one.
  C) Fixed length      - the feature vector has 3r entries whatever the input.
  D) Ordering          - modes are numbered by descending energy, and the
                         planted oscillations appear first.
  E) Column names      - names match the count and follow the documented order.

Run from the project root:
    py tests/test_dmd_features.py
"""

import sys
from pathlib import Path

import numpy as np

# Make src/ importable regardless of where we run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dmd_features import (
    NYQUIST_TOL_HZ,
    deduplicate_pairs,
    descriptor_column_names,
    drop_nyquist_modes,
    extract_dmd_descriptors,
)

DT = 0.1                      # snapshot spacing in seconds
NYQUIST = 1.0 / (2.0 * DT)    # the highest frequency this grid can represent
N_SNAP = 48                   # snapshots per signal, as in the real files


def check(name, got, expected, tol=1e-6):
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


def make_spectrogram(components, n_states=120, n_snapshots=N_SNAP, seed=0,
                     background=40.0, noise=0.0):
    """Build a spectrogram-shaped matrix from oscillations of known properties.

    Each entry of components is (frequency, growth rate, amplitude). Every
    oscillation is given two independent spatial shapes, since a real
    oscillation is carried by a complex-conjugate pair and a pair needs two
    dimensions of the state space.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_snapshots) * DT
    bins = np.arange(n_states)

    # a large positive background, as in a measured power spectrum
    data = np.full((n_states, n_snapshots), background, dtype=float)

    for k, (freq, growth, amplitude) in enumerate(components):
        # smooth spectral shapes, so the modes resemble real spectra
        centre = (k + 1) * n_states / (len(components) + 1)
        shape_cos = np.exp(-0.5 * ((bins - centre) / (n_states / 8)) ** 2)
        shape_sin = np.exp(-0.5 * ((bins - centre * 1.1) / (n_states / 8)) ** 2)

        envelope = np.exp(growth * t)
        data += amplitude * (
            np.outer(shape_cos, envelope * np.cos(2 * np.pi * freq * t))
            + np.outer(shape_sin, envelope * np.sin(2 * np.pi * freq * t))
        )

    if noise > 0:
        data += noise * rng.standard_normal((n_states, n_snapshots))

    return data


def test_a_pair_removal():
    """Two modes with identical descriptors must collapse to one."""
    print("A) Conjugate pair removal")

    # a pair appears as two entries carrying the same frequency and growth
    freqs = np.array([1.5, 1.5, 0.4, 0.4, 2.2])
    growths = np.array([-0.3, -0.3, -1.1, -1.1, -0.7])
    energies = np.array([10.0, 10.0, 8.0, 8.0, 3.0])

    f_out, g_out, a_out = deduplicate_pairs(freqs, growths, energies)

    report("  five modes reduce to three", len(f_out) == 3,
           f": got {len(f_out)}")
    report("  distinct frequencies are preserved",
           np.allclose(np.sort(f_out), [0.4, 1.5, 2.2]))
    report("  energies follow their own mode",
           np.allclose(np.sort(a_out), [3.0, 8.0, 10.0]))

    # a mode with no partner must survive untouched
    single_f = np.array([0.9])
    single_g = np.array([-0.5])
    single_a = np.array([4.0])
    f_s, _, _ = deduplicate_pairs(single_f, single_g, single_a)
    report("  an unpaired mode is kept", len(f_s) == 1)
    print()


def test_b_nyquist_removal():
    """Modes reported at the sampling limit must be discarded."""
    print("B) Nyquist mode removal")

    freqs = np.array([0.5, NYQUIST, 1.2, NYQUIST, 2.0])
    growths = np.array([-0.4, -30.0, -0.8, -47.0, -1.5])
    energies = np.array([9.0, 2.0, 7.0, 1.0, 5.0])

    f_out, g_out, a_out, n_dropped = drop_nyquist_modes(
        freqs, growths, energies, DT
    )

    report("  two modes at the limit are removed", n_dropped == 2,
           f": got {n_dropped}")
    report("  three modes survive", len(f_out) == 3, f": got {len(f_out)}")
    report("  no survivor sits at the limit",
           not np.any(np.abs(f_out - NYQUIST) < NYQUIST_TOL_HZ))

    # the extreme growth rates belonged to the discarded modes
    report("  extreme growth rates are gone", np.min(g_out) > -10.0,
           f": minimum now {np.min(g_out):.2f} 1/s")

    # a frequency close to but not at the limit must be kept
    near = np.array([NYQUIST - 0.01])
    f_near, _, _, n_near = drop_nyquist_modes(
        near, np.array([-1.0]), np.array([1.0]), DT
    )
    report("  a frequency near the limit is kept",
           len(f_near) == 1 and n_near == 0)
    print()


def test_c_fixed_length():
    """Every signal must yield the same number of modes."""
    print("C) Fixed feature length")

    # three signals differing in content, noise and seed
    cases = [
        ([(0.4, -0.2, 5.0), (1.1, -0.6, 3.0)], 0.0, 0),
        ([(0.3, -0.1, 6.0), (0.9, -0.5, 2.0), (2.0, -1.2, 1.0)], 0.05, 1),
        ([(0.6, -0.3, 4.0)], 0.10, 2),
    ]

    for n_modes in (2, 3, 4):
        counts = []
        for components, noise, seed in cases:
            spec = make_spectrogram(components, seed=seed, noise=noise)
            out = extract_dmd_descriptors(spec, n_modes=n_modes, dt=DT)
            counts.append(len(out["frequencies"]))

            # all three descriptor arrays must agree in length
            report(f"  r = {n_modes}: descriptors agree in length",
                   len(out["frequencies"]) == len(out["growth_rates"])
                   == len(out["energies"]))

        report(f"  r = {n_modes}: every signal returns {n_modes} modes",
               all(c == n_modes for c in counts), f": got {counts}")
    print()


def test_d_ordering():
    """Modes must be numbered by descending energy."""
    print("D) Ordering by energy")

    # the first component is planted far stronger than the second
    components = [(0.4, -0.2, 10.0), (1.8, -0.9, 0.5)]
    spec = make_spectrogram(components, noise=0.02, seed=3)

    out = extract_dmd_descriptors(spec, n_modes=4, dt=DT)
    energies = out["energies"]
    freqs = out["frequencies"]

    # energies must be non-increasing across the retained modes
    report("  energies are in descending order",
           np.all(np.diff(energies) <= 1e-9),
           f": {np.round(energies, 2)}")

    # the strongest planted oscillation must be the first mode
    check("  strongest mode is the planted 0.4 Hz", float(freqs[0]), 0.4, 5e-3)

    # the weaker one must still be present, further down the list
    distance = np.abs(freqs - 1.8)
    report("  the weaker planted mode is retained", np.min(distance) < 5e-3,
           f": closest frequency {freqs[int(np.argmin(distance))]:.4f} Hz")
    print()


def test_e_column_names():
    """Column names must match the mode count and the documented order."""
    print("E) Column names")

    for n_modes in (1, 2, 4, 6):
        names = descriptor_column_names(n_modes)
        report(f"  r = {n_modes} gives {3 * n_modes} names",
               len(names) == 3 * n_modes, f": got {len(names)}")

    names = descriptor_column_names(2)
    expected = ["dmd_f_1_hz", "dmd_g_1_per_s", "dmd_a_1",
                "dmd_f_2_hz", "dmd_g_2_per_s", "dmd_a_2"]
    report("  names follow frequency, growth, energy per mode",
           names == expected, f": got {names}")

    # names must be unique, or the feature table would silently lose columns
    names_six = descriptor_column_names(6)
    report("  names are unique", len(set(names_six)) == len(names_six))
    print()


def main():
    # Run all five test groups in order
    test_a_pair_removal()
    test_b_nyquist_removal()
    test_c_fixed_length()
    test_d_ordering()
    test_e_column_names()
    print("All DMD feature extraction tests passed.")


if __name__ == "__main__":
    main()