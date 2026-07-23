"""Verification of the DMD implementation on synthetic data with known answers.

Same discipline as tests/test_sinusoid_recovery.py : we build signals whose
modes we know exactly, then check DMD recovers them before trusting it on the
77 real spectrograms.

Groups :
  A : single decaying oscillation ; known frequency and known decay rate
  B : two superposed modes ; both must be recovered simultaneously
  C : pure growth with no oscillation ; frequency must come out at 0
  D : rank truncation keeps the dominant mode and discards the weak one
  E : mean-centring removes the constant background mode
"""

import sys
import os
import numpy as np

# allow importing src/ when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dmd import (
    dmd,
    mode_frequencies,
    mode_growth_rates,
    mode_energies,
    build_snapshot_pairs,
    choose_rank,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []


def check(label, got, expected, tol):
    """Report one comparison ; record failures for the final exit status."""
    ok = abs(got - expected) <= tol
    tag = PASS if ok else FAIL
    print(f"{tag} {label}: got {got:.6f}, expected {expected:.6f}")
    if not ok:
        failures.append(label)


def make_signal(freqs, growths, shapes, n_snapshots, dt, amplitudes=None,
                shapes_sin=None):
    """Build a real snapshot matrix from known modes.

    A real signal carrying an oscillation at frequency f is produced by a
    complex-conjugate pair of DMD modes ; representing that pair needs TWO
    independent spatial directions. So each oscillatory mode is built as
        amp * exp(g t) * [shape_cos * cos(2 pi f t) + shape_sin * sin(2 pi f t)]
    with shape_cos and shape_sin linearly independent. Building it from a
    single shape vector instead would make the snapshot matrix rank 1, and no
    algorithm could then recover a conjugate pair from it.

    A mode with f = 0 is a pure growth/decay term and needs only one direction.
    """
    t = np.arange(n_snapshots) * dt
    n_states = len(shapes[0])
    data = np.zeros((n_states, n_snapshots), dtype=float)
    if amplitudes is None:
        amplitudes = [1.0] * len(freqs)
    if shapes_sin is None:
        shapes_sin = [None] * len(freqs)

    for f, g, shape_c, shape_s, amp in zip(
        freqs, growths, shapes, shapes_sin, amplitudes
    ):
        shape_c = np.asarray(shape_c, dtype=float).reshape(-1, 1)
        envelope = np.exp(g * t).reshape(1, -1)
        if f == 0.0:
            # pure growth or decay ; one spatial direction is enough
            data += amp * (shape_c @ envelope)
            continue
        if shape_s is None:
            raise ValueError("an oscillatory mode needs a second shape vector")
        shape_s = np.asarray(shape_s, dtype=float).reshape(-1, 1)
        cos_t = np.cos(2.0 * np.pi * f * t).reshape(1, -1)
        sin_t = np.sin(2.0 * np.pi * f * t).reshape(1, -1)
        data += amp * (shape_c @ (envelope * cos_t) + shape_s @ (envelope * sin_t))
    return data


print("A) Single decaying oscillation ; known frequency and decay rate")
dt = 0.1
f_true = 1.25          # Hz ; well below Nyquist (1 / (2 dt) = 5 Hz)
g_true = -0.8          # 1/s ; decaying
shape_c = np.array([1.0, 0.5, -0.3, 0.2, 0.9])
shape_s = np.array([0.2, -0.7, 0.4, 1.0, -0.1])
data = make_signal([f_true], [g_true], [shape_c], n_snapshots=40, dt=dt,
                   shapes_sin=[shape_s])

res = dmd(data, rank=2, mean_centre=False, dt=dt)
freqs = mode_frequencies(res["eigenvalues"], dt)
growths = mode_growth_rates(res["eigenvalues"], dt)
# a real signal built from one complex mode appears as a conjugate pair ;
# both members carry the same frequency and growth rate
check("single mode frequency", float(freqs[0]), f_true, 1e-6)
check("single mode growth rate", float(growths[0]), g_true, 1e-6)
print()


print("B) Two superposed modes ; both recovered simultaneously")
f1, g1 = 0.5, -0.3
f2, g2 = 2.0, -1.5
shape1 = np.array([1.0, 0.0, 0.0, 0.5, 0.2])
shape2 = np.array([0.0, 1.0, 0.4, 0.0, -0.6])
# each oscillation needs its own independent sine-side shape ; four
# directions total, so the data matrix must have at least rank 4
shape1s = np.array([0.3, 0.8, -0.2, 0.0, 1.0])
shape2s = np.array([-0.5, 0.1, 1.0, 0.7, 0.0])
data = make_signal([f1, f2], [g1, g2], [shape1, shape2], n_snapshots=60, dt=dt,
                   shapes_sin=[shape1s, shape2s])

res = dmd(data, rank=4, mean_centre=False, dt=dt)
freqs = np.sort(np.unique(np.round(mode_frequencies(res["eigenvalues"], dt), 6)))
# expect exactly the two planted frequencies
check("two-mode : lower frequency", float(freqs[0]), min(f1, f2), 1e-5)
check("two-mode : upper frequency", float(freqs[-1]), max(f1, f2), 1e-5)

# match growth rates by pairing each recovered frequency to its planted mode
all_f = mode_frequencies(res["eigenvalues"], dt)
all_g = mode_growth_rates(res["eigenvalues"], dt)
idx1 = int(np.argmin(np.abs(all_f - f1)))
idx2 = int(np.argmin(np.abs(all_f - f2)))
check("two-mode : growth of mode 1", float(all_g[idx1]), g1, 1e-5)
check("two-mode : growth of mode 2", float(all_g[idx2]), g2, 1e-5)
print()


print("C) Pure growth, no oscillation ; frequency must be zero")
g_pure = 0.4
shape = np.array([0.7, -0.2, 0.5, 1.0, 0.1])
data = make_signal([0.0], [g_pure], [shape], n_snapshots=30, dt=dt)

res = dmd(data, rank=1, mean_centre=False, dt=dt)
freqs = mode_frequencies(res["eigenvalues"], dt)
growths = mode_growth_rates(res["eigenvalues"], dt)
check("pure growth : frequency is zero", float(freqs[0]), 0.0, 1e-9)
check("pure growth : growth rate", float(growths[0]), g_pure, 1e-6)
print()


print("D) Rank truncation keeps the dominant mode")
f_strong, g_strong = 1.0, -0.2
f_weak, g_weak = 3.0, -0.5
shape_s = np.array([1.0, 0.2, 0.0, 0.3, 0.1])
shape_w = np.array([0.0, 1.0, 0.5, 0.0, -0.2])
shape_s_sin = np.array([0.4, -0.6, 0.9, 0.0, 0.2])
shape_w_sin = np.array([0.1, 0.0, -0.3, 1.0, 0.5])
# the weak mode is planted at 1/1000 of the strong one
data = make_signal(
    [f_strong, f_weak], [g_strong, g_weak], [shape_s, shape_w],
    n_snapshots=60, dt=dt, amplitudes=[1.0, 0.001],
    shapes_sin=[shape_s_sin, shape_w_sin],
)

res_full = dmd(data, rank=4, mean_centre=False, dt=dt)
res_trunc = dmd(data, rank=2, mean_centre=False, dt=dt)
f_trunc = mode_frequencies(res_trunc["eigenvalues"], dt)
# with only 2 modes kept we should retain the strong one, not the weak one
check("truncated rank keeps strong mode", float(f_trunc[0]), f_strong, 1e-4)
print(f"{PASS if res_trunc['rank'] == 2 else FAIL} truncation returns rank 2: "
      f"got {res_trunc['rank']}")
if res_trunc["rank"] != 2:
    failures.append("truncation rank")

# energies must rank the strong mode above the weak one in the full run
e_full = mode_energies(res_full["modes"], res_full["amplitudes"])
f_full = mode_frequencies(res_full["eigenvalues"], dt)
i_strong = int(np.argmin(np.abs(f_full - f_strong)))
i_weak = int(np.argmin(np.abs(f_full - f_weak)))
ok = e_full[i_strong] > e_full[i_weak]
print(f"{PASS if ok else FAIL} strong mode has larger energy: "
      f"{e_full[i_strong]:.4e} > {e_full[i_weak]:.4e}")
if not ok:
    failures.append("energy ordering")
print()


print("E) Mean-centring removes the constant background")
f_osc, g_osc = 1.5, -0.4
shape_osc = np.array([1.0, -0.5, 0.3, 0.0, 0.6])
shape_osc_sin = np.array([0.2, 0.9, -0.4, 0.7, 0.0])
background = np.array([50.0, 48.0, 52.0, 49.0, 51.0]).reshape(-1, 1)
osc = make_signal([f_osc], [g_osc], [shape_osc], n_snapshots=50, dt=dt,
                  shapes_sin=[shape_osc_sin])
# a large stationary offset, like the average spectral shape of a pass
data = osc + background

res_raw = dmd(data, rank=3, mean_centre=False, dt=dt)
res_centred = dmd(data, rank=2, mean_centre=True, dt=dt)

f_raw = mode_frequencies(res_raw["eigenvalues"], dt)
g_raw = mode_growth_rates(res_raw["eigenvalues"], dt)
# uncentred : the dominant mode should be the stationary background,
# i.e. near-zero frequency and near-zero growth
i_stat = int(np.argmin(np.abs(f_raw) + np.abs(g_raw)))
print(f"       uncentred dominant-stationary mode: f={f_raw[i_stat]:.6f} Hz, "
      f"g={g_raw[i_stat]:.6f} 1/s")
ok = abs(f_raw[i_stat]) < 1e-6 and abs(g_raw[i_stat]) < 1e-6
print(f"{PASS if ok else FAIL} uncentred data produces a stationary mode")
if not ok:
    failures.append("stationary mode present")

# centred : the planted oscillation should be recovered cleanly
f_c = mode_frequencies(res_centred["eigenvalues"], dt)
g_c = mode_growth_rates(res_centred["eigenvalues"], dt)
# the subtracted time-mean is only an estimate of the true background ; a
# decaying oscillation has a small nonzero mean over a finite window, so
# centring introduces a slight bias. Tolerances reflect that, and this is the
# known caveat of mean-subtracted DMD worth stating to the supervisor.
check("centred : recovers planted frequency", float(f_c[0]), f_osc, 5e-3)
check("centred : recovers planted growth", float(g_c[0]), g_osc, 1e-2)
print()


print("F) Helper functions behave as expected")
d = np.arange(12, dtype=float).reshape(3, 4)
p, p1 = build_snapshot_pairs(d)
ok = p.shape == (3, 3) and p1.shape == (3, 3) and np.allclose(p[:, 1:], p1[:, :-1])
print(f"{PASS if ok else FAIL} snapshot pairs are correctly shifted")
if not ok:
    failures.append("snapshot pairs")

sig = np.array([10.0, 5.0, 1.0, 0.01, 0.0])
r90 = choose_rank(sig, energy_threshold=0.90)
r99 = choose_rank(sig, energy_threshold=0.999)
ok = r90 <= r99
print(f"{PASS if ok else FAIL} higher energy threshold keeps at least as many "
      f"modes: {r90} <= {r99}")
if not ok:
    failures.append("energy threshold monotonic")

r_capped = choose_rank(sig, rank=99)
ok = r_capped <= 4
print(f"{PASS if ok else FAIL} requested rank capped at numerical rank: "
      f"got {r_capped}")
if not ok:
    failures.append("rank capping")
print()


if failures:
    print(f"FAILURES: {len(failures)} -> {failures}")
    sys.exit(1)
print("All DMD verification tests passed.")