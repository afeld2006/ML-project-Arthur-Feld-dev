"""Dynamic Mode Decomposition (DMD) on a snapshot sequence.

Follows Algorithm 1 : the operator A is never formed at full size ; it is
projected onto a small basis built from the SVD of the first snapshot matrix.

Note on step 1 and 2 : Algorithm 1 obtains the basis from the Gram product
(Psi_p)^T Psi_p. That product squares the condition number, so the small
singular values lose accuracy ; measured on synthetic data, values that should
be near 1e-16 came out near 1e-8. This module takes the thin SVD of Psi_p
instead, which gives the same U, Sigma and V at full precision. The low-storage
motivation is kept : the thin SVD of an n x m matrix never forms anything
larger than n x m, so the n x n operator is still never built.

Terminology used throughout:
  snapshot     : one column of the data matrix ; one time frame of a spectrogram
  mode phi_i   : a characteristic spatial shape ; here a spectral profile
  eigenvalue lambda_i : how mode i advances from one snapshot to the next
  amplitude b_i : how strongly mode i is present in the first snapshot
"""

from __future__ import annotations

import numpy as np


def build_snapshot_pairs(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a snapshot matrix into the two shifted matrices Psi_p and Psi_p1.

    DMD relates each snapshot to the next one, so the sequence is paired
    against itself shifted by one column.

    Parameters
    ----------
    data : ndarray, shape (n, m)
        n rows are state components (frequency bins) ; m columns are snapshots.

    Returns
    -------
    psi_p, psi_p1 : ndarray, each shape (n, m-1)
        psi_p holds snapshots 1 to m-1 ; psi_p1 holds snapshots 2 to m.
    """
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_states, n_snapshots)")
    if data.shape[1] < 2:
        raise ValueError("need at least 2 snapshots to form a pair")

    return data[:, :-1], data[:, 1:]


def _thin_svd(psi_p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return U, sigma, V of the thin SVD of Psi_p, dominant direction first.

    See the module docstring for why this replaces the Gram product used in
    steps 1 and 2 of Algorithm 1.
    """
    # full_matrices=False keeps U at n x m rather than n x n
    U, sigma, Vh = np.linalg.svd(psi_p, full_matrices=False)
    # numpy already orders singular values descending
    return U, sigma, Vh.conj().T


def _integrated_energies(
    modes: np.ndarray,
    amplitudes: np.ndarray,
    eigenvalues: np.ndarray,
    n_snapshots: int,
) -> np.ndarray:
    """Energy of each mode summed over the whole snapshot window.

    Mode i contributes b_i * phi_i * lambda_i^p at snapshot p ; summing the
    magnitude over every p accounts for growth and decay. Ranking on the
    initial amplitude alone would place a mode that starts weak and grows
    below one that starts strong and dies immediately.

    a_i = |b_i| * ||phi_i|| * sum_p |lambda_i|^p
    """
    modes = np.asarray(modes)
    amplitudes = np.asarray(amplitudes)
    eigenvalues = np.asarray(eigenvalues)

    # eigenvector scale is arbitrary on its own ; it is always paired with the
    # amplitude, which is what ties the mode to this particular signal
    mode_norms = np.linalg.norm(modes, axis=0)

    # time factor : how much the mode grows or decays across the window
    powers = np.arange(n_snapshots)
    magnitudes = np.abs(eigenvalues).reshape(-1, 1) ** powers.reshape(1, -1)
    time_factor = magnitudes.sum(axis=1)

    return np.abs(amplitudes) * mode_norms * time_factor


def choose_rank(
    sigma: np.ndarray,
    rank: int | None = None,
    energy_threshold: float | None = None,
) -> int:
    """Decide how many modes to keep.

    Give either rank or energy_threshold, not both. With neither, every
    numerically meaningful direction is kept.

    Note : an energy threshold returns a different count for each signal, so
    it is a diagnostic tool for choosing r rather than a way to run the
    extraction ; the model needs the same number of features for every signal.
    """
    if rank is not None and energy_threshold is not None:
        raise ValueError("give either rank or energy_threshold, not both")

    # singular values below this are numerical noise, not information
    tol = np.finfo(float).eps * max(sigma.shape) * (sigma[0] if sigma.size else 0.0)
    n_nonzero = max(int(np.sum(sigma > tol)), 1)

    if rank is not None:
        # cannot keep more modes than the data supports
        return int(min(rank, n_nonzero))

    if energy_threshold is not None:
        if not 0.0 < energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1]")
        # energy is carried by the squared singular values
        energy = sigma ** 2
        total = energy.sum()
        if total <= 0:
            return 1
        cumulative = np.cumsum(energy) / total
        # first index at which the requested fraction is reached
        r = int(np.searchsorted(cumulative, energy_threshold) + 1)
        return int(min(max(r, 1), n_nonzero))

    return n_nonzero


def dmd(
    data: np.ndarray,
    rank: int | None = None,
    energy_threshold: float | None = None,
    mean_centre: bool = True,
    dt: float = 1.0,
) -> dict:
    """Decompose a snapshot matrix into modes, eigenvalues and amplitudes.

    Parameters
    ----------
    data : ndarray, shape (n, m)
        Snapshot matrix ; columns are successive time frames.
    rank : int or None
        Number of modes to keep ; mutually exclusive with energy_threshold.
    energy_threshold : float or None
        Keep the fewest modes capturing this fraction of the singular energy.
    mean_centre : bool
        Subtract the time-mean snapshot first. A spectrogram sits on a large
        stationary background ; without centring the dominant mode is spent
        representing that constant, giving frequency 0 and growth rate 0.
    dt : float
        Snapshot spacing in seconds ; carried through for the descriptors.

    Returns
    -------
    dict with keys
        modes           : (n, r) complex ; the DMD modes phi_i, ranked
        eigenvalues     : (r,) complex ; the eigenvalues lambda_i, same order
        amplitudes      : (r,) complex ; b_i, same order
        singular_values : (m-1,) real ; the full spectrum, for rank diagnostics
        rank            : int ; how many modes were kept
        mean_snapshot   : (n,) or None ; what was subtracted, if anything
        dt              : float ; the snapshot spacing that was supplied
    """
    data = np.asarray(data, dtype=float)

    mean_snapshot = None
    if mean_centre:
        mean_snapshot = data.mean(axis=1, keepdims=True)
        data = data - mean_snapshot

    psi_p, psi_p1 = build_snapshot_pairs(data)

    # steps 1 to 3 : the SVD of Psi_p supplies the reduced basis U
    U_full, sigma, V_full = _thin_svd(psi_p)

    r = choose_rank(sigma, rank=rank, energy_threshold=energy_threshold)

    U = U_full[:, :r]
    V_r = V_full[:, :r]
    sigma_r = sigma[:r]

    # guard against dividing by a vanishing singular value
    sigma_inv = np.zeros_like(sigma_r)
    nonzero = sigma_r > 0
    sigma_inv[nonzero] = 1.0 / sigma_r[nonzero]

    # step 4 : project the operator onto the basis ; A_tilde is only r x r
    A_tilde = U.conj().T @ psi_p1 @ V_r * sigma_inv

    # step 5 : eigendecomposition of the small projected operator
    lambdas, Y = np.linalg.eig(A_tilde)

    # step 6 : lift the eigenvectors back to the full state space
    modes = U @ Y

    # amplitudes : least-squares fit of the first snapshot onto the modes ;
    # this is what ties an arbitrarily scaled eigenvector to this signal
    b = np.linalg.lstsq(modes, data[:, 0], rcond=None)[0]

    # rank by energy integrated over the window, not by initial amplitude
    energies = _integrated_energies(modes, b, lambdas, data.shape[1])
    order = np.argsort(energies)[::-1]

    return {
        "modes": modes[:, order],
        "eigenvalues": lambdas[order],
        "amplitudes": b[order],
        "singular_values": sigma,
        "rank": r,
        "mean_snapshot": None if mean_snapshot is None else mean_snapshot.ravel(),
        "dt": dt,
    }


def mode_frequencies(eigenvalues: np.ndarray, dt: float) -> np.ndarray:
    """Oscillation frequency in Hz : f_i = |Im(log lambda_i)| / (2 pi dt).

    Writing lambda_i in polar form, log lambda_i = ln|lambda_i| + i theta_i ;
    the imaginary part theta_i is the phase advanced per snapshot, so dividing
    by 2 pi dt converts it to cycles per second.

    The absolute value collapses each complex-conjugate pair to one positive
    frequency. A real positive eigenvalue gives 0, meaning no oscillation.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(eigenvalues)
    return np.abs(np.imag(logs)) / (2.0 * np.pi * dt)


def mode_growth_rates(eigenvalues: np.ndarray, dt: float) -> np.ndarray:
    """Growth or decay rate in 1/s : g_i = Re(log lambda_i) / dt.

    The real part ln|lambda_i| is the change in magnitude per snapshot.
    Positive means the mode strengthens across the pass ; negative means it
    fades ; zero means it holds a constant level.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(eigenvalues)
    return np.real(logs) / dt


def mode_energies(
    modes: np.ndarray,
    amplitudes: np.ndarray,
    eigenvalues: np.ndarray,
    n_snapshots: int,
) -> np.ndarray:
    """Energy descriptor a_i : amplitude, mode norm and time evolution.

    a_i = |b_i| * ||phi_i|| * sum_p |lambda_i|^p over the snapshot window.

    The mode norm alone would carry no information, since an eigenvector is
    defined only up to a constant ; pairing it with the amplitude and the time
    factor is what makes the descriptor describe this signal.
    """
    return _integrated_energies(modes, amplitudes, eigenvalues, n_snapshots)