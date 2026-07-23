"""Dynamic Mode Decomposition (DMD) on a snapshot sequence.

Implements Algorithm 1 : the exact-DMD variant that avoids forming the full
n x n operator A by projecting onto a small basis built from the SVD.

Terminology used throughout:
  snapshot   : one column of the data matrix ; one time frame of the spectrogram
  mode phi_i : a characteristic spatial (here : spectral) shape
  eigenvalue lambda_i : how mode i evolves from one snapshot to the next
"""

import numpy as np


def build_snapshot_pairs(data):
    """Split a snapshot matrix into the two shifted matrices Psi_p and Psi_p1.

    Parameters
    ----------
    data : ndarray, shape (n, m)
        n rows are state components (frequency bins) ; m columns are snapshots.

    Returns
    -------
    psi_p, psi_p1 : each shape (n, m-1)
        psi_p holds snapshots 1..m-1 ; psi_p1 holds snapshots 2..m.
    """
    # DMD relates each snapshot to the next one ; so we need the sequence
    # shifted by one column against itself
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_states, n_snapshots)")
    if data.shape[1] < 2:
        raise ValueError("need at least 2 snapshots to form a pair")
    psi_p = data[:, :-1]
    psi_p1 = data[:, 1:]
    return psi_p, psi_p1


def _thin_svd(psi_p):
    """Return U, sigma, V of the thin SVD of Psi_p, dominant direction first.

    Algorithm 1 states the basis via the Gram product (Psi_p)^T Psi_p ; that is
    mathematically the same decomposition but it squares the condition number,
    so small singular values come out of it with badly degraded accuracy. The
    thin SVD gives the identical U, Sigma, V at full precision. Storage is not
    a concern either way : the thin SVD of an n x m matrix with m = 47 never
    forms anything larger than n x 47, so the huge n x n operator is still
    never built, which was the point of the low-storage step.
    """
    # full_matrices=False keeps U at n x m instead of n x n
    U, sigma, Vh = np.linalg.svd(psi_p, full_matrices=False)
    # numpy already returns singular values in descending order
    return U, sigma, Vh.conj().T


def choose_rank(sigma, rank=None, energy_threshold=None):
    """Decide how many modes to keep.

    Exactly one of rank or energy_threshold should be given.
    Returns an integer r >= 1.
    """
    # count of singular values that are numerically nonzero ; this is the
    # mathematical ceiling on how many modes carry information
    tol = np.finfo(float).eps * max(sigma.shape) * (sigma[0] if sigma.size else 0.0)
    n_nonzero = int(np.sum(sigma > tol))
    n_nonzero = max(n_nonzero, 1)

    if rank is not None and energy_threshold is not None:
        raise ValueError("give either rank or energy_threshold, not both")

    if rank is not None:
        # cannot keep more modes than there is information for
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
        # first index where we have captured the requested fraction
        r = int(np.searchsorted(cumulative, energy_threshold) + 1)
        return int(min(max(r, 1), n_nonzero))

    # default : keep everything that is numerically meaningful
    return n_nonzero


def dmd(data, rank=None, energy_threshold=None, mean_centre=True, dt=1.0):
    """Run DMD on a snapshot matrix and return modes, eigenvalues, amplitudes.

    Parameters
    ----------
    data : ndarray, shape (n, m)
        Snapshot matrix ; columns are successive time frames.
    rank : int or None
        Number of modes to keep ; mutually exclusive with energy_threshold.
    energy_threshold : float or None
        Keep the fewest modes capturing this fraction of the singular energy.
    mean_centre : bool
        Subtract the time-mean snapshot before decomposing ; this removes the
        large stationary background so modes describe actual evolution.
    dt : float
        Snapshot spacing in seconds ; used only by the descriptor helpers.

    Returns
    -------
    dict with keys :
        modes        : (n, r) complex ; the DMD modes phi_i
        eigenvalues  : (r,) complex ; the discrete-time eigenvalues lambda_i
        amplitudes   : (r,) complex ; b_i from projecting the first snapshot
        singular_values : (m-1,) real ; full spectrum, for rank diagnostics
        rank         : int ; how many modes were kept
        mean_snapshot : (n,) or None ; what was subtracted, if anything
    """
    data = np.asarray(data, dtype=float)

    # remove the stationary background so DMD does not spend its dominant mode
    # explaining a constant offset ; see project notes on mean-centring
    mean_snapshot = None
    if mean_centre:
        mean_snapshot = data.mean(axis=1, keepdims=True)
        data = data - mean_snapshot

    psi_p, psi_p1 = build_snapshot_pairs(data)

    # steps 1 to 3 : the SVD of Psi_p supplies the reduced basis U
    U_full, sigma, V_full = _thin_svd(psi_p)

    r = choose_rank(sigma, rank=rank, energy_threshold=energy_threshold)

    # truncate to the r dominant directions
    U = U_full[:, :r]
    V_r = V_full[:, :r]
    sigma_r = sigma[:r]
    # guard against dividing by a vanishing singular value
    sigma_inv = np.zeros_like(sigma_r)
    nonzero = sigma_r > 0
    sigma_inv[nonzero] = 1.0 / sigma_r[nonzero]

    # step 4 : project the operator onto that basis ; A_tilde is only r x r
    A_tilde = U.conj().T @ psi_p1 @ V_r * sigma_inv

    # step 5 : eigendecomposition of the small operator
    lambdas, Y = np.linalg.eig(A_tilde)

    # step 6 : lift the eigenvectors back to the full state space
    modes = U @ Y

    # amplitudes : how strongly each mode is present in the first snapshot ;
    # this is what makes the energy descriptor informative, unlike a bare norm
    # of a mode whose scale is arbitrary
    b = np.linalg.lstsq(modes, data[:, 0], rcond=None)[0]

    # order modes by their actual contribution, strongest first
    order = np.argsort(np.abs(b))[::-1]

    return {
        "modes": modes[:, order],
        "eigenvalues": lambdas[order],
        "amplitudes": b[order],
        "singular_values": sigma,
        "rank": r,
        "mean_snapshot": None if mean_snapshot is None else mean_snapshot.ravel(),
        "dt": dt,
    }


def mode_frequencies(eigenvalues, dt):
    """Oscillation frequency in Hz : f_i = |Im(log lambda_i)| / (2 pi dt).

    The absolute value collapses each complex-conjugate pair to one positive
    frequency ; a purely real positive eigenvalue gives f = 0 (no oscillation).
    """
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(eigenvalues)
    return np.abs(np.imag(logs)) / (2.0 * np.pi * dt)


def mode_growth_rates(eigenvalues, dt):
    """Growth or decay rate in 1/s : g_i = Re(log lambda_i) / dt.

    Positive means the mode strengthens over the pass ; negative means it fades.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(eigenvalues)
    return np.real(logs) / dt


def mode_energies(modes, amplitudes):
    """Energy of each mode : |b_i| times the mode norm.

    Using the amplitude b_i is essential ; eigenvector scale is arbitrary, so
    the norm of a mode on its own carries no information about the signal.
    """
    modes = np.asarray(modes)
    amplitudes = np.asarray(amplitudes)
    return np.abs(amplitudes) * np.linalg.norm(modes, axis=0)