# Functions for determining eigenmodes

# Imports
import numpy as np
from LightPipes import *
import cmath
from scipy.optimize import linear_sum_assignment

from functions.propagation import *


# Find eigenvalues and eigenvectors
def eigen_vals_vecs(end_fields):
    # Define transfer matrix and its transpose
    transferMatrix=end_fields
    transferMatrixT=end_fields.T

    # Find Hermitian transfer matrix. not in use currently
    # transferMatrixH=end_fields.T.conj()
    # transferMatrixHermitian=transferMatrixH@transferMatrix

    # print("Determining eigenvalues and eigenvectors...")
    eigVals,eigVecs=np.linalg.eig(transferMatrixT)
    # print("Done.")

    #Get the absolute values and phases of the eigenvalues
    eigMags=abs(eigVals)

    # Sort the Eigenvalues by the magnitudes
    sort_indices = np.argsort(eigMags)[::-1]
    eigMags = eigMags[sort_indices]
    eigVals = eigVals[sort_indices]
    eigVecs = eigVecs[:, sort_indices]

    return eigVals, eigVecs, eigMags


# Smoothness score of eigenvector "modes" (columns), used to prioritise modes
# with few lobes. A mode with few lobes has its intensity concentrated near the
# centre of its 2D spatial-frequency spectrum; a many-lobed / speckle-like mode
# spreads energy out to high spatial frequencies. We score each mode by the
# fraction of its total power that sits inside a low-frequency disc of the 2D FFT:
# 1.0 == all energy at DC (a single smooth blob), → 0 as lobes proliferate.
def mode_smoothness(vecs, N, lowpass_frac=0.25):
    """
    Fraction of each mode's spectral power inside a central low-frequency disc.

    :param vecs: eigenvectors as columns, shape (N**2, M). Each column is reshaped to (N, N).
    :type vecs: np.ndarray
    :param N: linear grid size (columns have length N**2).
    :type N: int
    :param lowpass_frac: radius of the low-frequency disc as a fraction of the
        Nyquist frequency. Smaller = stricter "few lobes" definition.
    :type lowpass_frac: float
    :return: smoothness score in [0, 1] per column, higher == fewer lobes.
    :rtype: np.ndarray
    """
    M = vecs.shape[1]
    fields = vecs.reshape(N, N, M)                     # (N, N, mode)
    spec = np.fft.fftshift(np.fft.fft2(fields, axes=(0, 1)), axes=(0, 1))
    power = np.abs(spec) ** 2                          # spectral power density
    # Radial coordinate of the shifted 2D FFT, normalised so Nyquist ~ 1.
    ky = np.fft.fftshift(np.fft.fftfreq(N))[:, None]
    kx = np.fft.fftshift(np.fft.fftfreq(N))[None, :]
    r = np.sqrt(kx ** 2 + ky ** 2) / 0.5              # 0.5 cyc/px == Nyquist
    lowpass = (r <= lowpass_frac)[:, :, None]         # (N, N, 1) disc mask
    total = power.sum(axis=(0, 1)) + 1e-30
    return (power * lowpass).sum(axis=(0, 1)) / total


# Match this timestep's eigenmodes to a per-track reference (an anchor that
# follows real drift but resists single-frame swaps), so that a given index
# follows the same physical mode across time (rather than following raw
# eigenvalue-magnitude rank, which shuffles as the turbulence evolves).
def track_modes(ref_vecs, eigVecs, n_track=4, pool=None,
                N=None, lobe_weight=0.0, min_overlap=0.5):
    """
    Reorder eigVecs to match ref_vecs by overlap, using optimal assignment,
    with an optional low-lobe tiebreak and a hold-on-bad-match policy.

    Similarity is the phase-invariant normalised overlap magnitude
        S[j, c] = |<ref_vecs[:, j] | eigVecs[:, c]>| / (||ref_vecs[:, j]|| ||eigVecs[:, c]||)
    which lies in [0, 1] and is unaffected by the arbitrary global phase that
    np.linalg.eig assigns each eigenvector. Modes are matched one-to-one by
    minimising cost = (1 - S) - lobe_weight * smoothness(candidate) via the
    Hungarian algorithm, so a small `lobe_weight` breaks near-ties in favour of
    the smoother (fewer-lobed) candidate without overriding a clear overlap.

    :param ref_vecs: reference/anchor eigenvectors, shape (N**2, n_track), columns are modes.
    :type ref_vecs: np.ndarray
    :param eigVecs: current eigenvectors (columns), already sorted by eigenvalue magnitude.
    :type eigVecs: np.ndarray
    :param n_track: number of modes (tracks) to follow.
    :type n_track: int
    :param pool: number of top current modes to consider as match candidates (default max(4*n_track, n_track)).
    :type pool: int
    :param N: linear grid size, required if lobe_weight > 0 (for the smoothness tiebreak).
    :type N: int
    :param lobe_weight: weight of the low-lobe tiebreak added to the assignment cost.
        0 disables it (pure overlap matching). Keep small (~0.05) so it only breaks ties.
    :type lobe_weight: float
    :param min_overlap: matched overlaps below this are flagged as unreliable in `weak`
        (the caller decides whether to hold the anchor for those tracks).
    :type min_overlap: float
    :return: (perm, overlaps, weak) where perm reorders the columns of eigVecs so that
             column j is the mode matched to track j (untracked modes appended in their
             existing order), overlaps[j] is the matched similarity S for track j, and
             weak[j] is True where overlaps[j] < min_overlap (likely mode swap).
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    total = eigVecs.shape[1]
    n_track = min(n_track, ref_vecs.shape[1], total)
    if pool is None:
        pool = max(4 * n_track, n_track)
    pool = min(pool, total)

    ref = ref_vecs[:, :n_track]
    cand = eigVecs[:, :pool]

    # Normalised overlap magnitude between each reference track and each candidate
    ref_n = ref / (np.linalg.norm(ref, axis=0, keepdims=True) + 1e-30)
    cand_n = cand / (np.linalg.norm(cand, axis=0, keepdims=True) + 1e-30)
    S = np.abs(ref_n.conj().T @ cand_n)   # shape (n_track, pool)

    cost = 1.0 - S
    # Low-lobe tiebreak: subtract a smoothness bonus per candidate (same for every
    # track/row), so among near-equal-overlap candidates the smoother one wins.
    if lobe_weight > 0.0:
        if N is None:
            raise ValueError("track_modes: N is required when lobe_weight > 0")
        smooth = mode_smoothness(cand, N)             # (pool,)
        cost = cost - lobe_weight * smooth[None, :]

    # Optimal one-to-one assignment (minimise cost == maximise overlap + lobe bonus)
    rows, cols = linear_sum_assignment(cost)
    # rows are 0..n_track-1 in order; cols are the matched candidate indices
    order = np.argsort(rows)
    matched = cols[order]
    overlaps = S[rows[order], matched]
    weak = overlaps < min_overlap

    # Build a full permutation: matched tracks first (in track order), then the
    # remaining modes in their original order so no mode is dropped.
    matched_set = set(matched.tolist())
    remaining = [c for c in range(total) if c not in matched_set]
    perm = np.array(list(matched) + remaining, dtype=int)

    return perm, overlaps, weak

# Find eigenmodes
def eigenmodes(size, wavelength, N, z, eigVecs, abbs):
    #Making Eigenvector optical modes
    F=Begin(size,wavelength,N)
    eigenBeams=[]
    for i in progress(range(100) if len(eigVecs)>100 else range(N**2), desc="Determining eigenbeams...", disable=True):
        mode=eigVecs[:,i]
        eigenInt=[abs(val)**2 for val in mode]
        eigenInt=np.pad(np.array(eigenInt).reshape((N,N)),pad_width=int(0), mode='constant', constant_values=0)
        eigenPhase=[cmath.phase(val) for val in mode]
        eigenPhase=np.pad(np.array(eigenPhase).reshape((N,N)),pad_width=int(0), mode='constant', constant_values=0)
        F=SubPhase(SubIntensity(F,eigenInt),eigenPhase)
        eigenBeams.append(F)

    eigenBeamsPropagated=[propChannel(beam,z,abbs) for beam in progress(eigenBeams, desc="Propagating eigenbeams...", disable=True)]

    # Normalize phase
    # Loop through eigenbeams and find position with max intensity and then find the phase there. Then find the difference in 
    # phase in the same position after propagating and subtract the entire array after propagating by this phase difference
    for i, eigenBeam in enumerate(eigenBeams):
        intensity = Intensity(eigenBeam,1)
        max_pos = np.unravel_index(intensity.argmax(), intensity.shape)
        phase_diff = np.mod(Phase(eigenBeamsPropagated[i]), 2*np.pi)[max_pos] - np.mod(Phase(eigenBeam), 2*np.pi)[max_pos]
        eigenBeamsPropagated[i].field *= np.exp(1j * (-phase_diff)) # Subtract phase difference

    return eigenBeams, eigenBeamsPropagated
