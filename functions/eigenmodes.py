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


# Match this timestep's eigenmodes to the previous timestep's, so that a given
# index follows the same physical mode across time (rather than following raw
# eigenvalue-magnitude rank, which shuffles as the turbulence evolves).
def track_modes(prev_vecs, eigVecs, n_track=4, pool=None):
    """
    Reorder eigVecs to match prev_vecs by overlap, using optimal assignment.

    Similarity is the phase-invariant normalised overlap magnitude
        S[j, c] = |<prev_vecs[:, j] | eigVecs[:, c]>| / (||prev_vecs[:, j]|| ||eigVecs[:, c]||)
    which lies in [0, 1] and is unaffected by the arbitrary global phase that
    np.linalg.eig assigns each eigenvector. Modes are matched one-to-one by
    minimising cost = 1 - S via the Hungarian algorithm.

    :param prev_vecs: reference eigenvectors from the previous timestep, shape (N**2, n_track), columns are modes.
    :type prev_vecs: np.ndarray
    :param eigVecs: current eigenvectors (columns), already sorted by eigenvalue magnitude.
    :type eigVecs: np.ndarray
    :param n_track: number of modes (tracks) to follow.
    :type n_track: int
    :param pool: number of top current modes to consider as match candidates (default max(4*n_track, n_track)).
    :type pool: int
    :return: (perm, overlaps) where perm is an index array reordering the columns of eigVecs
             so that column j is the mode matched to track j (untracked modes appended in
             their existing order), and overlaps[j] is the matched similarity S for track j.
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    total = eigVecs.shape[1]
    n_track = min(n_track, prev_vecs.shape[1], total)
    if pool is None:
        pool = max(4 * n_track, n_track)
    pool = min(pool, total)

    ref = prev_vecs[:, :n_track]
    cand = eigVecs[:, :pool]

    # Normalised overlap magnitude between each reference track and each candidate
    ref_n = ref / (np.linalg.norm(ref, axis=0, keepdims=True) + 1e-30)
    cand_n = cand / (np.linalg.norm(cand, axis=0, keepdims=True) + 1e-30)
    S = np.abs(ref_n.conj().T @ cand_n)   # shape (n_track, pool)

    # Optimal one-to-one assignment (minimise 1 - S == maximise similarity)
    rows, cols = linear_sum_assignment(1.0 - S)
    # rows are 0..n_track-1 in order; cols are the matched candidate indices
    order = np.argsort(rows)
    matched = cols[order]
    overlaps = S[rows[order], matched]

    # Build a full permutation: matched tracks first (in track order), then the
    # remaining modes in their original order so no mode is dropped.
    matched_set = set(matched.tolist())
    remaining = [c for c in range(total) if c not in matched_set]
    perm = np.array(list(matched) + remaining, dtype=int)

    return perm, overlaps

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
