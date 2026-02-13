# Functions for determining eigenmodes

# Imports 
import numpy as np
from LightPipes import *
import cmath

from functions.propagation import *


# Find eigenvalues and eigenvectors
def eigen_vals_vecs(end_fields):
    # Define transfer matrix and its transpose
    transferMatrix=end_fields
    transferMatrixT=end_fields.T

    # Find Hermitian transfer matrix. not in use currently
    # transferMatrixH=end_fields.T.conj()
    # transferMatrixHermitian=transferMatrixH@transferMatrix

    eigVals,eigVecs=np.linalg.eig(transferMatrixT)

    #Get the absolute values and phases of the eigenvalues
    eigMags=abs(eigVals)

    # Sort the Eigenvalues by the magnitudes
    sort_indices = np.argsort(eigMags)[::-1]
    eigMags = eigMags[sort_indices]
    eigVals = eigVals[sort_indices]
    eigVecs = eigVecs[:, sort_indices]

    return eigVals, eigVecs

# Find eigenmodes
def eigenmodes(size, wavelength, N, z, eigVecs, abbs):
    #Making Eigenvector optical modes
    F=Begin(size,wavelength,N)
    eigenBeams=[]
    for i in progress(range(100)):
        mode=eigVecs[:,i]
        eigenInt=[abs(val)**2 for val in mode]
        eigenInt=np.pad(np.array(eigenInt).reshape((N,N)),pad_width=int(0), mode='constant', constant_values=0)
        eigenPhase=[cmath.phase(val) for val in mode]
        eigenPhase=np.pad(np.array(eigenPhase).reshape((N,N)),pad_width=int(0), mode='constant', constant_values=0)
        F=SubPhase(SubIntensity(F,eigenInt),eigenPhase)
        eigenBeams.append(F)

    eigenBeamsPropagated=[propChannel(beam,z,abbs) for beam in progress(eigenBeams)]

    # Normalize phase
    # Loop through eigenbeams and find position with max intensity and then find the phase there. Then find the difference in 
    # phase in the same position after propagating and subtract the entire array after propagating by this phase difference
    for i, eigenBeam in enumerate(eigenBeams):
        intensity = Intensity(eigenBeam,1)
        max_pos = np.unravel_index(intensity.argmax(), intensity.shape)
        phase_diff = np.mod(Phase(eigenBeamsPropagated[i]), 2*np.pi)[max_pos] - np.mod(Phase(eigenBeam), 2*np.pi)[max_pos]
        eigenBeamsPropagated[i].field *= np.exp(1j * (-phase_diff)) # Subtract phase difference

    return eigenBeams, eigenBeamsPropagated
