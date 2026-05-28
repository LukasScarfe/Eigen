# Functions for determining eigenmodes

# Imports 
import numpy as np
from LightPipes import *
import cmath
import matplotlib.pyplot as plt

from functions.propagation import *


# Find eigenvalues and eigenvectors
def eigen_vals_vecs(end_fields):
    # Define transfer matrix and its transpose
    transferMatrix=end_fields
    transferMatrixT=end_fields.T

    # Find Hermitian transfer matrix. not in use currently
    # transferMatrixH=end_fields.T.conj()
    # transferMatrixHermitian=transferMatrixH@transferMatrix

    print("Determining eigenvalues and eigenvectors...")

    eigVals,eigVecs=np.linalg.eig(transferMatrixT)

    print("Done.")

    #Get the absolute values and phases of the eigenvalues
    eigMags=abs(eigVals)

    # Sort the Eigenvalues by the magnitudes
    sort_indices = np.argsort(eigMags)[::-1]
    eigMags = eigMags[sort_indices]
    eigVals = eigVals[sort_indices]
    eigVecs = eigVecs[:, sort_indices]

    return eigVals, eigVecs, eigMags

# Find eigenmodes
def eigenmodes(size, wavelength, N, z, eigVecs, abbs):

    #Making Eigenvector optical modes
    F=Begin(size,wavelength,N)
    eigenBeams=[]
    
    for i in progress(range(100) if len(eigVecs)>100 else range(N**2), desc="Determining eigenbeams..."):

        mode=eigVecs[:,i]
        
        eigenInt=[abs(val)**2 for val in mode]
        eigenInt = np.array(eigenInt).reshape((N,N))
        #eigenInt=np.pad(np.array(eigenInt).reshape((N,N)), pad_width=int(0), mode='constant', constant_values=0)
        eigenPhase=[cmath.phase(val) for val in mode]
        eigenPhase = np.array(eigenPhase).reshape((N,N))
        #eigenPhase=np.pad(np.array(eigenPhase).reshape((N,N)),pad_width=int(0), mode='constant', constant_values=0)
        F=SubPhase(SubIntensity(F,eigenInt),eigenPhase)
        eigenBeams.append(F)

        # Plot intensity and phase for this eigenmode
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'Eigenmode {i+1}')

        im0 = axes[0].imshow(eigenInt, cmap='inferno')
        axes[0].set_title('Intensity')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(eigenPhase, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Phase')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show(block=False)
        input(f"  Showing eigenmode {i+1}. Press Enter to continue to the next eigenmode...")
        plt.close(fig)

    eigenBeamsPropagated=[propChannel(beam,z,abbs) for beam in progress(eigenBeams, desc="Propagating eigenbeams...")]

    # Normalize phase
    # Loop through eigenbeams and find position with max intensity and then find the phase there. Then find the difference in 
    # phase in the same position after propagating and subtract the entire array after propagating by this phase difference
    for i, eigenBeam in enumerate(eigenBeams):
        intensity = Intensity(eigenBeam,1)
        max_pos = np.unravel_index(intensity.argmax(), intensity.shape)
        phase_diff = np.mod(Phase(eigenBeamsPropagated[i]), 2*np.pi)[max_pos] - np.mod(Phase(eigenBeam), 2*np.pi)[max_pos]
        eigenBeamsPropagated[i].field *= np.exp(1j * (-phase_diff)) # Subtract phase difference

    return eigenBeams, eigenBeamsPropagated
