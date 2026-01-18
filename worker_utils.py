
import numpy as np
from LightPipes import SubIntensity, SubPhase, CircAperture, Phase

# Global dictionary to store data in the worker's memory space
worker_data = {}

def init_worker(field_data, size, wavelength, N, z, lensSize, abbs):
    """
    Initializes the worker process by storing shared data.
    We pass raw data (like numpy arrays) rather than complex objects
    to ensure smooth serialization (pickling) on Windows.
    """
    global worker_data
    worker_data['field_data'] = field_data
    worker_data['size'] = size
    worker_data['wavelength'] = wavelength
    worker_data['N'] = N
    worker_data['z'] = z
    worker_data['lensSize'] = lensSize
    worker_data['abbs'] = abbs

def propagate_channel_local(F, distance, abbs):
    """
    Local helper to handle propagation logic within the worker.
    """
    from LightPipes import Fresnel, Forvard
    
    # Logic based on your propChannel function
    F = Fresnel(F, distance / (len(abbs) + 1))
    for screen in abbs:
        F = SubPhase(F, Phase(F) + screen)
        F = Fresnel(F, distance / (len(abbs) + 1))
    return F

def propagateSinglePixel_Optimized(j, i):
    """
    The function executed by the ProcessPoolExecutor.
    """
    from LightPipes import Begin
    
    # Retrieve data from worker's global storage
    field_data = worker_data['field_data']
    size = worker_data['size']
    wavelength = worker_data['wavelength']
    N = worker_data['N']
    z = worker_data['z']
    lensSize = worker_data['lensSize']
    abbs = worker_data['abbs']

    # Reconstruct the Field object from raw data
    F_in = Begin(size, wavelength, N)
    F_in.field = field_data

    # 1. Create the intensity matrix
    intens = np.zeros((N, N))
    intens[j][i] = 1

    # 2. Prepare field at sender
    fieldAtSender = CircAperture(
        SubPhase(SubIntensity(F_in, intens), np.zeros((N, N))), 
        lensSize
    )

    # 3. Propagate through the channel
    FieldOut = CircAperture(propagate_channel_local(fieldAtSender, z, abbs), lensSize)

    # 4. Extract data for return (Numpy arrays are picklable)
    endField_data = np.array(FieldOut.field).reshape(N**2)

    return FieldOut, endField_data