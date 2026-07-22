# Writing parameters to yaml file

import yaml
from LightPipes                 import *
import numpy as np

# parameters used to define other parameters
size = 45e-2 # window length
lensSize=size/4 # Radius

#Beam Parameters
wavelength = 633*nm

#Propagataion Params
z=5000*m
num_phase_screens=4

#Turbulence Parameters
C2_n = { # These can be toyed around with
    'WeakestTurb' : 1e-19,
    'WeakerTurb' : 1e-18,
    'WeakTurb' : 1e-17,
    'MidWeakerTurb' : 1.5e-17,
    'MidWeakTurb' : 1e-16,
    'MidTurb' : 1e-15,
    'StrongTurb' : 1e-14,
    'StrongerTurb' : 1e-13 
}

TurbStrength = 'MidTurb'

# Parameters to be written to the YAML file
parameters = {
    'size' : 45*cm, # window length
    'N' : 64, # resolution
    'lensSize' : size/4, # Radius
    'wavelength' : 633*nm,
    'w0' : lensSize/1.2, # Radius
    'z' : 5000*m, # Propagation distance
    'num_phase_screens' : 4,
    'C2_n' : { #Turbulence Parameters # These can be toyed around with
        'WeakestTurb' : 1e-19,
        'WeakerTurb' : 1e-18,
        'WeakTurb' : 1e-17,
        'MidWeakerTurb' : 1.5e-17,
        'MidWeakTurb' : 1e-16,
        'MidTurb' : 1e-15,
        'StrongTurb' : 1e-14,
        'StrongerTurb' : 1e-13 
    },
    'TurbStrength' : 'MidTurb',
    'shift' : 1, # number of pixels we move per fluctuation
    'num_of_fluctuations' : 3,
    'start_point' : 0,
    'phase_screen_seed' : 47, # Seed to use to generate initial phase screens
    # Determine the "total" and "partial" Rytov parameter
        # Total does not consider phase screens
        # Partial considers phase screens
    'Rytov_total' : 1.23*C2_n[TurbStrength]*pow(2*np.pi/wavelength, 7/6)*pow(z, 11/6),
    'Rytov_part' : 1.23*C2_n[TurbStrength]*pow(2*np.pi/wavelength, 7/6)*pow(z/num_phase_screens, 11/6),
    'dataset_collect' : True, # Set parameter to save data to a folder and propagate a Gaussian through the channel too
    'plotting' : False, # Plot stuff boolean
    'dataset_folder_name' : "dataset", # Dataset folder name
    'animate' : True, # Render the end-of-run top-4 eigenmode animation
    'fps' : 2, # Frames per second for the eigenmode animation
}

# Writing the data to a YAML file
with open('parameters.yaml', 'w') as file:
    yaml.dump(parameters, file)

print("Data has been written to 'parameters.yaml'")