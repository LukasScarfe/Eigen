# Generate parameters.yaml for data_generation_yaml.py.
# Every quantity is defined exactly once below, then assembled into `parameters`
# and dumped to YAML. Edit the values here; do not hand-edit parameters.yaml.

import yaml
import numpy as np
from LightPipes import nm, cm, m

# --- Grid / geometry ---
size = 45 * cm          # window side length
N = 32                  # grid resolution (N x N)
lensSize = size / 4     # lens radius

# --- Beam ---
wavelength = 633 * nm
w0 = lensSize / 1.2     # beam radius

# --- Propagation / channel ---
z = 5000 * m            # total propagation distance
num_phase_screens = 3

# --- Eigenmodes ---
n_eigenmodes = 2        # number of eigenmodes to track/save/animate each timestep

# --- Turbulence ---
# Cn^2 presets; pick one via TurbStrength. (These can be toyed around with.)
C2_n = {
    'WeakestTurb':   1e-19,
    'WeakerTurb':    1e-18,
    'WeakTurb':      1e-17,
    'MidWeakerTurb': 1.5e-17,
    'MidWeakTurb':   1e-16,
    'MidTurb':       1e-15,
    'StrongTurb':    1e-14,
    'StrongerTurb':  1e-13,
}
TurbStrength = 'WeakTurb'
cn2 = C2_n[TurbStrength]


# --- Time evolution ---
shift = 1                       # pixels each phase screen advances per step
screen_update_mode = 'cycle'    # 'all' = every screen advances each step; 'cycle' = one per step, round-robin
num_of_fluctuations = 200       # number of timesteps generated is this + 1
start_point = 0                 # warm-up steps to evolve the channel before collecting data
phase_screen_seed = 47          # seed for the initial phase screens

# --- Output ---
dataset_collect = True          # save fields to a folder (and propagate a Gaussian through the channel)
dataset_folder_name = f"datasets/dataset_{N}_{TurbStrength}"
plotting = False                # show per-step eigenbeam plots
animate = True                  # render the end-of-run top n_eigenmode animation
fps = 8                         # frames per second for the animations


# Rytov variance (informational; written to parameters.txt).
#   total   ignores phase screens; partial uses per-screen distance z / num_phase_screens.
rytov = lambda dist: 1.23 * cn2 * pow(2 * np.pi / wavelength, 7 / 6) * pow(dist, 11 / 6)
Rytov_total = rytov(z)
Rytov_part = rytov(z / num_phase_screens)

# Assemble and write.
parameters = {
    # Grid / geometry
    'size': size,
    'N': N,
    'lensSize': lensSize,
    # Beam
    'wavelength': wavelength,
    'w0': w0,
    # Propagation / channel
    'z': z,
    'num_phase_screens': num_phase_screens,
    # Eigenmodes
    'n_eigenmodes': n_eigenmodes,
    # Turbulence
    'C2_n': C2_n,
    'TurbStrength': TurbStrength,
    'Rytov_total': Rytov_total,
    'Rytov_part': Rytov_part,
    # Time evolution
    'shift': shift,
    'screen_update_mode': screen_update_mode,
    'num_of_fluctuations': num_of_fluctuations,
    'start_point': start_point,
    'phase_screen_seed': phase_screen_seed,
    # Output
    'dataset_collect': dataset_collect,
    'dataset_folder_name': dataset_folder_name,
    'plotting': plotting,
    'animate': animate,
    'fps': fps,
}

with open('parameters.yaml', 'w') as file:
    yaml.dump(parameters, file)

print("Data has been written to 'parameters.yaml'")
