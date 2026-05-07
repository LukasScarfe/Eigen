"""
generate_fresnel_dataset.py
----------------------------
Generates a dataset of input-output pairs for training a Fourier Neural Operator
to model Fresnel propagation of a Gaussian beam.

Each sample consists of:
  - input:  [prop_dist, Re(field), Im(field)]  — shape (3, N, N)
  - output: [Re(prop_field), Im(prop_field)]   — shape (2, N, N)

Adjustable parameters are defined in the CONFIG section below.
"""

import os
import numpy as np
import scipy
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Physical constants

nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2

RESOLUTION = 64          # Grid points per dimension (N × N)

# Fixed optical parameters

LAMBDA  = 0.78 * um             # Wavelength (m)
W0      = 0.10 * mm             # Beam waist (m)
MAXX    = 20 * um * RESOLUTION  # Full side length of numerical window (m)
K = (2*np.pi)/LAMBDA
ZR =  0.50*K*W0**2

# ─────────────────────────────────────────────
#  CONFIG  —  edit these before running
# ─────────────────────────────────────────────

N_SAMPLES       = 2000          # Number of input-output pairs to generate
Z_MIN           = -2.5*ZR          # Minimum propagation distance (m)  — 1 cm
Z_MAX           = 2.5*ZR        # Maximum propagation distance (m)  — 50 cm
SEED            = 42            # Random seed for reproducibility
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "fresnel_dataset")

# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────

def cart2pol(x, y):
    """Cartesian → polar coordinates."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def make_coords(N, maxx):
    """Return 2-D meshgrid (r, phi) and the grid step h."""
    dx = maxx / N
    X  = dx * (np.arange(N) - N // 2)
    Y  = dx * (np.arange(N) - N // 2)
    xx, yy = np.meshgrid(X, Y)
    r, phi = cart2pol(xx, yy)
    return r, phi, dx


def LG(RHO, PHI, ell, p, w0, h, z, k):
    """
    Normalized Laguerre-Gaussian field.
    For the Gaussian beam, use ell=0, p=0.
    """
    wL    = (2 * np.pi) / k
    z_o   = np.pi * w0**2 / wL
    w_z   = lambda z: w0 * np.sqrt(1 + (z / z_o)**2)
    R_z   = lambda z: z * (1 + (z_o / z)**2)
    zeta_z = lambda z: np.arctan(z / z_o)

    if z == 0:
        AK = (np.exp(-(RHO / w0)**2)
              * (RHO / w0)**abs(ell)
              * scipy.special.eval_genlaguerre(p, abs(ell), 2 * (RHO / w0)**2)
              * np.exp(1j * ell * PHI))
    else:
        AK = ((w0 / w_z(z))
              * np.exp(-(RHO / w_z(z))**2)
              * (RHO / w_z(z))**abs(ell)
              * scipy.special.eval_genlaguerre(p, abs(ell), 2 * (RHO / w_z(z))**2)
              * np.exp(1j * ell * PHI)
              * np.exp(-1j * k * z)
              * np.exp(-1j * k * (RHO**2 / (2 * R_z(z))))
              * np.exp(1j * (abs(ell) + 2 * p + 1) * zeta_z(z)))

    reNormFactor = np.sqrt(np.sum(np.conj(AK) * AK * h**2))
    return AK / reNormFactor


def propTF(u1, L, la, z):
    """
    Fresnel propagation via Transfer Function method.

    Parameters
    ----------
    u1 : (N, N) complex array — source field
    L  : float                — side length of numerical window (m)
    la : float                — wavelength (m)
    z  : float                — propagation distance (m)
    """
    M, _ = u1.shape
    dx   = L / M
    fx   = np.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / L)
    Fx, Fy = np.meshgrid(fx, fx)
    H    = np.exp(-1j * np.pi * la * z * (Fx**2 + Fy**2))
    H    = fftshift(H)
    U2   = H * fft2(fftshift(u1))
    u2   = ifftshift(ifft2(U2))
    return u2


# ─────────────────────────────────────────────
#  Dataset generation
# ─────────────────────────────────────────────

def generate_dataset(n_samples, resolution, z_min, z_max, seed, output_dir):
    print("Getting started")
    rng = np.random.default_rng(seed)

    k    = (2 * np.pi) / LAMBDA
    maxx = 20 * um * resolution

    r, phi, h = make_coords(resolution, maxx)

    # Pre-generate the source Gaussian field (same for every sample)
    source_field = LG(r, phi, ell=0, p=0, w0=W0, h=h, z=0, k=k)

    # Normalise so that the maximum absolute value is 1.
    # Because propTF is linear, the same factor scales every propagated field.
    norm_factor  = np.max(np.abs(source_field))
    source_field = source_field / norm_factor

    # Draw propagation distances from a uniform distribution
    prop_distances = rng.uniform(z_min, z_max, size=n_samples)

    # Arrays to hold the dataset
    # inputs:  (n_samples, 3, N, N)  — [z_channel, Re(field), Im(field)]
    # outputs: (n_samples, 2, N, N)  — [Re(prop_field), Im(prop_field)]
    inputs  = np.zeros((n_samples, 3, resolution, resolution), dtype=np.float32)
    outputs = np.zeros((n_samples, 2, resolution, resolution), dtype=np.float32)

    print(f"Generating {n_samples} samples at {resolution}×{resolution} resolution …")
    for i, z in enumerate(prop_distances):
        prop_field = propTF(source_field, maxx, LAMBDA, z)

        # Channel 0: propagation distance (broadcast as a constant plane)
        inputs[i, 0, :, :] = np.float32(z)
        # Channel 1 & 2: real and imaginary parts of the input field
        inputs[i, 1, :, :] = source_field.real.astype(np.float32)
        inputs[i, 2, :, :] = source_field.imag.astype(np.float32)

        # Output channels: real and imaginary parts of the propagated field
        outputs[i, 0, :, :] = prop_field.real.astype(np.float32)
        outputs[i, 1, :, :] = prop_field.imag.astype(np.float32)

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"  [{i+1:>{len(str(n_samples))}}/{n_samples}] done")

    # ── Save ──────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "inputs.npy"),          inputs)
    np.save(os.path.join(output_dir, "outputs.npy"),         outputs)
    np.save(os.path.join(output_dir, "prop_distances.npy"),  prop_distances)

    # Save metadata as a plain text file for quick reference
    meta_path = os.path.join(output_dir, "metadata.txt")
    with open(meta_path, "w") as f:
        f.write("Fresnel Gaussian Beam Dataset\n")
        f.write("=" * 40 + "\n")
        f.write(f"n_samples   : {n_samples}\n")
        f.write(f"resolution  : {resolution} x {resolution}\n")
        f.write(f"z_min (m)   : {z_min}\n")
        f.write(f"z_max (m)   : {z_max}\n")
        f.write(f"lambda (m)  : {LAMBDA}\n")
        f.write(f"w0 (m)      : {W0}\n")
        f.write(f"maxx (m)    : {maxx}\n")
        f.write(f"seed        : {seed}\n")
        f.write("\nArray shapes\n")
        f.write(f"  inputs          : {inputs.shape}   (n, 3, N, N)\n")
        f.write(f"    ch 0          : propagation distance z (m) — broadcast constant\n")
        f.write(f"    ch 1          : Re(source field)\n")
        f.write(f"    ch 2          : Im(source field)\n")
        f.write(f"  outputs         : {outputs.shape}  (n, 2, N, N)\n")
        f.write(f"    ch 0          : Re(propagated field)\n")
        f.write(f"    ch 1          : Im(propagated field)\n")
        f.write(f"  prop_distances  : {prop_distances.shape}\n")

    print(f"\nDataset saved to: {output_dir}")
    print(f"  inputs.npy         {inputs.nbytes  / 1e6:.1f} MB")
    print(f"  outputs.npy        {outputs.nbytes / 1e6:.1f} MB")
    print(f"  prop_distances.npy {prop_distances.nbytes / 1e3:.1f} kB")
    print(f"  metadata.txt")


if __name__ == "__main__":
    generate_dataset(
        n_samples  = N_SAMPLES,
        resolution = RESOLUTION,
        z_min      = Z_MIN,
        z_max      = Z_MAX,
        seed       = SEED,
        output_dir = OUTPUT_DIR,
    )

