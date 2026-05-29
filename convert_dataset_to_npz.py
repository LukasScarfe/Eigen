"""
convert_dataset_to_npz.py
--------------------------
Converts the existing CSV-based eigenmode dataset into a single .npz file
for faster loading during FNO training.

The .npz file stores one array per field key, shaped (N_samples, H, W),
where N_samples is the total number of timestep directories found.

Keys stored
-----------
  eigenmode_{k}_real / eigenmode_{k}_imag  : k = 1..4
  gaussian_forward_real / gaussian_forward_imag
  gaussian_reversed_real / gaussian_reversed_imag

Usage
-----
  python convert_dataset_to_npz.py --dataset_dir datasets/dataset --output dataset.npz
"""

import os
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Convert CSV eigenmode dataset to a single .npz file.")
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=r"datasets\dataset2",
    help="Path to the root dataset directory containing t_XXXXX subdirectories.",
)
parser.add_argument(
    "--output",
    type=str,
    default="dataset2.npz",
    help="Output .npz file path.",
)
parser.add_argument(
    "--n_eigenmodes",
    type=int,
    default=4,
    help="Number of eigenmodes per sample (default: 4).",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Helpers (mirrors train_eigenmodes.py exactly)
# ---------------------------------------------------------------------------

def load_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def load_sample(sample_dir: str, n_eigenmodes: int) -> dict:
    """Load all eigenmodes and Gaussian propagation fields from a sample directory."""
    data = {}
    for k in range(1, n_eigenmodes + 1):
        data[f"eigenmode_{k}_real"] = load_csv(os.path.join(sample_dir, f"eigenmode_{k:03d}_real.csv"))
        data[f"eigenmode_{k}_imag"] = load_csv(os.path.join(sample_dir, f"eigenmode_{k:03d}_imag.csv"))
    data["gaussian_forward_real"]  = load_csv(os.path.join(sample_dir, "gaussian_prop_forward_real.csv"))
    data["gaussian_forward_imag"]  = load_csv(os.path.join(sample_dir, "gaussian_prop_forward_imag.csv"))
    data["gaussian_reversed_real"] = load_csv(os.path.join(sample_dir, "gaussian_prop_reversed_real.csv"))
    data["gaussian_reversed_imag"] = load_csv(os.path.join(sample_dir, "gaussian_prop_reversed_imag.csv"))
    return data

# ---------------------------------------------------------------------------
# Discover sample directories (same sort order as train_eigenmodes.py)
# ---------------------------------------------------------------------------

sample_dirs = sorted([
    os.path.join(args.dataset_dir, d)
    for d in os.listdir(args.dataset_dir)
    if os.path.isdir(os.path.join(args.dataset_dir, d))
])

n_samples = len(sample_dirs)
print(f"Found {n_samples} sample directories in '{args.dataset_dir}'")

# ---------------------------------------------------------------------------
# Pre-allocate arrays
# ---------------------------------------------------------------------------

# Peek at the first sample to get the grid shape
first = load_sample(sample_dirs[0], args.n_eigenmodes)
H, W  = first["gaussian_forward_real"].shape

print(f"Grid shape: H={H}, W={W}")
print(f"Allocating arrays for {n_samples} samples ...")

keys = (
    [f"eigenmode_{k}_real" for k in range(1, args.n_eigenmodes + 1)]
    + [f"eigenmode_{k}_imag" for k in range(1, args.n_eigenmodes + 1)]
    + ["gaussian_forward_real", "gaussian_forward_imag",
       "gaussian_reversed_real", "gaussian_reversed_imag"]
)

arrays = {key: np.empty((n_samples, H, W), dtype=np.float32) for key in keys}

# ---------------------------------------------------------------------------
# Fill arrays
# ---------------------------------------------------------------------------

for i, sample_dir in enumerate(sample_dirs):
    if (i + 1) % 50 == 0 or i == 0 or i == n_samples - 1:
        print(f"  Loading sample {i+1}/{n_samples}  ({sample_dir})")
    sample = load_sample(sample_dir, args.n_eigenmodes)
    for key in keys:
        arrays[key][i] = sample[key].astype(np.float32)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print(f"\nSaving to '{args.output}' ...")
np.savez(args.output, **arrays)
print("Done!")
print(f"\nArrays saved:")
for key in keys:
    print(f"  {key:35s}  shape={arrays[key].shape}  dtype={arrays[key].dtype}")
