"""
train_eigenmodes.py
-------------------
Standalone training script for the FNO eigenmode predictor.
Covers:
  1. Dataset loading
  2. Data preparation (single-step predictor pairs)
  3. Train / test split and DataLoaders
  4. FNO instantiation
  5. Training loop
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

import fno
from fno import FNO2d, spectral_relative_l2, make_freq_weights

import yaml
from yaml import Loader

import argparse

# Select configuration file according to parse index

parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii

# OMIT IF YOU ARE USING THIS IN A CLUSTER

#shift = 0

# These parameters we can safely leave fixed/hard-coded

N_EIGENMODES = 4
INPUT_CHANS  = 6
OUTPUT_CHANS = 2
RETRAIN_FNO  = np.random.randint(1, 9999)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load yaml file

with open(f'configs/train{shift}.yaml', 'r') as file:
    cnfg = yaml.load(file, Loader=Loader)

training_dataset = cnfg["training_dataset"]

# Training parameters

train_test_split = cnfg["train_test_split"] # train/test split
batch_size = cnfg["batch_size"] # batch training size
learn_rate = float(cnfg["learn_rate"]) # Initial learning rate 
step_size = cnfg["step_size"] # Epochs to decay the learning rate
gamma = cnfg["gamma"] # LR Decay Factor
epochs = cnfg["epochs"] # Number of training epochs 
stop_criterion_epochs = cnfg["stop_criterion_epochs"] # Number of epochs without improvement before stopping
min_delta = cnfg.get("min_delta", 1e-4)   # Minimum improvement in validation loss to count as progress

NPZ_PATH = cnfg.get("npz_path", f"datasets/{training_dataset}.npz")  # pre-converted .npz dataset
print(type(learn_rate))

# FNO architecture

modes1       = cnfg["modes1"]  # spectral modes along dimension 1
modes2       = cnfg["modes2"]    # spectral modes along dimension 2
width        = cnfg["width"]     # channel width inside the FNO
n_layers     = cnfg["n_layers"]     # number of Fourier layers    

use_mse_loss = cnfg['mse_loss']  # False → spectral relative L2; True → standard MSE

freq_print = 1        # print summary every N epochs

# Physical parameters for the circular aperture

cnfg.setdefault("size", 0.45)
size = cnfg["size"] # simulation window side length (metres)
lensSize = size/4 # aperture radius (metres) [= size/4]

cnfg.setdefault("model_checkpoint_dir", "trained_model_eigens")
model_checkpoint_dir = cnfg["model_checkpoint_dir"]

# Create model directory (if it doesn't already exist)
os.makedirs(f"models\{model_checkpoint_dir}", exist_ok=True)
model_path = os.path.join(f"models\{model_checkpoint_dir}", "fno_eigen.pth")

# Create plotting directory (if it doesn't exist yet.) 

os.makedirs(f"training_plots\{model_checkpoint_dir}", exist_ok=True)
plotting_path = f"training_plots\{model_checkpoint_dir}"

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

data = np.load(NPZ_PATH)
print(f"Loaded dataset from '{NPZ_PATH}'")
print(f"  Keys    : {list(data.keys())}")
print(f"  Samples : {data['gaussian_forward_real'].shape[0]}")
print(f"  Grid    : {data['gaussian_forward_real'].shape[1:]} (H x W)")

# ---------------------------------------------------------------------------
# 2. Build single-step predictor pairs
# ---------------------------------------------------------------------------

def normalize_11(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to the range [-1, +1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:   # constant array — avoid division by zero
        return np.zeros_like(arr)
    return 2.0 * (arr - lo) / (hi - lo) - 1.0


def prepare_pairs(data: np.lib.npyio.NpzFile, n_eigenmodes: int = N_EIGENMODES):
    """
    For each consecutive pair of timesteps (t, t+1) and each eigenmode k:
      Input  : [eigenmode_k_real_t,        eigenmode_k_imag_t,
                gaussian_forward_real_t+1,  gaussian_forward_imag_t+1,
                gaussian_reversed_real_t+1, gaussian_reversed_imag_t+1]  → (H, W, 6)
      Target : [eigenmode_k_real_{t+1}, eigenmode_k_imag_{t+1}]          → (H, W, 2)

    Samples are stored in timestep-major order — row = t * n_eigenmodes + (k-1) —
    so that a simple head/tail train/test slice yields a proper temporal holdout.
    Gaussian channels are normalised to [-1, +1] before packing.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile  (or equivalent dict-like)
        Arrays of shape (N_samples, H, W) per key.

    Returns
    -------
    X : torch.Tensor  shape ((N-1)*n_eigenmodes, H, W, 6)
    Y : torch.Tensor  shape ((N-1)*n_eigenmodes, H, W, 2)
    """
    N_samples = data["gaussian_forward_real"].shape[0]
    H, W      = data["gaussian_forward_real"].shape[1:]
    N_pairs   = N_samples - 1
    total     = N_pairs * n_eigenmodes

    # Pre-allocate output arrays
    X = np.empty((total, H, W, 6), dtype=np.float32)
    Y = np.empty((total, H, W, 2), dtype=np.float32)

    # Load and normalise Gaussian arrays once — shared across all eigenmodes
    gfwd_real = normalize_11(data["gaussian_forward_real"][1:].astype(np.float32))
    gfwd_imag = normalize_11(data["gaussian_forward_imag"][1:].astype(np.float32))
    grev_real = normalize_11(data["gaussian_reversed_real"][1:].astype(np.float32))
    grev_imag = normalize_11(data["gaussian_reversed_imag"][1:].astype(np.float32))

    # Load all eigenmode arrays up front
    em = {}
    for k in range(1, n_eigenmodes + 1):
        em[k] = (data[f"eigenmode_{k}_real"].astype(np.float32),
                 data[f"eigenmode_{k}_imag"].astype(np.float32))

    # Fill in timestep-major order: row = t * n_eigenmodes + (k-1)
    for t in range(N_pairs):
        for k in range(1, n_eigenmodes + 1):
            row = t * n_eigenmodes + (k - 1)
            em_real, em_imag = em[k]

            X[row, ..., 0] = em_real[t]       # eigenmode k at t
            X[row, ..., 1] = em_imag[t]
            X[row, ..., 2] = gfwd_real[t]     # Gaussian forward at t+1  (normalised)
            X[row, ..., 3] = gfwd_imag[t]
            X[row, ..., 4] = grev_real[t]     # Gaussian reversed at t+1 (normalised)
            X[row, ..., 5] = grev_imag[t]

            Y[row, ..., 0] = em_real[t + 1]   # eigenmode k at t+1
            Y[row, ..., 1] = em_imag[t + 1]

        if (t + 1) % 50 == 0 or (t + 1) == N_pairs:
            print(f"  Processed {t+1}/{N_pairs} timestep pairs ...")

    return torch.from_numpy(X), torch.from_numpy(Y)


X, Y = prepare_pairs(data)

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_loss_curves(
    train_mse_history: list,
    test_rel_l2_history: list,
    plots_dir: str = "training_plots",
    filename: str = "training_curves.png",
) -> None:
    """
    Plot training MSE and validation relative-L2 loss against epoch number,
    save the figure to `plots_dir/filename`, and display it.

    Parameters
    ----------
    train_mse_history    : per-epoch training MSE values.
    test_rel_l2_history  : per-epoch validation relative-L2 values.
    plots_dir            : directory in which to save the figure.
    filename             : output filename (PNG).
    """
    epochs_range = range(1, len(train_mse_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, train_mse_history, color='steelblue', linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Training MSE")
    axes[0].grid(True)

    axes[1].plot(epochs_range, test_rel_l2_history, color='tomato', linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative L2 (%)")
    axes[1].set_title("Validation Mean Relative L2 Loss")
    axes[1].grid(True)

    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Circular aperture helper
# ---------------------------------------------------------------------------

def torch_circ_aperture(field_batch: torch.Tensor, size: float, lensSize: float) -> torch.Tensor:
    """
    Applies a circular aperture to a batch of complex fields represented
    as real/imaginary channels.

    Parameters
    ----------
    field_batch : torch.Tensor  shape (batch, H, W, C)
                  where C=2 holds [real, imag] channels
    size        : float — physical side length of the window (metres)
    lensSize    : float — aperture radius (metres)

    Returns
    -------
    torch.Tensor — same shape as field_batch, zeroed outside the aperture
    """
    _, H, W, _ = field_batch.shape
    dev = field_batch.device

    x = torch.linspace(-size / 2, size / 2, W, device=dev)
    y = torch.linspace(-size / 2, size / 2, H, device=dev)
    yy, xx = torch.meshgrid(y, x, indexing="ij")   # (H, W)

    mask = (xx**2 + yy**2 <= lensSize**2).float()  # (H, W)
    mask = mask.unsqueeze(0).unsqueeze(-1)           # (1, H, W, 1)

    return field_batch * mask


# ---------------------------------------------------------------------------
# 3. Train / test split and DataLoaders
# ---------------------------------------------------------------------------

# If it is not pre-specified in the configuration file, then the total number of training samples, n_total, reflects the actual number of training samples in the training set.

cnfg.setdefault("n_total", len(X))

n_total = cnfg["n_total"]
n_train = int(n_total*train_test_split) # so if train_test_split is 0.90, then 90% of the dataset is used for training, while the rest is used for validation

x_train, x_test = X[:n_train], X[n_train:]
y_train, y_test = Y[:n_train], Y[n_train:]

N = x_train.shape[2]   # spatial grid size 

training_set = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
testing_set  = DataLoader(TensorDataset(x_test,  y_test),  batch_size=batch_size, shuffle=False)

# Report the train/validation size
print(f" Total number of training samples: {n_total}")
print(f" Number of training set batches: {len(training_set)} ")
print(f" Number of validation set batches: {len(testing_set)} ")

# --------------------------------------------------------------------------- #
# 4. Instantiate the FNO                                                      #
# --------------------------------------------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

fno_architecture = {
    "modes1":      modes1,
    "modes2":      modes2,
    "width":       width,
    "n_layers":    n_layers,
    "retrain_fno": RETRAIN_FNO,
    "input_chans": INPUT_CHANS,
    "output_chans": OUTPUT_CHANS,
}

model = FNO2d(fno_architecture, device=device)

optim     = Adam(model.parameters(), lr=learn_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)

if use_mse_loss:
    mse_loss = torch.nn.MSELoss()
else:
    freq_weights = make_freq_weights(N, k_cutoff_frac=0.99, alpha=2.0, device=device)

# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------

n_train_batches = len(training_set)
n_test_batches  = len(testing_set)

train_mse_history    = []
test_rel_l2_history  = []

# Early stopping state
best_val_loss     = float('inf')
patience_counter  = 0

for epoch in range(epochs):

    # --- Training ---
    model.train()
    train_mse = 0.0

    for step, (input_batch, output_batch) in enumerate(training_set):
        input_batch  = input_batch.to(device)
        output_batch = output_batch.to(device)

        optim.zero_grad()
        output_pred_batch = torch_circ_aperture(model(input_batch), size=size, lensSize=lensSize)   # (batch, H, W, 2)

        if use_mse_loss:
            loss_f = mse_loss(output_pred_batch, output_batch)
        else:
            loss_f = spectral_relative_l2(output_pred_batch, output_batch, freq_weights)

        loss_f.backward()
        optim.step()
        train_mse += loss_f.item()

    train_mse /= n_train_batches
    print("Training batches computed")

    scheduler.step()

    # --- Evaluation ---
    with torch.no_grad():
        model.eval()
        test_relative_l2 = 0.0

        for step, (input_batch, output_batch) in enumerate(testing_set):
            input_batch  = input_batch.to(device)
            output_batch = output_batch.to(device)
            output_pred_batch = torch_circ_aperture(model(input_batch), size=size, lensSize=lensSize)   # (batch, H, W, 2)

            if use_mse_loss:
                loss_f = (
                    torch.mean((output_pred_batch - output_batch) ** 2)
                    / torch.mean(output_batch ** 2)
                ) ** 0.5 * 100
            else:
                loss_f = spectral_relative_l2(output_pred_batch, output_batch, freq_weights)

            test_relative_l2 += loss_f.item()

        test_relative_l2 /= n_test_batches

    train_mse_history.append(train_mse)
    test_rel_l2_history.append(test_relative_l2)

    # Plot the train and test loss iteratively with each epoch
    
    plot_loss_curves(train_mse_history, test_rel_l2_history, plots_dir=plotting_path)

    if epoch % freq_print == 0:
        print(
            f"========= Epoch {epoch+1}/{epochs} Summary ========= "
            f"Train MSE: {train_mse:.6f} | "
            f"Mean Relative L2 Test: {test_relative_l2:.4f}%\n"
        )
    
    # Lastly, check the early stopping criterion: if the validation loss has not
    # improved by at least min_delta for stop_criterion_epochs consecutive epochs, stop.

    if test_relative_l2 < best_val_loss - min_delta:
        best_val_loss    = test_relative_l2
        patience_counter = 0

        # Save model checkpoint

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "fno_architecture": fno_architecture,
            "epochs": epochs,
            "train_mse_history": train_mse_history,
            "test_rel_l2_history": test_rel_l2_history,
        }, model_path)

        print(f"Model saved to {model_path}")

    else:
        patience_counter += 1
        print(f"  [Early stopping] No improvement for {patience_counter}/{stop_criterion_epochs} epoch(s). Best val loss: {best_val_loss:.6f}")

    if patience_counter >= stop_criterion_epochs:
        print(f"\n The validation loss has not improved for {stop_criterion_epochs} consecutive epochs.")
        print(f" Activating early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}\n")
        break

