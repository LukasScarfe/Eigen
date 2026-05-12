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



# Select configuration file according to parse index


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------



# Load yaml file

with open(f'train{ii}.yaml', 'r') as file:
    config = yaml.load(file, Loader=Loader)





DATASET_DIR  = "dataset"
N_SAMPLES    = 1000
N_EIGENMODES = 4

n_train    = 3600
batch_size = 10

# Training hyper-parameters
epochs     = 50
learn_rate = 0.001   # initial learning rate
step_size  = 12      # LR decay period (epochs)
gamma      = 0.5     # LR decay factor

# FNO architecture
modes1       = 36    # spectral modes along dimension 1
modes2       = 36    # spectral modes along dimension 2
width        = 64    # channel width inside the FNO
n_layers     = 1     # number of Fourier layers
retrain_fno  = 42    # random seed for weight initialisation
input_chans  = 4
output_chans = 2

use_mse_loss = False  # False → spectral relative L2; True → standard MSE

freq_print = 1        # print summary every N epochs

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

def load_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def load_sample(sample_dir: str) -> dict:
    """Load all four eigenmodes and Gaussian propagation fields from a sample directory."""
    return {
        "eigenmode_1_real":       load_csv(os.path.join(sample_dir, "eigenmode_001_real.csv")),
        "eigenmode_1_imag":       load_csv(os.path.join(sample_dir, "eigenmode_001_imag.csv")),
        "eigenmode_2_real":       load_csv(os.path.join(sample_dir, "eigenmode_002_real.csv")),
        "eigenmode_2_imag":       load_csv(os.path.join(sample_dir, "eigenmode_002_imag.csv")),
        "eigenmode_3_real":       load_csv(os.path.join(sample_dir, "eigenmode_003_real.csv")),
        "eigenmode_3_imag":       load_csv(os.path.join(sample_dir, "eigenmode_003_imag.csv")),
        "eigenmode_4_real":       load_csv(os.path.join(sample_dir, "eigenmode_004_real.csv")),
        "eigenmode_4_imag":       load_csv(os.path.join(sample_dir, "eigenmode_004_imag.csv")),
        "gaussian_forward_real":  load_csv(os.path.join(sample_dir, "gaussian_prop_forward_real.csv")),
        "gaussian_forward_imag":  load_csv(os.path.join(sample_dir, "gaussian_prop_forward_imag.csv")),
        "gaussian_reversed_real": load_csv(os.path.join(sample_dir, "gaussian_prop_reversed_real.csv")),
        "gaussian_reversed_imag": load_csv(os.path.join(sample_dir, "gaussian_prop_reversed_imag.csv")),
    }


sample_dirs = sorted([
    os.path.join(DATASET_DIR, d)
    for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])[:N_SAMPLES]

dataset = [load_sample(d) for d in sample_dirs]

print(f"Loaded {len(dataset)} samples")

# ---------------------------------------------------------------------------
# 2. Build single-step predictor pairs
# ---------------------------------------------------------------------------

def prepare_pairs(dataset: list, n_eigenmodes: int = N_EIGENMODES):
    """
    For each consecutive pair of timesteps (t, t+1) and each eigenmode k:
      Input  : [eigenmode_k_real_t, eigenmode_k_imag_t,
                gaussian_forward_real_t, gaussian_forward_imag_t]  → (H, W, 4)
      Target : [eigenmode_k_real_{t+1}, eigenmode_k_imag_{t+1}]   → (H, W, 2)

    Returns
    -------
    X : torch.Tensor  shape ((N-1)*n_eigenmodes, H, W, 4)
    Y : torch.Tensor  shape ((N-1)*n_eigenmodes, H, W, 2)
    """
    X_list, Y_list = [], []

    for t in range(len(dataset) - 1):
        curr = dataset[t]
        nxt  = dataset[t + 1]

        for k in range(1, n_eigenmodes + 1):
            x = np.stack([
                curr[f"eigenmode_{k}_real"],
                curr[f"eigenmode_{k}_imag"],
                curr["gaussian_forward_real"],
                curr["gaussian_forward_imag"],
            ], axis=-1).astype(np.float32)   # (H, W, 4)

            y = np.stack([
                nxt[f"eigenmode_{k}_real"],
                nxt[f"eigenmode_{k}_imag"],
            ], axis=-1).astype(np.float32)   # (H, W, 2)

            X_list.append(x)
            Y_list.append(y)

    X = torch.tensor(np.stack(X_list, axis=0))
    Y = torch.tensor(np.stack(Y_list, axis=0))
    return X, Y


X, Y = prepare_pairs(dataset)

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
# 3. Train / test split and DataLoaders
# ---------------------------------------------------------------------------

x_train, x_test = X[:n_train], X[n_train:]
y_train, y_test = Y[:n_train], Y[n_train:]

N = x_train.shape[2]   # spatial grid size (used for freq_weights)

training_set = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
testing_set  = DataLoader(TensorDataset(x_test,  y_test),  batch_size=batch_size, shuffle=False)

# ---------------------------------------------------------------------------
# 4. Instantiate the FNO
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

fno_architecture = {
    "modes1":      modes1,
    "modes2":      modes2,
    "width":       width,
    "n_layers":    n_layers,
    "retrain_fno": retrain_fno,
    "input_chans": input_chans,
    "output_chans":output_chans,
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

for epoch in range(epochs):

    # --- Training ---
    model.train()
    train_mse = 0.0

    for step, (input_batch, output_batch) in enumerate(training_set):
        input_batch  = input_batch.to(device)
        output_batch = output_batch.to(device)

        optim.zero_grad()
        output_pred_batch = model(input_batch)   # (batch, H, W, 2)

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
            output_pred_batch = model(input_batch)   # (batch, H, W, 2)

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

    plot_loss_curves(train_mse_history, test_rel_l2_history)

    if epoch % freq_print == 0:
        print(
            f"========= Epoch {epoch+1}/{epochs} Summary ========= "
            f"Train MSE: {train_mse:.6f} | "
            f"Mean Relative L2 Test: {test_relative_l2:.4f}%\n"
        )
