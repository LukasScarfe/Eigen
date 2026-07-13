"""
Sanity-check experiment for the PyTorch UNet implementation (unet.py).

Goal: NOT a formal predictive benchmark, just a smoke test proving that:
  1. A batch of inputs can be passed through the network (forward pass works,
     shapes are correct, outputs land in the expected ranges).
  2. The network can actually learn: gradients flow and the training loss on a
     small synthetic input/output pair decreases substantially over a handful
     of optimization steps (i.e. it can overfit a tiny dataset).

Run with:
    python test_unet_learning.py
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from unet import UNet


def test_forward_shapes_and_range():
    torch.manual_seed(0)
    num_pixel = 64
    model = UNet(num_pixel=num_pixel, nnType=0, kernelSize=3, dropRate=0.1,
                 layers=1, sixMeasure=False)
    model.eval()

    x = torch.randn(4, 5, num_pixel, num_pixel)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (4, 2, num_pixel, num_pixel), f"Unexpected output shape: {y.shape}"

    # Channel 0 (intensity) should be in [0, 1], channel 1 (phase) in [0, 2*pi]
    assert torch.all(y[:, 0] >= 0) and torch.all(y[:, 0] <= 1 + 1e-4)
    assert torch.all(y[:, 1] >= 0) and torch.all(y[:, 1] <= 2 * torch.pi + 1e-4)

    print("[PASS] forward pass shapes/ranges look correct:", tuple(y.shape))


def test_overfit_tiny_batch():
    """
    Train the network on a single (input, target) pair for a small number of
    steps and confirm the loss decreases meaningfully. This demonstrates the
    model + optimizer + loss wiring all works end-to-end (gradients flow
    through every block, including skip connections and GroupNorm layers).
    """
    torch.manual_seed(0)
    num_pixel = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(num_pixel=num_pixel, nnType=0, kernelSize=3, dropRate=0.0,
                 layers=1, sixMeasure=False).to(device)
    model.train()

    # Synthetic input and a synthetic "ground truth" target within the model's
    # valid output range, so the network has a learnable target to chase.
    x = torch.randn(2, 5, num_pixel, num_pixel, device=device)
    target = torch.rand(2, 2, num_pixel, num_pixel, device=device)
    target[:, 0] *= 1.0
    target[:, 1] *= 2 * torch.pi

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    num_steps = 120
    for step in range(num_steps):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == num_steps - 1:
            print(f"  step {step:3d}  loss = {loss.item():.6f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    print(f"[INFO] initial loss = {initial_loss:.6f}, final loss = {final_loss:.6f}")

    assert final_loss < initial_loss * 0.5, (
        "Loss did not decrease enough; model may not be learning correctly "
        f"(initial={initial_loss:.6f}, final={final_loss:.6f})"
    )
    print("[PASS] model successfully overfits a tiny synthetic batch "
          f"(loss dropped from {initial_loss:.6f} to {final_loss:.6f})")

    plot_loss_curve(losses)


def plot_loss_curve(losses, save_path="unet_overfit_loss.png"):
    """Plot the training loss over steps and save it to disk for visual inspection."""
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
    plt.xlabel("Optimization step")
    plt.ylabel("MSE loss")
    plt.title("UNet overfitting a tiny synthetic batch")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] loss curve saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    test_forward_shapes_and_range()
    test_overfit_tiny_batch()
    print("\nAll checks passed.")
