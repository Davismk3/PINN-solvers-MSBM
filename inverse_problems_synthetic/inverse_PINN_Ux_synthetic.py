import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------------- 
# SOLVES THE INVERSE PROBLEM FOR Ux FOR SYNTHETIC DATA
# ----------------------------------------------------------------------------- 

# Controls & Hyperparameters --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)
np.random.seed(0)

# Model controls
data_file_1 = True  # which example data file do you want to try? one or two? if one, one = True. if two, one = False
save_images = False  # save images during training for making an animation later
use_scheduler = True  # if True, the learning rates will decrease over time as the loss plateaus

# Define the paths for data, save, and visualization
base_path = Path(__file__).parent
data_path = base_path / "synthetic_data"
save_path = base_path / "saved_models"
visualization_path = base_path / "saved_visuals"

# Define the model names and data file
Ux_model_saved_name = "synthetic_saved_Ux_model_example_1" if data_file_1 else "synthetic_saved_Ux_model_example_2"
data_file_name = "synthetic_data_example_1.csv" if data_file_1 else "synthetic_data_example_2.csv"

# Load the data
df = pd.read_csv(data_path / data_file_name)

# PINN parameters 
Ux_NEURONS = 64  # hidden‑layer width
Ux_LAYERS = 3
EPOCHS_ADAM = 10000  # iterations for faster optimizer
EPOCHS_LBFGS = 100  # iterations for better optimizer
LEARNING_RATE = 1e-2
COLLOCATION_PTS = 100 # collocation points

# Scheduler parameters
SCHEDULER_PATIENCE = 50  # Epochs to wait for loss improvement before reducing LR
SCHEDULER_FACTOR = 0.70  # Factor to multiply LR by when reducing (e.g., 0.5 halves it)
SCHEDULER_MIN_LR = 1e-6  # Minimum LR to stop reducing below this

# Physical parameters
Ux_max = torch.tensor(df['U_0'].values.max(), device=device)  # max steady state velocity (m/s)
H = torch.tensor(25e-6, device=device) if data_file_1 else torch.tensor(50e-6, device=device)  # channel height (m)

# Data tensors
y_data = 2.0 * torch.tensor(df['y'].values, dtype=torch.float32, device=device).unsqueeze(1) / H - 1.0
Ux_data = torch.tensor(df['U_0'].values, dtype=torch.float32, device=device).unsqueeze(1) / Ux_max

# Trial function
Ux_trial = lambda y: Ux_PINN(torch.cat([y.to(device)], dim=1))[:,0:1] * (1 + y.to(device)) * (1 - y.to(device))  # torch.Size([y, 1]) | normalized 

# Loss history for plotting
class LossHistory:
    def __init__(self):
        self.total = []

    def append(self, ℒ):
        self.total.append(ℒ.item())

# PINN ------------------------------------------------------------------------
# Compile Ux_PINN layers 
layers = [nn.Linear(1, Ux_NEURONS), nn.Tanh()]
for layer in range(Ux_LAYERS): layers.extend([nn.Linear(Ux_NEURONS, Ux_NEURONS), nn.Tanh()])
layers.append(nn.Linear(Ux_NEURONS, 1))

# Create the Ux_PINN
Ux_PINN = nn.Sequential(*layers).to(device)

# Loss ------------------------------------------------------------------------
def Ux_data_loss():
    Ux_pred = Ux_trial(y_data)
    ℒ = torch.sqrt(torch.mean((Ux_pred - Ux_data)**2))
    return ℒ

# Visualize -------------------------------------------------------------------
def visualize():
    with torch.no_grad():
        y_plot = torch.linspace(-1.0, 1.0, COLLOCATION_PTS, device=device).unsqueeze(1)
        y_plot_dim = ((y_plot + 1.0) / 2.0 * H).cpu().numpy()
        Ux_pinn = (Ux_trial(y_plot) * Ux_max).cpu().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))
        ax_ux = axs[0]
        ax_total_loss = axs[1]

        # Ux
        ax_ux.plot(((y_data + 1)/2 * H).cpu(), (Ux_data * Ux_max).cpu(), 'ko', markersize=3, label='Data')
        ax_ux.plot(y_plot_dim, Ux_pinn, 'b-', label='PINN')
        ax_ux.set_xlabel('y [m]')
        ax_ux.set_ylabel('Ux [m/s]')
        ax_ux.legend()
        ax_ux.grid(True)

        # Total Loss
        if loss_history.total:
            ax_total_loss.semilogy(loss_history.total, 'g-', label='Total Loss')
            ax_total_loss.set_xlabel('Epoch')
            ax_total_loss.set_ylabel('Loss')
            ax_total_loss.legend()
            ax_total_loss.grid(True)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)

# Training Loop ---------------------------------------------------------------
# Define parameters
Ux_PINN_parameters = list(Ux_PINN.parameters())

# Define optimizers
Ux_PINN_optimizer_Adam = torch.optim.Adam(Ux_PINN_parameters, lr=LEARNING_RATE)
Ux_PINN_optimizer_LBFGS = torch.optim.LBFGS(list(Ux_PINN.parameters()), lr=LEARNING_RATE)

# Define schedulers (ReduceLROnPlateau type does exactly that)
Ux_PINN_scheduler_Adam = torch.optim.lr_scheduler.ReduceLROnPlateau(Ux_PINN_optimizer_Adam, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
Ux_PINN_scheduler_LBFGS = torch.optim.lr_scheduler.ReduceLROnPlateau(Ux_PINN_optimizer_LBFGS, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)

# Call the loss history class
loss_history = LossHistory()

# ADAM training loop 
for epoch in range(EPOCHS_ADAM):
    Ux_PINN_optimizer_Adam.zero_grad() 

    # Forward & Backward passes
    ℒ = Ux_data_loss()  # forward pass to compute loss
    ℒ.backward()  # backward pass to compute gradients

    # Gradient descent & scheduler update
    Ux_PINN_optimizer_Adam.step()  # gradient descent updating PINN parameters
    Ux_PINN_scheduler_Adam.step(ℒ.item()) if use_scheduler else None

    # Update loss history for plotting
    loss_history.append(ℒ)

    # Visuals
    print(f"Epoch: {epoch} | Loss: {ℒ.item()} | Ux Lr: {Ux_PINN_scheduler_Adam.get_last_lr()}")
    if epoch % 50 == 0:
        visualize()

# LBFGS closure function
def closure():
    Ux_PINN_optimizer_LBFGS.zero_grad()
    ℒ = Ux_data_loss()
    ℒ.backward()
    return ℒ

# LBFGS training loop 
for epoch in range(EPOCHS_LBFGS):
    Ux_PINN_optimizer_LBFGS.step(closure)
    ℒ = Ux_data_loss()
    loss_history.append(ℒ)
    Ux_PINN_scheduler_LBFGS.step(ℒ.item()) if use_scheduler else None

    print(f"Epoch: {epoch} | Loss: {ℒ.item()} | Ux Lr: {Ux_PINN_scheduler_LBFGS.get_last_lr()}")
    if epoch % 10 == 0:
        visualize()

# Save ϕ model when finished
Ux_saved_path = save_path / Ux_model_saved_name
Path(save_path).mkdir(parents=True, exist_ok=True)
torch.save(Ux_PINN.state_dict(), Ux_saved_path)
