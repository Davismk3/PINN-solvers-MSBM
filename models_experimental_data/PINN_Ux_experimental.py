import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# This script attempts to predict the particle volume fraction profile through 
# a channel from data of the fluid's velocity profile. 
# - 
# The parameters are set to reproduce a modified suspension balance model which 
# includes a lift force observed in blood flow.
# -
# Convergence to the correct profile depends greatly on the weights to the loss
# functions (BUFFER and AVG_WEIGHT), and also the trig_scaling for the 
# FourierFeatures layer. Also, convergence seems to depend drastically on
# the values of FOURIER_SCALE_ϕ and FOURIER_SCALE_Ux. Their current values allow 
# for a quick and proper convergence.
# -----------------------------------------------------------------------------

# clean up math
# make sure sensitive parameters match OpenFOAM simulations 
# revisit joint Ux and phi training

# Controls & Hyperparameters --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)
np.random.seed(0)
loss_history = []

# Define the paths for data, save, and visualization
base_path = "/Users/michaeldavis/Desktop/Desktop Storage/Machine Learning/Physics Informed Neural Networks (PINNs)/Modified Suspension Balance Project/"
data_path = base_path + "Data/MSBM/"
save_path = base_path + "Results/Saved Models/"
visualization_path = base_path + "Results/Visualizations/"

# Define the model names and data file
Ux_model_saved_name = "Ux_trial_MSBM_channel_height_50_um"
ϕ_model_saved_name = "phi_trial_MSBM_channel_height_50_um"
data_file_name = "channel_height_50_um.csv"

df = pd.read_csv(data_path + data_file_name)

# visuals 
animate = True  # plot results every 10 epochs?

# PINN parameters 
NEURONS = 64 # hidden‑layer width
EPOCHS_ADAM = 100000  # iterations for faster optimizer
EPOCHS_LBFGS = 1000  # iterations for better optimizer
LEARNING_RATE_ADAM = 1e-2
LEARNING_RATE_LBFGS = 1e-4
N_PTS = 1000 # collocation points
FOURIER_SCALE = 0.02  # larger values seem to yeild worse results and fail to allow ϕ convergence

# parameters
H = 50e-6  # channel height (m)

# Clean data
y_min = df['y'].values.min()
y_max = df['y'].values.max()
y_mid = (y_max + y_min) / 2
y_shifted = df['y'].values - y_mid
channel_scaling = np.max(np.abs(y_shifted))

# Data tensors
y_data = torch.tensor(y_shifted / channel_scaling, dtype=torch.float32, device=device).unsqueeze(1)
Ux_data = torch.tensor(df['U_0'].values / df['U_0'].values.max(), dtype=torch.float32, device=device).unsqueeze(1)  # torch.Size([1001, 1]) | [0, 1]

# trial function
Ux_trial = lambda y: PINN_Ux(torch.cat([y], dim=1))[:,0:1]  # torch.Size([y, 1]) | normalized 

# PINN ------------------------------------------------------------------------
# Fourier features from Tancik et al. 
class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size=64, scale=1):
        super().__init__()
        self.B = nn.Parameter(scale * torch.randn((in_features, mapping_size)), requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class NN(nn.Module):
    def __init__(self, neurons, trig_scale):
        super().__init__()
        self.net = nn.Sequential(
            FourierFeatures(1, mapping_size=neurons, scale=trig_scale),
            nn.Linear(2 * neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, neurons),
            nn.Tanh(),
            nn.Linear(neurons, 1),
        )
    def forward(self, x):
        return self.net(x)

PINN_Ux = NN(neurons=NEURONS, trig_scale=FOURIER_SCALE)

# Loss ------------------------------------------------------------------------
def Ux_data_loss(y): 
    return torch.mean((Ux_trial(y) - Ux_data)**2)

# Visualize -------------------------------------------------------------------
def visualize(true_values, predicted_values, label):
    with torch.no_grad():
        y_plot_data = ((y_data + 1.0) / 2.0 * H).squeeze().cpu().numpy()
        data_plot = true_values.squeeze().cpu().numpy()
        y_pinn = torch.linspace(-1.0, 1.0, N_PTS, device=device).unsqueeze(1)
        y_plot_pinn = ((y_pinn + 1.0) / 2.0 * H).squeeze().cpu().numpy()
        pinn_plot = predicted_values(y_pinn).squeeze().cpu().numpy()

    if animate:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

        ax_left.plot(y_plot_data, data_plot, 'ko',  markersize=3, label='Data')
        ax_left.plot(y_plot_pinn, pinn_plot, 'b-',  linewidth=2, label='PINN')
        ax_left.set_xlabel('y [m]')
        ax_left.set_ylabel(label)
        ax_left.set_title('Profile')
        ax_left.legend()
        ax_left.grid(True)

        if loss_history:
            ax_right.semilogy(loss_history, 'r-')
            ax_right.set_xlabel('Iteration')
            ax_right.set_ylabel('Total loss')
            ax_right.set_title('Convergence')
            ax_right.grid(True, which='both', ls='--', alpha=0.6)

        fig.suptitle("PINN fit & loss history", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)

# Optimizers ------------------------------------------------------------------
def Adam_Velocity(learning_rate, epochs):
    PINN_Ux.train()
    optimizer = torch.optim.Adam(PINN_Ux.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.5, patience=500, verbose=True)
    for epoch in range(epochs): 
        optimizer.zero_grad()
        loss = Ux_data_loss(y_data)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)    
        print("Adam epoch: ", epoch, " | Loss: ", loss.item())

        # if epoch % 10 == 0:
            # visualize(Ux_data * df['U_0'].values.max(), lambda y: Ux_trial(y) * torch.tensor(df['U_0'].max(), dtype=torch.float32, device=device), 'Ux (Velocity)')
    visualize(Ux_data * df['U_0'].values.max(), lambda y: Ux_trial(y) * torch.tensor(df['U_0'].max(), dtype=torch.float32, device=device), 'Ux (Velocity)')

def LBFGS_Velocity(learning_rate, epochs):
    optimizer = torch.optim.LBFGS(PINN_Ux.parameters(), lr=learning_rate)

    def closure():
        optimizer.zero_grad()
        Ux_data_loss(y_data).backward()
        return Ux_data_loss(y_data)

    for epoch in range(epochs):
        optimizer.step(closure)
        print("LBFGS epoch: ", epoch, " | Loss: ", Ux_data_loss(y_data).item())
    visualize(Ux_data * df['U_0'].values.max(), lambda y: Ux_trial(y) * torch.tensor(df['U_0'].max(), dtype=torch.float32, device=device), 'Ux (Velocity)')

# Training Loop ---------------------------------------------------------------
Adam_Velocity(epochs=1000, learning_rate=1e-3)
Adam_Velocity(epochs=2000, learning_rate=1e-4)
Adam_Velocity(epochs=2000, learning_rate=1e-5)
Adam_Velocity(epochs=1000, learning_rate=1e-6)
Adam_Velocity(epochs=1000, learning_rate=1e-7)
Adam_Velocity(epochs=1000, learning_rate=1e-8)
Adam_Velocity(epochs=1000, learning_rate=1e-9)
Adam_Velocity(epochs=1000, learning_rate=1e-10)
Adam_Velocity(epochs=1000, learning_rate=1e-11)
Adam_Velocity(epochs=1000, learning_rate=1e-12)

Ux_saved_path = save_path + Ux_model_saved_name
Path(save_path).mkdir(parents=True, exist_ok=True)
torch.save(obj=PINN_Ux.state_dict(), f=Ux_saved_path)