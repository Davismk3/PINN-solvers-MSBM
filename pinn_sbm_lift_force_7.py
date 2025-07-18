import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# Script for predicting particle volume fraction (phi) from fluid velocity profile (Ux) using a PINN.
# Key improvements:
# - Total loss and the individual MSEs are visualized over time, which is visually interesting and has also helped
#   with debugging by better visualizing where the issues may be.
# - A scheduler has been added with the intended function of reducing the model's learning rate over time, although
#   this may not be necessary.
# - The loss wieghts are now learnable, and all are initially the value 1.0. This removes the need for trial-and-error
#   for both determining and initializing the loss weights.
# - The MSEs in the loss function are normalized differently. MSEs that start off small and only plateau from there 
#   will hinder convergence if weighted too much.
# Notes:
# - ϕ_trial(y) is enforced to be zero at the walls, but removing only slows convergence, and does not prevent it, so 
#   this can be safely removed for problems with non-zero values at the walls.
# - The current PINN parameters seem to produce the best results for different scenarios. 
# References:
# - pytorch.org has been the primary reference.
# - Fourier features from Tancik et al.
# - Learned Loss Weights from McClenny & Braga-Neto. 
# -----------------------------------------------------------------------------

# Controls & Hyperparameters --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)
np.random.seed(0)
loss_history = []
sym_mse_history = []
mass_mse_history = []
jdiv_mse_history = []
sigxy_mse_history = []
sigyy_mse_history = []

saved_models = Path("/Users/michaeldavis/Desktop/Python/SBM/Final/saved_models")  # saved models directory path
data_files = Path("/Users/michaeldavis/Desktop/Variable_width/H_50_micrometers")  # data directory path
Ux_saved = "Ux_trial_MSBM_Channel_width_50um_beta_1_2_phi_0_1"  # Ux model name
ϕ_saved = "phi_trial_MSBM_Channel_width_50um_beta_1_2_phi_0_1"  # phi model name
data_file = "Channel_width_50um_beta_1_2_phi_0_1.csv"  # data file
Ux_saved_path = saved_models / Ux_saved
ϕ_saved_path = saved_models / ϕ_saved
data_path = data_files / data_file
df = pd.read_csv(data_path)

# Training parameters
joint_training = False # Train Ux and phi jointly (recommended for reliability)
use_phi_data_in_loss = False  # Set to False for prediction without using phi data in training (only physics)
animate = True  # Plot every 10 epochs?
scope = 30  # for visualizing the lift force

# PINN parameters 
NEURONS = 64  # Increased for better capacity
EPOCHS_ADAM = 5000
EPOCHS_LBFGS = 200
LEARNING_RATE_ADAM = 1e-3
LEARNING_RATE_LBFGS = 1e-3
LEARNING_RATE_WEIGHTS = 1e-2  # Separate LR for loss weights, often higher to adapt faster
N_PTS = 500  # More collocation points
BUFFER = 1e-6  # for loss function
FOURIER_SCALE_Ux = 0.2 
FOURIER_SCALE_ϕ = 5.0 
H0 = 1e-12  # control parameter to avoid division by zero in lift force term
SCHEDULER_PATIENCE = 50  # Epochs to wait for loss improvement before reducing LR
SCHEDULER_FACTOR = 0.90  # Factor to multiply LR by when reducing (e.g., 0.5 halves it)
SCHEDULER_MIN_LR = 1e-6  # Minimum LR to stop reducing below this

# Physical parameters (match OpenFOAM)
p = df['p'].values.mean()  # steady state pressure (Pa) 
Ux_max = df['U_0'].values.max()  # max steady state velocity (m/s)
ϕ_average = df['c'].values.mean() # average ϕ (dimensionless)
H = 50e-6  # channel height (m)
ρ = 1190  # solvent density (Kg/m³)
η = 0.48  # dynamic viscosity (Pa·s)
η0 = η / ρ # kinematic viscosity (m²/s)
Kn = 0.75  # fitting parameter (dimensionless)
λ2 = 0.8  # fitting parameter (dimensionless)
λ3 = 0.5  # fitting parameter (dimensionless)
α = 4.0  # fitting parameter α ∈ [2, 5] (dimensionless)
a = 2.82e-6 # particle radius (m)
ϕ_max = 0.5  # max ϕ (dimensionless)
ε = a / ((H / 2)**2)  # non-local shear-rate coefficient (1/m)
β = 1.2  # power-law coefficient 
frv = 1.2  # function of the reduced volume

# Data tensors
y_data = torch.tensor(2.0 * (df['y'].values / H) - 1.0, dtype=torch.float32, device=device).unsqueeze(1)
Ux_data = torch.tensor(df['U_0'].values / Ux_max, dtype=torch.float32, device=device).unsqueeze(1)
ϕ_data = torch.tensor(df['c'].values, dtype=torch.float32, device=device).unsqueeze(1) if 'c' in df.columns else None

# Collocation points and trial functions 
y_uniform = torch.linspace(-1.0, 1.0, N_PTS, device=device).unsqueeze(1).requires_grad_(True)  # For mass conservation
Ux_trial = lambda y: PINN_Ux(torch.cat([y], dim=1))[:,0:1] * (1 + y) * (1 - y)  # torch.Size([y, 1]) | normalized 
ϕ_trial = lambda y: ϕ_max * torch.sigmoid(PINN_ϕ(y)[:,0:1]) * (1 + y) * (1 - y)  # torch.Size([y, 1])

# PINN ------------------------------------------------------------------------
# Fourier features from Tancik et al. 
class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size=64, scale=1):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale)

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

PINN_Ux = NN(neurons=NEURONS, trig_scale=FOURIER_SCALE_Ux).to(device)
PINN_ϕ = NN(neurons=NEURONS, trig_scale=FOURIER_SCALE_ϕ).to(device)

# Learned Loss Weights --------------------------------------------------------
# Learned Loss Weights from McClenny & Braga-Neto 
class LossScalars(nn.Module):
    def __init__(self, initial_values=(1.0, 1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.weights = nn.Parameter((torch.tensor(initial_values, device=device)))

    def forward(self):
        return torch.nn.functional.softplus(self.weights)

weights = LossScalars().to(device)

# Loss Functions --------------------------------------------------------------
def ϕ_physics_loss(y):
    ystar = y  # already normalized 
    Uxstar = Ux_trial(ystar)  # already normalized 
    ϕ = ϕ_trial(ystar)
    A = 2 * a / H
    pstar = p * H / (2 * η0 * Ux_max)
    zero = torch.zeros_like(ystar)  # torch.Size([y, 1])

    # Normal stress viscosity (ηₙ(ϕ))
    def ηN(ϕ):
        return Kn * (ϕ/ϕ_max)**2 * (1 - ϕ/ϕ_max)**(-2)  # torch.Size([y, 1]), a scalar for each y

    # Shear viscosity of the particle phase (ηₚ(ϕ))
    def ηp(ϕ):
        ηs = (1 - ϕ/ϕ_max)**(-2)
        return ηs - 1  # torch.Size([y, 1]), a scalar for each y

    # Sedimentation hinderence function for mobility of particle phase (f(ϕ))
    def f(ϕ):
        return (1 - ϕ/ϕ_max) * (1 - ϕ)**(α - 1)  # torch.Size([y, 1]), a scalar for each y

    # Gradient of the velocity field (∇U)
    dUxstar_dystar = torch.autograd.grad(Uxstar, ystar, torch.ones_like(Uxstar), create_graph=True)[0]  # torch.Size([y, 1])
    Ustar_gradient = torch.stack([
        torch.cat([zero, dUxstar_dystar, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Strain rate tensor (E)
    Estar = 0.5 * (Ustar_gradient + Ustar_gradient.transpose(1, 2))  # torch.Size([y, 3, 3]), a matrix for each y

    # Shear rate tensor (γ̇)
    γ̇star = torch.sqrt(2 * torch.sum(Estar * Estar, dim=(1, 2))).unsqueeze(1)  # torch.Size([y, 1])

    # Lift force (L)
    γ̇ = γ̇star * 2 * Ux_max / H  # dimensionalize for calculating it
    left_wall = 3 * η0 * γ̇ / (4 * torch.pi * ((H/2)*(ystar + 1) + H0)**β) * frv
    right_wall = 3 * η0 * γ̇ / (4 * torch.pi * ((H/2)*(1 - ystar) + H0)**β) * frv
    scale_L = H / (2 * η0 * Ux_max)  # nondimensionalize after calculating it
    L = torch.stack([
        torch.cat([zero], dim=1),
        torch.cat([scale_L * (left_wall - right_wall)], dim=1),
        torch.cat([zero], dim=1)
    ], dim=1)
    lift_force = torch.cat([(left_wall - right_wall)], dim=1).detach().numpy()[scope:-scope, 0]  # does not include singularities at walls


    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, λ2, 0.0], [0.0, 0.0, λ3]], device=device).repeat(y.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Non-local shear rate tensor
    γ̇NLstar = ε * H / 2

    # Particle normal stress diagonal tensor (Σₙₙᵖ)
    Σpnnstar = ηN(ϕ).view(-1, 1, 1) * (γ̇star.unsqueeze(1) + γ̇NLstar) * Q  # torch.Size([y, 3, 3]), a matrix for each y

    # Oriented particle stress tensor (Σᵖ)
    Σpstar = -Σpnnstar + (2 * ηp(ϕ).view(-1, 1, 1) * Estar)  # torch.Size([y, 3, 3]), a matrix for each y

    # Divergence of oriented particle stress tensor (∇⋅Σᵖ)
    dΣpxystar_dystar = torch.autograd.grad(Σpstar[:, 0, 1], ystar, torch.ones_like(Σpstar[:, 0, 1]), create_graph=True)[0]  # torch.Size([y, 1])
    dΣpyystar_dystar = torch.autograd.grad(Σpstar[:, 1, 1], ystar, torch.ones_like(Σpstar[:, 1, 1]), create_graph=True)[0]  # torch.Size([y, 1])
    Σpstar_divergence = torch.stack([
        torch.cat([zero + dΣpxystar_dystar + zero], dim=1),
        torch.cat([zero + dΣpyystar_dystar + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
    ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y

    # Migration flux (J)
    Jstar = - (2 * A**2 / 9) * f(ϕ).unsqueeze(1) * (Σpstar_divergence + ϕ.view(-1, 1, 1) * L)  # torch.Size([y, 3, 1])

    # Divergence of migration flux (∇⋅J)
    dJxstar_dxstar = dJzstar_dzstar = zero
    dJystar_dystar = torch.autograd.grad(Jstar[:, 1, 0], ystar, torch.ones_like(Jstar[:, 1, 0]), create_graph=True)[0]  # torch.Size([y, 1])
    Jstar_divergence = dJxstar_dxstar + dJystar_dystar + dJzstar_dzstar  # torch.Size([y, 1])

    # Identity matrix (I)
    I = torch.eye(3, device=device).repeat(y.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase stress (Σᶠ)
    Σfstar = - pstar * I + 2 * Estar

    # Total stress (Σ)
    Σstar = Σpstar + Σfstar

    # Suspension momentum balance (∇⋅Σ)
    dΣxystar_dystar = torch.autograd.grad(Σstar[:, 0, 1], ystar, torch.ones_like(Σstar[:, 0, 1]), create_graph=True)[0]
    dΣyystar_dystar = torch.autograd.grad(Σstar[:, 1, 1], ystar, torch.ones_like(Σstar[:, 1, 1]), create_graph=True)[0]

    # Enforce mass conservation for particles 
    mass_conservation = torch.mean(ϕ_trial(y_uniform)) - ϕ_average

    # Enforce a symmetrical solution
    symmetry = ϕ_trial(y) - ϕ_trial(-y)

    # Define wieghts 
    weight_symmetry, weight_mass, weight_J, weight_Σxy, weight_Σyy = weights()

    # Loss Terms 
    total_loss = (
        weight_symmetry * torch.mean(symmetry**2) / (torch.mean(symmetry**2).detach() + BUFFER) +  # normalization seems to help with convergence
        weight_mass * torch.mean(mass_conservation**2) / (torch.mean(mass_conservation**2).detach() + BUFFER) +  # normalization seems to help with convergence
        weight_J * torch.mean(Jstar_divergence**2) / (torch.mean(Jstar_divergence**2).detach() + BUFFER) +  # normalization seems to help with convergence
        weight_Σxy * torch.mean(dΣxystar_dystar**2) +  # not normalized because it starts off already small and only plateaus from there
        weight_Σyy * torch.mean(dΣyystar_dystar**2)  # not normalized because it starts off already small and only plateaus from there
    )
    
    individual_loss = (
        torch.mean(symmetry**2).item(), 
        torch.mean(mass_conservation**2).item(), 
        torch.mean(Jstar_divergence**2).item(), 
        torch.mean(dΣxystar_dystar**2).item(), 
        torch.mean(dΣyystar_dystar**2).item()
          )
    
    return total_loss, individual_loss, lift_force 

# Visualize -------------------------------------------------------------------
def visualize():
    with torch.no_grad():
        y_plot = torch.linspace(-1.0, 1.0, N_PTS, device=device).unsqueeze(1)
        y_plot_dim = ((y_plot + 1.0) / 2.0 * H).cpu().numpy()
        Ux_pinn = (Ux_trial(y_plot) * Ux_max).cpu().numpy()
        phi_pinn = ϕ_trial(y_plot).cpu().numpy()

        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        ax_ux = axs[0][0]
        ax_phi = axs[0][1]
        ax_total_loss = axs[0][2]
        ax_indiv_loss = axs[1][0]
        ax_lift_force = axs[1][1]

        # Ux
        ax_ux.plot(((y_data + 1)/2 * H).cpu(), (Ux_data * Ux_max).cpu(), 'ko', markersize=3, label='Data')
        ax_ux.plot(y_plot_dim, Ux_pinn, 'b-', label='PINN')
        ax_ux.set_xlabel('y [m]')
        ax_ux.set_ylabel('Ux [m/s]')
        ax_ux.legend()
        ax_ux.grid(True)

        # Phi
        if ϕ_data is not None:
            ax_phi.plot(((y_data + 1)/2 * H).cpu(), ϕ_data.cpu(), 'ko', markersize=3, label='Data')
        ax_phi.plot(y_plot_dim, phi_pinn, 'r-', label='PINN')
        ax_phi.set_xlabel('y [m]')
        ax_phi.set_ylabel('phi')
        ax_phi.legend()
        ax_phi.grid(True)

        # Total Loss
        if loss_history:
            ax_total_loss.semilogy(loss_history, 'g-', label='Total Loss')
            ax_total_loss.set_xlabel('Epoch')
            ax_total_loss.set_ylabel('Loss')
            ax_total_loss.legend()
            ax_total_loss.grid(True)

        # Individual MSE Terms
        epochs = range(len(sym_mse_history))
        if sym_mse_history:
            ax_indiv_loss.semilogy(epochs, sym_mse_history, label='Symmetry MSE')
            ax_indiv_loss.semilogy(epochs, mass_mse_history, label='Mass Cons. MSE')
            ax_indiv_loss.semilogy(epochs, jdiv_mse_history, label='J Div. MSE')
            ax_indiv_loss.semilogy(epochs, sigxy_mse_history, label='dΣxy/dy MSE')
            ax_indiv_loss.semilogy(epochs, sigyy_mse_history, label='dΣyy/dy MSE')
            ax_indiv_loss.set_xlabel('Epoch')
            ax_indiv_loss.set_ylabel('Individual MSE')
            ax_indiv_loss.legend()
            ax_indiv_loss.grid(True)

        # Lift force
        ax_lift_force.plot(y_plot_dim[scope:-scope], lift_force, 'y-', label='Lift Force')
        ax_lift_force.set_xlabel('y [m]')
        ax_lift_force.set_ylabel('y [Newtons]')
        ax_lift_force.legend()
        ax_lift_force.grid(True)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)

# Training --------------------------------------------------------------------
nn_params = list(PINN_Ux.parameters()) + list(PINN_ϕ.parameters())
if not joint_training:
    nn_params = list(PINN_ϕ.parameters())

    for p in PINN_Ux.parameters():
        p.requires_grad_(False)
    PINN_Ux.load_state_dict(torch.load(f=Ux_saved_path))
    PINN_Ux.eval()

optimizer_adam = torch.optim.Adam([
    {'params': nn_params, 'lr': LEARNING_RATE_ADAM},
    {'params': weights.parameters(), 'lr': LEARNING_RATE_WEIGHTS},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adam, 
    factor=SCHEDULER_FACTOR, 
    patience=SCHEDULER_PATIENCE, 
    min_lr=SCHEDULER_MIN_LR, 
    verbose=True
)

for epoch in range(EPOCHS_ADAM):
    optimizer_adam.zero_grad()
    y = y_uniform
    total_loss, individual_loss, lift_force = ϕ_physics_loss(y)
    loss = total_loss
    loss.backward()
    optimizer_adam.step()
    scheduler.step(loss)
    loss_history.append(loss.item())
    sym_mse_history.append(individual_loss[0])
    mass_mse_history.append(individual_loss[1])
    jdiv_mse_history.append(individual_loss[2])
    sigxy_mse_history.append(individual_loss[3])
    sigyy_mse_history.append(individual_loss[4])
    print(f"Adam epoch: {epoch} | Loss: {loss.item()} | Weights: {weights().detach().cpu().numpy()} | LR: {optimizer_adam.param_groups[0]['lr']}")

    if epoch % 50 == 0:
        visualize()

optimizer_lbfgs = torch.optim.LBFGS(nn_params, lr=LEARNING_RATE_LBFGS)

def closure():
    optimizer_lbfgs.zero_grad()
    total_loss, individual_loss, lift_force = ϕ_physics_loss(y_uniform)
    loss = total_loss
    loss.backward()
    return loss

for epoch in range(EPOCHS_LBFGS):
    optimizer_lbfgs.step(closure)
    total_loss, individual_loss, lift_force = ϕ_physics_loss(y_uniform)
    loss = total_loss
    loss_history.append(loss.item())
    sym_mse_history.append(individual_loss[0])
    mass_mse_history.append(individual_loss[1])
    jdiv_mse_history.append(individual_loss[2])
    sigxy_mse_history.append(individual_loss[3])
    sigyy_mse_history.append(individual_loss[4])
    print(f"LBFGS epoch: {epoch} | Loss: {loss.item()}")
    visualize()

# Save models
saved_models.mkdir(parents=True, exist_ok=True)
# torch.save(PINN_Ux.state_dict(), Ux_saved_path)
torch.save(PINN_ϕ.state_dict(), ϕ_saved_path)
