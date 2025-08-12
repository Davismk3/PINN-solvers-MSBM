import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# Script for predicting particle volume fraction (phi) from fluid velocity profile (Ux) using a PINN.
# -----------------------------------------------------------------------------

# Controls & Hyperparameters --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(0)
np.random.seed(0)

# Define the paths for data, save, and visualization
base_path = Path(__file__).parent
data_path = base_path / ""
save_path = base_path / ""
visualization_path = base_path / ""

# Define the model names and data file
Ux_model_saved_name = "synthetic_saved_Ux_model_example"
ϕ_model_saved_name = "synthetic_saved_phi_example"
data_file_name = "synthetic_data_example.csv"

# Load the data
df = pd.read_csv(data_path / data_file_name)

# PINN parameters 
NEURONS = 64  # Increased for better capacity
EPOCHS_ADAM = 1000
LEARNING_RATE_PINN = 1e-3
LEARNING_RATE_λ = 1e-2
N_PTS = 500  # More collocation points
FOURIER_SCALE_Ux = 0.2
FOURIER_SCALE_ϕ = 5.0
H0 = 1e-12  # control parameter to avoid division by zero in lift force term
SCHEDULER_PATIENCE = 5  # Epochs to wait for loss improvement before reducing LR
SCHEDULER_FACTOR = 0.90  # Factor to multiply LR by when reducing (e.g., 0.5 halves it)
SCHEDULER_MIN_LR = 1e-6  # Minimum LR to stop reducing below this
λ_INIT = 1 # initial value for each self adaptive weight
BUFFER = 1e-6

# Physical parameters (match OpenFOAM)
p = df['p'].values.mean()  # steady state pressure (Pa) 
Ux_max = df['U_0'].values.max()  # max steady state velocity (m/s)
ϕ_average = df['c'].values.mean()  # average ϕ (dimensionless)
H = 25e-6  # channel height (m)
ρ = 1190  # solvent density (Kg/m³)
η = 0.48  # dynamic viscosity (Pa·s)
η0 = η / ρ # kinematic viscosity (m²/s)  
Kn = 0.75  # fitting parameter (dimensionless)
λ2 = 0.8  # fitting parameter (dimensionless)
λ3 = 0.5  # fitting parameter (dimensionless)
α = 4.0  # fitting parameter α ∈ [2, 5] (dimensionless)
a = 2.82e-6  # particle radius (m)
ϕ_max = 0.5  # max ϕ (dimensionless)
ε = a / ((H / 2)**2)  # non-local shear-rate coefficient (1/m)
β = 1.2  # power-law coefficient 
frv = 1.2  # function of the reduced volume

# Data tensors
y_data = torch.tensor(2.0 * (df['y'].values / H) - 1.0, dtype=torch.float32, device=device).unsqueeze(1)
Ux_data = torch.tensor(df['U_0'].values / Ux_max, dtype=torch.float32, device=device).unsqueeze(1)
ϕ_data = torch.tensor(df['c'].values, dtype=torch.float32, device=device).unsqueeze(1) if 'c' in df.columns else None

# Collocation points (uniform, non-random points required for self-adaptive weights)
y_uniform = torch.linspace(-1.0, 1.0, N_PTS, device=device).unsqueeze(1).requires_grad_(True)

# Trial functions
Ux_trial = lambda y: PINN_Ux(torch.cat([y], dim=1))[:,0:1] * (1 + y) * (1 - y)  # torch.Size([y, 1]), normalized 
ϕ_trial = lambda y: ϕ_max * torch.sigmoid(PINN_ϕ(y)[:,0:1]) * (1 + y) * (1 - y)  # torch.Size([y, 1]), sigmoid keeps bounded between 0 and ϕ_max

# Loss history for plotting
class LossHistory:
    def __init__(self):
        self.total = []
        self.individuals = []

    def append(self, ℒ, ℒ_individuals):
        self.total.append(ℒ.item())
        self.individuals.append([ℒ_individual.item() for ℒ_individual in ℒ_individuals])

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

# Separate PINNs for Ux and ϕ
PINN_Ux = NN(neurons=NEURONS, trig_scale=FOURIER_SCALE_Ux).to(device)
PINN_ϕ = NN(neurons=NEURONS, trig_scale=FOURIER_SCALE_ϕ).to(device)

# Loss Functions --------------------------------------------------------------
def ϕ_physics_loss():  # physics ensures ∇⋅J = ∇⋅Σ = 0
    ystar = y_uniform  # already normalized 
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
    scale_L = (H ** 2) / (2 * η0 * Ux_max)  # nondimensionalize after calculating it
    # η0 removed from scale_L
    L = torch.stack([
        torch.cat([zero], dim=1),
        torch.cat([scale_L * (left_wall - right_wall)], dim=1),
        torch.cat([zero], dim=1)
    ], dim=1)
    # NOTE ensure that the units are nondimensional!

    # Diagonal tensor of the SBM (Q)
    Q = torch.tensor([[1.0, 0.0, 0.0], [0.0, λ2, 0.0], [0.0, 0.0, λ3]], device=device).repeat(ystar.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

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
    I = torch.eye(3, device=device).repeat(ystar.shape[0], 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # Fluid phase stress (Σᶠ)
    Σfstar = - pstar * I + 2 * Estar

    # Total stress (Σ)
    Σstar = Σpstar + Σfstar

    # Suspension momentum balance (∇⋅Σ)
    dΣxystar_dystar = torch.autograd.grad(Σstar[:, 0, 1], ystar, torch.ones_like(Σstar[:, 0, 1]), create_graph=True)[0]
    dΣyystar_dystar = torch.autograd.grad(Σstar[:, 1, 1], ystar, torch.ones_like(Σstar[:, 1, 1]), create_graph=True)[0]

    return Jstar_divergence, dΣxystar_dystar, dΣyystar_dystar

def ϕ_IC_loss():  # IC ensures mean(ϕ) never changes
    ystar = y_uniform   # already normalized
    ϕ = ϕ_trial(ystar)

    mass_conservation = torch.mean(ϕ) - ϕ_average 

    return mass_conservation

def ϕ_symmetry_loss():  # ensures ϕ is symmetric along centerflow axis
    ystar = y_uniform   # already normalized

    symmetry = ϕ_trial(ystar) - ϕ_trial(-ystar)

    return symmetry

def ϕ_total_loss():  # combining losses
    Jstar_divergence, dΣxystar_dystar, dΣyystar_dystar = ϕ_physics_loss()  # torch.Size([y, 1]) for each
    mass_conservation = ϕ_IC_loss()  # torch.Size([1])
    symmetry = ϕ_symmetry_loss()  # torch.Size([1])

    # Indivisual losses, in the style of McClenny & Braga-Neto, all become scalars
    ℒ_J = torch.mean(Jstar_divergence**2) / (torch.mean(Jstar_divergence**2).detach() + BUFFER)
    ℒ_Σxy = torch.mean(dΣxystar_dystar**2)
    ℒ_Σyy = torch.mean(dΣyystar_dystar**2)
    ℒ_mass = torch.mean(mass_conservation**2) / (torch.mean(mass_conservation**2).detach() + BUFFER)
    ℒ_symmetry = torch.mean(symmetry**2) / (torch.mean(symmetry**2).detach() + BUFFER)

    # loss function, in the style of McClenny & Braga-Neto
    ℒ = ℒ_J + ℒ_Σxy + ℒ_Σyy + ℒ_mass + ℒ_symmetry

    # Unweighted losses for visualization
    ℒ_J_un = torch.mean(torch.sqrt(Jstar_divergence**2))
    ℒ_Σxy_un = torch.mean(torch.sqrt(dΣxystar_dystar**2))
    ℒ_Σyy_un = torch.mean(torch.sqrt(dΣyystar_dystar**2))
    ℒ_mass_un = torch.mean(torch.sqrt(mass_conservation**2))
    ℒ_symmetry_un = torch.mean(torch.sqrt(symmetry**2))

    # Unweighted loss for visualization of loss decrease
    ℒ_un = ℒ_J_un + ℒ_Σxy_un + ℒ_Σyy_un + ℒ_mass_un

    # individual losses for tracking and eventual visualizatio for debugging
    ℒ_individuals = [ℒ_J_un, ℒ_Σxy_un, ℒ_Σyy_un, ℒ_mass_un, ℒ_symmetry_un]

    return ℒ, ℒ_un, ℒ_individuals

# Visualize -------------------------------------------------------------------
def visualize():
    with torch.no_grad():
        y_plot = torch.linspace(-1.0, 1.0, N_PTS, device=device).unsqueeze(1)
        y_plot_dim = ((y_plot + 1.0) / 2.0 * H).cpu().numpy()
        Ux_pinn = (Ux_trial(y_plot) * Ux_max).cpu().numpy()
        phi_pinn = ϕ_trial(y_plot).cpu().numpy()

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        ax_ux = axs[0][0]
        ax_phi = axs[0][1]
        ax_total_loss = axs[1][1]
        ax_indiv_loss = axs[1][0]

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
        if loss_history.total:
            ax_total_loss.semilogy(loss_history.total, 'g-', label='Total Loss')
            ax_total_loss.set_xlabel('Epoch')
            ax_total_loss.set_ylabel('Loss')
            ax_total_loss.legend()
            ax_total_loss.grid(True)

        # Individual Losses plot
        if loss_history.individuals: # ∇⋅J = ∇⋅Σ
            component_names = ["∇⋅J", "∇⋅Σ (xy)", "∇⋅Σ (yy)", "IC", "sym"]
            for i, indiv_loss in enumerate(zip(*loss_history.individuals)):
                ax_indiv_loss.semilogy(indiv_loss, label=f'Loss {component_names[i]}')
            ax_indiv_loss.set_xlabel('Epoch')
            ax_indiv_loss.set_ylabel('Loss')
            ax_indiv_loss.legend()
            ax_indiv_loss.grid(True)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close(fig)

# Training Loop ---------------------------------------------------------------
# Define parameters
PINN_ϕ_parameters = [{'params': list(PINN_ϕ.parameters()), 'lr': LEARNING_RATE_PINN}]

# Define optimizers
PINN_ϕ_optimizer = torch.optim.Adam(PINN_ϕ_parameters, lr=LEARNING_RATE_PINN)
PINN_ϕ_optimizer_LBFGS = torch.optim.LBFGS(list(PINN_ϕ.parameters()), lr=LEARNING_RATE_PINN)

# Define schedulers (ReduceLROnPlateau type does exactly that)
PINN_ϕ_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(PINN_ϕ_optimizer, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR, verbose=True)

# Call the loss history class
loss_history = LossHistory()

# Load previously saved Ux model
Ux_saved_path = save_path / Ux_model_saved_name
PINN_Ux.load_state_dict(torch.load(f=Ux_saved_path))
PINN_Ux.eval()

# ADAM training loop 
for epoch in range(EPOCHS_ADAM):
    PINN_ϕ_optimizer.zero_grad() 

    # Forward & Backward passes
    ℒ, ℒ_un, ℒ_individuals = ϕ_total_loss()  # forward pass to compute loss
    ℒ.backward()  # backward pass to compute gradients

    # Gradient descent & scheduler update
    PINN_ϕ_optimizer.step()  # gradient descent updating PINN parameters
    PINN_ϕ_scheduler.step(ℒ)

    # Update loss history for plotting
    loss_history.append(ℒ_un, ℒ_individuals)

    # Visuals
    # print(λ_J)
    print(f"Epoch: {epoch} | Loss: {ℒ_un.item()} | Individual Losses: {[f'{l.item():.5f}' for l in ℒ_individuals]} | ϕ Lr: {PINN_ϕ_scheduler.get_last_lr()}")
    if epoch % 50 == 0:
        visualize()

# LBFGS closure function
def closure():
    PINN_ϕ_optimizer_LBFGS.zero_grad()
    ℒ, ℒ_un, ℒ_individuals = ϕ_total_loss()
    ℒ.backward()
    return ℒ

# LBFGS training loop 
for epoch in range(100):
    PINN_ϕ_optimizer_LBFGS.step(closure)
    ℒ, ℒ_un, ℒ_individuals = ϕ_total_loss()
    loss_history.append(ℒ_un, ℒ_individuals)

    print(f"Epoch: {epoch} | Loss: {ℒ_un.item()} | Individual Losses: {[f'{l.item():.5f}' for l in ℒ_individuals]} | ϕ Lr: {PINN_ϕ_scheduler.get_last_lr()}")
    if epoch % 10 == 0:
        visualize()

# Save ϕ model when finished
ϕ_saved_path = save_path / ϕ_model_saved_name
Path(save_path).mkdir(parents=True, exist_ok=True)
torch.save(PINN_ϕ.state_dict(), ϕ_saved_path)
