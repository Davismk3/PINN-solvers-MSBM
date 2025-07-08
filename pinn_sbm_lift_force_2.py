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

saved_models = Path("/Users/michaeldavis/Desktop/Python/SBM/Final/saved_models")  # saved models directory path
data_files = Path("/Users/michaeldavis/Desktop/Python/SBM/data")  # data directory path
Ux_saved = "Ux_trial_MSBM_bulk_0_05"  # Ux model name to use and/or create
ϕ_saved = "phi_trial_MSBM_0_05"  # Ux model name to create
data_file = "Initial_bulk_vol_frac_0_05.csv"  # data file, must be csv
Ux_saved_path = saved_models / Ux_saved
ϕ_saved_path = saved_models / ϕ_saved
data_path = data_files / data_file
df = pd.read_csv(data_path)

# visuals 
animate = True  # plot results every 10 epochs?

# PINN type
training_ϕ = True  # Ux and ϕ are trained separately 

# PINN parameters 
NEURONS = 64  # hidden‑layer width
EPOCHS_ADAM = 10000  # iterations for faster optimizer
EPOCHS_LBFGS = 100  # iterations for better optimizer
LEARNING_RATE_ADAM = 1e-3
LEARNING_RATE_LBFGS = 1e-4
N_PTS = 1000 # collocation points
BUFFER = 1e-3 # torch.finfo(torch.float32).eps  # for loss function
AVG_WEIGHT = 10  # weight coefficient for enforcing mean ϕ
FOURIER_SCALE_Ux = 0.2  # larger values seem to yeild worse results and fail to allow ϕ convergence
FOURIER_SCALE_ϕ = 4  # this seems to not be as sensitive as FOURIER_SCALE_Ux, larger values give more detail
H0 = 1e-12  # control parameter to avoid division by zero

# parameters
p = df['p'].values.mean()  # steady state pressure (Pa) 
Ux_max = df['U_0'].values.max()  # max steady state velocity (m/s)
ϕ_average = df['c'].values.mean() # average ϕ (dimensionless)
H = 5e-5  # channel height (m)
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

# data tensors | y and Ux are made dimensionless
y_data = torch.tensor(2.0 * (df['y'].values / H) - 1.0, dtype=torch.float32, device=device, requires_grad=False).unsqueeze(1)  # torch.Size([1001, 1]) | [-1, 1]
Ux_data = torch.tensor(df['U_0'].values / df['U_0'].values.max(), dtype=torch.float32, device=device).unsqueeze(1)  # torch.Size([1001, 1]) | [0, 1]
ϕ_data = torch.tensor(df['c'].values, dtype=torch.float32, device=device).unsqueeze(1)  # torch.Size([1001, 1])

# trial functions 
def y_random(n_pts):  # normalized [-1, 1]
    interior = torch.rand(n_pts-3, 1, device=device, requires_grad=True) * 2 - 1
    boundaries = torch.tensor([[-1.0], [0.0], [1.0]], device=device).requires_grad_()
    return torch.cat([interior, boundaries], dim=0)
y_uniform = torch.linspace(-1.,1.,N_PTS,device=device).unsqueeze(1).requires_grad_(True)  # average ϕ will be enforced over unchanging y values
Ux_trial = lambda y: PINN_Ux(torch.cat([y], dim=1))[:,0:1] * (1 + y) * (1 - y)  # torch.Size([y, 1]) | normalized 
ϕ_trial = lambda y: ϕ_max * torch.sigmoid(PINN_ϕ(y)[:,0:1]) * (1 + y) * (1 - y)  # torch.Size([y, 1])

# PINN ------------------------------------------------------------------------
class FourierFeatures(nn.Module): # reference: https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb
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

PINN_Ux = NN(neurons=NEURONS, trig_scale=FOURIER_SCALE_Ux)
PINN_ϕ = NN(neurons=NEURONS, trig_scale=FOURIER_SCALE_ϕ)

# load saved Ux model
if training_ϕ:
    for parameter in PINN_Ux.parameters():
        parameter.requires_grad_(False)
    PINN_Ux.load_state_dict(torch.load(f=Ux_saved_path))
    PINN_Ux.eval()

# Loss Weights ----------------------------------------------------------------
class LossScalars(nn.Module):
    # weight_symmetry, weight_Jwall, weight_mass, weight_J, weight_Σxy, weight_Σyy
    def __init__(self, initial_values=(10.0, 1.0, 30.0, 1.0, 1.0, 1.0)):
        super().__init__()
        self.weights = nn.Parameter(torch.log(torch.tensor(initial_values, device=device)))

    def forward(self):
        return torch.nn.functional.softplus(self.weights)

weights = LossScalars()

# Loss ------------------------------------------------------------------------
def Ux_data_loss(y): 
    return torch.mean((Ux_trial(y) - Ux_data)**2)

def ϕ_loss(y):
    ystar = y  # already normalized 
    Uxstar = Ux_trial(y)  # already normalized
    ϕ = ϕ_trial(y)
    A = 2 * a / H
    pstar = p * H / (2 * η0 * Ux_max)

    ϕ = ϕ_trial(y)
    zero = torch.zeros_like(y)  # torch.Size([y, 1])

    # normal stress viscosity (ηₙ(ϕ))
    def ηN(ϕ):
        return Kn * (ϕ/ϕ_max)**2 * (1 - ϕ/ϕ_max)**(-2)  # torch.Size([y, 1]), a scalar for each y

    # shear viscosity of the particle phase (ηₚ(ϕ))
    def ηp(ϕ):
        ηs = (1 - ϕ/ϕ_max)**(-2)
        return ηs - 1  # torch.Size([y, 1]), a scalar for each y

    # sedimentation hinderence function for mobility of particle phase (f(ϕ))
    def f(ϕ):
        return (1 - ϕ/ϕ_max) * (1 - ϕ)**(α - 1)  # torch.Size([y, 1]), a scalar for each y

    # gradient of the velocity field (∇U)
    dUxstar_dystar = torch.autograd.grad(Uxstar, ystar, torch.ones_like(Uxstar), create_graph=True)[0]  # torch.Size([y, 1])
    Ustar_gradient = torch.stack([
        torch.cat([zero, dUxstar_dystar, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1),
        torch.cat([zero, zero, zero], dim=1)
        ], dim=1)  # torch.Size([y, 3, 3]), a matrix for each y
    
    # strain rate tensor (E)
    Estar = 0.5 * (Ustar_gradient + Ustar_gradient.transpose(1, 2))  # torch.Size([y, 3, 3]), a matrix for each y

    # shear rate tensor (γ̇)
    γ̇star = torch.sqrt(2 * torch.sum(Estar * Estar, dim=(1, 2))).unsqueeze(1)  # torch.Size([y, 1])
    # print("γ̇: ", γ̇[0, :])

    # lift force (L)
    γ̇ = γ̇star * 2 * Ux_max / H  # dimensionalize
    left_wall = 3 * η0 * γ̇ / (4 * torch.pi * ((H/2)*(ystar + 1) + H0)**β) * frv
    right_wall = 3 * η0 * γ̇ / (4 * torch.pi * ((H/2)*(1 - ystar) + H0)**β) * frv
    scale_L = H / (2 * η0 * Ux_max)  # nondimensionalize 
    L = torch.stack([
        torch.cat([zero], dim=1), 
        torch.cat([scale_L * (left_wall - right_wall)], dim=1),
        torch.cat([zero], dim=1)
        ], dim=1)
    # print ("L: ", L)  # torch.Size([y, 3, 1]), a vector for each y

    # diagonal tensor of the SBM (Q)
    Q = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, λ2, 0.0],
        [0.0, 0.0, λ3]
        ], dtype=torch.float32, device=device).repeat(N_PTS, 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y
  
    # non-local shear rate tensor
    γ̇NLstar = ε * H / 2
    # print("γ̇NL: ", γ̇NL)

    # particle normal stress diagonal tensor (Σₙₙᵖ)
    Σpnnstar = ηN(ϕ).view(-1, 1, 1) * (γ̇star.unsqueeze(1) + γ̇NLstar) * Q  # torch.Size([y, 3, 3]), a matrix for each y

    # oriented particle stress tensor (Σᵖ)
    Σpstar = - Σpnnstar + (2 * ηp(ϕ).view(-1, 1, 1) * Estar)  # torch.Size([y, 3, 3]), a matrix for each y

    # divergence of oriented particle stress tensor (∇⋅Σᵖ)
    dΣpxystar_dystar = torch.autograd.grad(Σpstar[:, 0, 1], ystar, torch.ones_like(Σpstar[:, 0, 1]), create_graph=True)[0]  # torch.Size([1001, 1])
    dΣpyystar_dystar = torch.autograd.grad(Σpstar[:, 1, 1], ystar, torch.ones_like(Σpstar[:, 1, 1]), create_graph=True)[0]  # torch.Size([1001, 1])
    Σpstar_divergence = torch.stack([
        torch.cat([zero + dΣpxystar_dystar + zero], dim=1), 
        torch.cat([zero + dΣpyystar_dystar + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
        ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y
    
    # migration flux (J)
    Jstar = - (2 * A**2 / 9) * f(ϕ).unsqueeze(1) * (Σpstar_divergence + ϕ.view(-1, 1, 1) * L)  # torch.Size([1001, 3, 1])
    Jstar_wall = Jstar[-3, 1, 0]**2 + Jstar[-1, 1, 0]**2  # these are the correct indices due to how y_random is setup
    # print("J: ", J[0, :, :])

    # divergence of migration flux (∇⋅J)
    dJxstar_dxstar = dJzstar_dzstar = zero
    dJystar_dystar = torch.autograd.grad(Jstar[:, 1, 0], ystar, torch.ones_like(Jstar[:, 1, 0]), create_graph=True)[0]  # torch.Size([1001, 1])
    Jstar_divergence = dJxstar_dxstar + dJystar_dystar + dJzstar_dzstar  # torch.Size([1001, 1])
    # print("J: ", J[0, :, :] * df['U_0'].max() * 2.0 / H)
    
    # identity matrix (I)
    I = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device).repeat(N_PTS, 1, 1)  # torch.Size([y, 3, 3]), a matrix for each y

    # fluid phase stress (Σᶠ)
    Σfstar = - pstar * I + 2 * Estar

    # total stress (Σ)
    Σstar = Σpstar + Σfstar
    # print("Σ: ", Σ[0, :, :])

    # suspension momentum balance (∇⋅Σ)
    dΣxystar_dystar = torch.autograd.grad(Σstar[:, 0, 1], ystar, torch.ones_like(Σstar[:, 0, 1]), create_graph=True)[0]  # torch.Size([y, 1])
    dΣyystar_dystar = torch.autograd.grad(Σstar[:, 1, 1], ystar, torch.ones_like(Σstar[:, 1, 1]), create_graph=True)[0]  # torch.Size([y, 1])
    Σstar_divergence = torch.stack([
        torch.cat([zero + dΣxystar_dystar + zero], dim=1), 
        torch.cat([zero + dΣyystar_dystar + zero], dim=1),
        torch.cat([zero + zero + zero], dim=1)
        ], dim=1)  # torch.Size([y, 3, 1]), a vector for each y
    
    # enforce the known average number of particles in the channel
    mass_conservation = torch.mean(ϕ_trial(y_uniform)) - ϕ_average

    # enforce symmetry
    phi_pos = ϕ_trial(y)
    phi_neg = ϕ_trial(-y)
    symmetry = phi_pos - phi_neg
    
    # return ∇⋅J = 0 and ∇⋅Σ = 0
    # print("J loss: ", torch.mean(Jstar_divergence**2))
    # print("Σ loss: ", torch.mean(Σstar_divergence**2))
    weight_symmetry, weight_Jwall, weight_mass, weight_J, weight_Σxy, weight_Σyy = weights()
    return (weight_symmetry * torch.mean(symmetry**2) / (torch.mean(symmetry**2).detach() + BUFFER) + 
            weight_Jwall * torch.mean(Jstar_wall) / (torch.mean(Jstar_wall).detach() + BUFFER) + 
            weight_mass * torch.mean(mass_conservation**2) / (torch.mean(mass_conservation**2).detach() + BUFFER) +
            weight_J * torch.mean(Jstar_divergence**2) / (torch.mean(Jstar_divergence**2).detach() + BUFFER) + 
            weight_Σxy * torch.mean(dΣxystar_dystar**2) / (torch.mean(dΣxystar_dystar**2).detach() + BUFFER) + 
            weight_Σyy * torch.mean(dΣyystar_dystar**2)/ (torch.mean(dΣyystar_dystar**2).detach() + BUFFER))

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

def Adam_Particle(learning_rate, epochs):
    PINN_ϕ.train()
    parameters = list(PINN_ϕ.parameters()) + list(weights.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
    for epoch in range(epochs): 
        optimizer.zero_grad()
        y = y_random(N_PTS)
        loss = ϕ_loss(y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print("Adam epoch: ", epoch, " | Loss: ", loss.item())
        loss_history.append(loss.item())

        if epoch % 10 == 0:
            visualize(ϕ_data, ϕ_trial, 'ϕ (Particle Concentration)')
            print("learned weights:", weights().detach().cpu().numpy())

def LBFGS_Particle(learning_rate, epochs):
    parameters = list(PINN_ϕ.parameters()) + list(weights.parameters())
    optimizer = torch.optim.LBFGS(parameters, lr=learning_rate)

    def closure():
        optimizer.zero_grad()
        loss = ϕ_loss(y_random(N_PTS))
        loss.backward()
        loss_history.append(loss.item())
        return loss

    for epoch in range(epochs):
        optimizer.step(closure)
        print("LBFGS epoch: ", epoch, " | Loss: ", ϕ_loss(y_random(N_PTS)).item())
    visualize(ϕ_data, ϕ_trial, 'ϕ (Particle Concentration)')

# Training Loop ---------------------------------------------------------------
if not training_ϕ:
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

    saved_models.mkdir(parents=True, exist_ok=True)
    torch.save(obj=PINN_Ux.state_dict(), f=Ux_saved_path)

elif training_ϕ:
    visualize(Ux_data * df['U_0'].values.max(), lambda y: Ux_trial(y) * torch.tensor(df['U_0'].max(), dtype=torch.float32, device=device), 'Ux (Velocity)')
    Adam_Particle(epochs=200, learning_rate=1e-3)
    Adam_Particle(epochs=300, learning_rate=1e-4)
    Adam_Particle(epochs=100, learning_rate=1e-5)
    Adam_Particle(epochs=100, learning_rate=1e-6)

    saved_models.mkdir(parents=True, exist_ok=True)
    torch.save(obj=PINN_ϕ.state_dict(), f=ϕ_saved_path)