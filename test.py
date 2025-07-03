import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.lines import Line2D

from BSVIE import volterra_fbsde
from BSVIE import NN_Y
from BSVIE import NN_Z
from BSVIE import Result

device = torch.device("cpu")

print("Working directory:", os.getcwd())

new_folder_flag = True
folder = "barrier/"
path = os.path.join(folder, "state_dicts")
graph_path = os.path.join(folder, "Graphs")
if new_folder_flag:
    os.makedirs(graph_path, exist_ok=True)  # <-- This avoids the error if the folder already exists

#load parameters
load_path = os.path.join(path, "parameters.json")
if not os.path.exists(load_path):
    raise FileNotFoundError(f"Parameter file not found at {load_path}")
with open(load_path, "r") as f:
    params = json.load(f)

#unpack parameters
dim_x = params["dim_x"]
dim_y = params["dim_y"]
dim_d = params["dim_d"]
dim_h_Y = params["dim_h_Y"]
dim_h_Z = params["dim_h_Z"]
N = params["N"]
itr = params["itr"]
batch_size = params["batch_size"]
multiplier = params["multiplier"]

#unpack BSVIE parameters
T = params["T"]
x0_value = params["x0_value"]
mu = params["mu"]
sig = params["sig"]
lam = params["lam"]
lam0= params["lam0"]
K = params["K"]
x_0 = torch.ones(dim_x) * x0_value


equation = volterra_fbsde(x_0, mu, sig, lam, lam0,K, T, dim_x, dim_y, dim_d)

loss_path = os.path.join(path, "loss.json")
with open(loss_path, 'r') as f:
    loss = json.load(f)

modelY = NN_Y(equation, dim_h_Y)
modelZ = NN_Z(equation, dim_h_Z)
modelY.eval()
modelZ.eval()

result = Result(modelY, modelZ, equation)

flag = True
while flag:
    W = result.gen_b_motion(batch_size, N)
    x = result.gen_x(batch_size, N, W)
    flag = torch.isnan(x).any()

###########################
Wt = torch.cumsum(W, dim=-1)
Wt = torch.roll(Wt, 1, -1)
Wt[:, :, 0] = torch.zeros(batch_size, dim_d)

y, z = result.predict(N, batch_size, x, path)


#loss analysis
itr_ax_fine = np.linspace(1, itr * multiplier, itr * multiplier)
itr_ax_coarse = np.linspace(1, itr, itr)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))

# Plot 1: Loss at time step N-1
axs[0].plot(itr_ax_fine[:], loss[0][:])
axs[0].set_title("Loss at time step N-1")

# Plot 2: Loss at time step N-2
axs[1].plot(itr_ax_fine[:], loss[1][:])
axs[1].set_title("Loss at time step N-2")

# Plot 4: Loss at time step N-5
axs[2].plot(itr_ax_coarse, loss[-1][:])
axs[2].set_title("Loss at time step 0")

# Improve layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(graph_path, "loss.png"))
plt.show()



######### PLOT single loss
plt.figure(figsize=(8, 6))
plt.plot(itr_ax_fine[:], loss[-1][:])
plt.xlabel("Iterations")
plt.ylabel("Loss")
#plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "loss0.png"))
plt.show()




x_np = x.detach().cpu().numpy()
y_np = y.detach().cpu().numpy()
z_np = z.detach().cpu().numpy()

times = np.linspace(0, T, N+1)
times_torch = torch.linspace(0, T, N+1)

L_np = equation.barrier(times, x).detach().numpy()
g_np = equation.g(times_torch, x).detach().numpy()


plt.figure(figsize=(10, 6))
n_samples = 3
sample_indices = np.random.choice(batch_size, size=n_samples, replace=False)
for i, j in enumerate(sample_indices):
    # Plot analytical solution (dashed blue)
    #plt.plot(times, Y[j, 0, :], linestyle='--', color='blue',label="Analytical Solution" if i == 0 else None)

    # Plot neural prediction (solid red)
    plt.plot(times,x_np[j, 0, :],linestyle='-',color='red',alpha=0.8,label="X" if i == 0 else None)
    plt.plot(times,L_np[j, 0, :],linestyle='--',color='blue',alpha=0.8,label="Barrier" if i == 0 else None)

plt.xlabel("Time t")
plt.ylabel("X_t")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "forward.png"))
plt.show()





n_samples = 3
sample_indices = np.random.choice(batch_size, size=n_samples, replace=False)

plt.figure(figsize=(10, 6))
for i, j in enumerate(sample_indices):
    # Plot neural prediction (solid red)
    plt.plot(times,L_np[j, 0, :],linestyle='--',color='blue',alpha=0.8,label="Barrier" if i == 0 else None)

    plt.plot(times, y_np[j, 0, :],linestyle='-',color='red',alpha=0.8,label="Approximated Solution" if i == 0 else None)
plt.xlabel("Time t")
plt.ylabel("Y(t)")
#plt.title("Analytical vs. Approximated BSVIE")
plt.legend(fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "explicit_solution_multiple_samples.png"))
plt.show()



z_np = z_np[:, 0,0,:-1,:-1]

sample_index = torch.randint(0, batch_size, (1,)).item()
#sample_index=2052
z_sample = z_np[sample_index] # shape: (49, 49)
t_vals = np.arange(N)[:-1]
s_vals = np.arange(N)[:-1]
t_grid, s_grid = np.meshgrid(t_vals, s_vals, indexing='ij')
# Mask Z(t,s) to only show s > t
z_masked = np.where(s_grid >= t_grid, z_sample, np.nan)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(t_grid, s_grid, z_masked, cmap='viridis', edgecolor='k', linewidth=0.3)
ax.view_init(elev=30, azim=30)  # Adjust these values as needed
#ax.set_title(f"Surface Plot of Z(t, s) — Sample #{sample_index}")
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("Z(t, s)")
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"z_surface_plot_sample_{sample_index}.png"))
plt.show()

