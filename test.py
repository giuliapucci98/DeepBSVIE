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
folder = "Linear2/"
path = os.path.join(folder, "state_dicts")
graph_path = os.path.join(folder, "Graphs")
if new_folder_flag:
    os.makedirs(graph_path, exist_ok=True)  # <-- This avoids the error if the folder already exists

load_path = os.path.join(path, "parameters.json")
if not os.path.exists(load_path):
    raise FileNotFoundError(f"Parameter file not found at {load_path}")
with open(load_path, "r") as f:
    params = json.load(f)

dim_x = params["dim_x"]
dim_y = params["dim_y"]
dim_d = params["dim_d"]
dim_h_Y = params["dim_h_Y"]
dim_h_Z = params["dim_h_Z"]
N = params["N"]
itr = params["itr"]
batch_size = params["batch_size"]
multiplier = params["multiplier"]

T = params["T"]
x0_value = params["x0_value"]
mu = params["mu"]
sig = params["sig"]
lam = params["lam"]
lam0= params["lam0"]
x_0 = torch.ones(dim_x) * x0_value


equation = volterra_fbsde(x_0, mu, sig, lam, lam0, T, dim_x, dim_y, dim_d)

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

Wt = torch.cumsum(W, dim=-1)
Wt = torch.roll(Wt, 1, -1)
Wt[:, :, 0] = torch.zeros(batch_size, dim_d)

y, z = result.predict(N, batch_size, x, path)


#loss analysis
itr_ax_fine = np.linspace(1, itr * multiplier, itr * multiplier)
itr_ax_coarse = np.linspace(1, itr, itr)
# Set up the figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
#fig.suptitle("Loss at Specific Time Steps", fontsize=16)

# Plot 1: Loss at time step N-1
axs[0].plot(itr_ax_fine[:], loss[0][:])
axs[0].set_title("Loss at time step N-1")

# Plot 2: Loss at time step N-2
axs[1].plot(itr_ax_fine[:], loss[1][:])
axs[1].set_title("Loss at time step N-2")

# Plot 4: Loss at time step N-5
axs[2].plot(itr_ax_coarse, loss[-1][:])
axs[2].set_title("Loss at time step 0")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(graph_path, "loss.png"))
plt.show()



######### PLOT single loss
plt.figure(figsize=(8, 6))
plt.plot(itr_ax_fine[:], loss[12][:])
plt.xlabel("Iterations")
plt.ylabel("Loss")
#plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "loss0.png"))
plt.show()




x_np = x.detach().cpu().numpy()  # Convert to numpy array
y_np = y.detach().cpu().numpy()
z_np = z.detach().cpu().numpy()

def analytical_Y(times, x):
    #LINEAR EXAMPLE 1
    #factor1 = np.sin(np.pi * times) + (-np.cos(np.pi * T) + np.cos(np.pi * times)) / np.pi  # shape: (N,)
    #factor1 = factor1 # shape: (1, 1, N)
    ##factor2 = x + (T - times)  # shape: (batch_size, 1, N) #Linear example 1
    #factor2= x + np.exp(T) - np.exp(times)  # shape: (batch_size, 1, N) #Linear example 2
    #return factor1 * factor2  # shape: (batch_size, 1, N)

    #LINEAR EXAMPLE 2
    exp_term = np.exp(-lam * times) * np.exp(mu * (T - times))
    integral_term = 0.5 * (np.exp(mu * (T - times)) - 1) / mu
    return x * (exp_term + integral_term)

times = np.linspace(0, T, N+1)
Y = analytical_Y(times, x_np)  # (batch_size, 1, N)



######### PLOT Y
n_samples = 3
sample_indices = np.random.choice(batch_size, size=n_samples, replace=False)

plt.figure(figsize=(10, 6))
for i, j in enumerate(sample_indices):
    # Plot analytical solution
    plt.plot(times, Y[j, 0, :], linestyle='--', color='blue',label="Analytical Solution" if i == 0 else None)
    # Plot prediction
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

######### SURFACE PLOT Z
sample_index = torch.randint(0, batch_size, (1,)).item()
z_sample = z_np[sample_index] # shape: (49, 49)
t_vals = np.arange(N)[:-1]
s_vals = np.arange(N)[:-1]
t_grid, s_grid = np.meshgrid(t_vals, s_vals, indexing='ij')
# Mask Z(t,s) to only show s > t
z_masked = np.where(s_grid >= t_grid, z_sample, np.nan)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(t_grid, s_grid, z_masked, cmap='viridis', edgecolor='k', linewidth=0.3)
ax.view_init(elev=30, azim=0)  # Adjust these values as needed
#ax.set_title(f"Surface Plot of Z(t, s) — Sample #{sample_index}")
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("Z(t, s)")
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"z_surface_plot_sample_{sample_index}.png"))
plt.show()

def analytical_Z(times, z):
    t_grid = np.linspace(0, T, N)  # length N-1
    #z_analytical = np.zeros_like(z[0, 0, 0, :, :]) #this for first example
    z_analytical = np.zeros_like(z[:, :, :]) #this for second example

    for t_idx in range(N-1):
        for s_idx in range(t_idx, N-1):
            #example linear 2
            exp_term = np.exp(-lam * times[s_idx]) * np.exp(mu * (T - times[s_idx]))
            integral_term = 0.5 * (np.exp(mu * (T - times[s_idx])) - 1) / mu
            z_analytical[:, t_idx, s_idx] = sig * x_np[:, 0, s_idx] * (exp_term + integral_term)

        return z_analytical

'''
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import trapezoid

#LINEAR EXAMPLE 1
def analytical_Z(times, z):
    dt = T / N
    t_grid = np.linspace(0, T, N)[:-1]  # time grid with N-1 points
    z_analytical = np.zeros_like(z[0, :, :])  # shape (N-1, N-1)

    for t_idx, t in enumerate(t_grid):
        for s_idx, s in enumerate(t_grid):
            if s < t:
                continue  # skip: integration starts only at s >= t

            # r_grid = [s, ..., T]
            r_grid = np.linspace(s, T, max(2, N - s_idx))  # at least 2 points
            dr = r_grid[1] - r_grid[0]

            I1 = np.zeros_like(r_grid)

            for i, r in enumerate(r_grid):
                # u_grid = [r, ..., T]
                u_start_idx = int(np.floor(r / dt))
                if u_start_idx >= N - 1:
                    I1[i] = 0
                    continue

                u_grid = np.linspace(r, T, max(2, N - u_start_idx))
                sin_u = np.sin(np.pi * u_grid)
                I1[i] = trapezoid(sin_u, u_grid)

            sin_r = np.sin(np.pi * r_grid)
            kernel = np.exp(-(r_grid - t))
            integrand = kernel * (sin_r + I1)
            outer_integral = trapezoid(integrand, r_grid)

            z_analytical[t_idx, s_idx] = np.sin(np.pi * t) + outer_integral

    return z_analytical
'''


#surface plot of the analytical Z
######### SURFACE PLOT Z
Z = analytical_Z(times[:-1],z_np)  # (batch_size, N-1, N-1)
# Mask Z(t,s) to only show s > t
#Z_masked = np.where(s_grid >= t_grid, Z, np.nan) #linear example 1
Z_sample = Z[sample_index] # shape: (49, 49) #linear example 2
Z_masked = np.where(s_grid >= t_grid, Z_sample, np.nan)



#SURFACES OVERLAPPED
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# Plot reference surface Z
surf_z = ax.plot_surface(t_grid, s_grid, z_masked, cmap='plasma', alpha=0.7, edgecolor='none', label='z: approximated solution')
surf_Z = ax.plot_surface(t_grid, s_grid, Z_masked, cmap='viridis', alpha=0.7, edgecolor='none', label='Z: reference solution')
ax.view_init(elev=30, azim=0)  # azim=45 gives more depth
# Axis labels
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("Z(t,s)")
# Custom legend using dummy lines
custom_lines = [
    Line2D([0], [0], color='mediumseagreen', lw=4, label='Z(t,s) reference'),
    Line2D([0], [0], color='orange', lw=4, label='Z(t,s) approximated')
]
ax.legend(handles=custom_lines, loc='upper left',fontsize=13)
# Layout and save
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"z_surface_plot_sample_{sample_index}.png"))
plt.show()








Z = analytical_Z(times[:-1],z_np)  # (batch_size, N , N)



sample_index = torch.randint(0, batch_size, (1,)).item()
s_grid = np.linspace(0, T, N)[:-1]  # Shape: (N-1,)
t_indices = [5, 10, 20]
colors = ['b', 'g', 'r', 'c', 'm']
labels = [f"t = {np.round(s_grid[t_idx], 2)}" for t_idx in t_indices]

plt.figure(figsize=(10, 6))
for i, t_idx in enumerate(t_indices):
    z_s = z_np[sample_index, t_idx, t_idx:-1]  # predicted Z(t,s)
    Z_s = Z[sample_index, t_idx, t_idx:-1]          # analytical Z(t,s)
    s_subgrid = s_grid[t_idx:-1]

    plt.plot(s_subgrid, z_s, color=colors[i], linestyle='-', linewidth=2,
             label=f'Predicted Z({np.round(s_grid[t_idx], 2)}, s)')
    plt.plot(s_subgrid, Z_s, color=colors[i], linestyle='--', linewidth=2,
             label=f'Analytical Z({np.round(s_grid[t_idx], 2)}, s)')
plt.xlabel('s')
plt.ylabel('Z(t,s)')
#plt.title('Comparison of Predicted and Analytical Z(t,s) for Various t')
plt.legend(fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"Z_comparison_sample{sample_index}.png"))
plt.show()


sample_index= torch.randint(0, batch_size, (1,)).item()
z_s = z_np[sample_index, 15, :]
Z_s = Z[sample_index, 15, :]
#Z_s = Z[15, :]
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0,T,N)[:-1], z_s, 'b-', label='Predicted Z(t,t)', linewidth=2)
plt.plot(np.linspace(0,T,N)[:-1], Z_s, 'r--', label='Analytical Z(t,t)', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Z value')
plt.title('Z(t,s)')
#plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"z(t,t){sample_index}.png"))
plt.show()

t_indices = [5, 20, 30]
colors = ['b', 'g', 'r', 'c', 'm']
sample_index = torch.randint(0, batch_size, (1,)).item()
plt.figure(figsize=(10, 6))
for i, index in enumerate(t_indices):
    color = colors[i % len(colors)]  # ensure we cycle through colors if needed
    z_s = z_np[sample_index, index, :]
    Z_s = Z[sample_index, index, :]
    plt.plot(np.linspace(0, T, N)[:-1], z_s, '-', color=color, label=f'Predicted Z(t,s), t={index}', linewidth=2)
    plt.plot(np.linspace(0, T, N)[:-1], Z_s, '--', color=color, label=f'Analytical Z(t,s), t={index}', linewidth=2)
plt.xlabel('s')
plt.ylabel('Z value')
#plt.title('Z(t,s)')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"z(t,t){sample_index}.png"))
plt.show()

s_indices = [5, 20, 30]
colors = ['b', 'g', 'r', 'c', 'm']
labels = [f"s = {np.round(s_grid[s_idx], 2)}" for s_idx in s_indices]
sample_index = torch.randint(0, batch_size, (1,)).item()
plt.figure(figsize=(10, 6))
for i, s_idx in enumerate(s_indices):
    # Extract slices for fixed s
    z_t = z_np[sample_index, :, s_idx]      # predicted Z(t, s_i)
    #Z_t = Z[:, s_idx]                       #linear example 1
    Z_t = Z[sample_index, :, s_idx] #linear example 2
    t_subgrid = s_grid[:len(z_t)]           # time grid (same as t_grid[:, s_idx])
    # Plot predicted and reference
    plt.plot(t_subgrid, z_t, color=colors[i], linestyle='-', linewidth=2,
             label=f'Predicted Z(t, {np.round(s_grid[s_idx], 2)})')
    plt.plot(t_subgrid, Z_t, color=colors[i], linestyle='--', linewidth=2,
             label=f'Analytical Z(t, {np.round(s_grid[s_idx], 2)})')
plt.xlabel('t')
plt.ylabel('Z(t,s)')
plt.ylim(top=0.4)
plt.legend(fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, f"Z_t_fixed_s_sample{sample_index}.png"))
plt.show()


def Err_Y(Y_true, Y_pred, T):
    """
    Compute Err(Y) as defined above.
    """
    M, _, N = Y_true.shape
    h = T / N
    squared_diff = np.linalg.norm(Y_true - Y_pred, axis=1) ** 2  # shape (M,1, N)
    return np.mean(np.sum(squared_diff, axis=-1) * h, axis=0)


def Err_Z(Z_true, Z_pred, T):
    M, N, N = Z_pred.shape
    h = T / N
    err_sum = 0.0
    for k in range(N):
        for n in range(k, N):  # only upper triangle
            #err_sum += (Z_true[None, k, n] - Z_pred[:, k, n]) ** 2 #first example
            err_sum += (Z_true[:, k, n] - Z_pred[:,  k, n]) ** 2 #second example
    return (h ** 2 ) * np.mean(err_sum, axis = 0)

err_y = Err_Y(Y, y_np, T)
print("Err(Y):", err_y)

err_z = Err_Z(Z, z_np, T)
print("Err(Z):", err_z)

def Rel_Err_Y(Y_true, Y_pred, T):
    M, _, N = Y_true.shape
    h = T / N
    norm1 = np.linalg.norm(Y_true - Y_pred, axis=1)**2  # shape (M,1, N)
    norm2 = np.linalg.norm(Y_true, axis=1)**2
    return np.mean(np.sum(norm1, axis=-1) / np.sum(norm2, axis=-1), axis=0)



def Rel_Err_Z(Z_true, Z_pred, T):
    M, N, N = Z_pred.shape
    h = T / N
    err_sum = 0.0
    err_sum2 = 0.0
    for k in range(N):
        for n in range(k, N):  # only upper triangle
            #err_sum += (Z_true[None, k, n] - Z_pred[:, k, n]) ** 2  # first example
            #err_sum2 += Z_true[None, k, n] ** 2  # first example
            err_sum += (Z_true[:, k, n] - Z_pred[:, k, n]) ** 2 #second example
            err_sum2 += Z_true[:, k, n] ** 2  # first example

    return (h ** 2) * np.mean(err_sum/err_sum2, axis=0)


rel_err_y = Rel_Err_Y(Y, y_np, T)
print("Relative Error (Y):", rel_err_y)

rel_err_z = Rel_Err_Z(Z, z_np, T)
print("Relative Error (Z):", rel_err_z)


import numpy as np
import matplotlib.pyplot as plt

Y_true_all_batches = Y[:, 0, :]       # shape (bs, N)
Y_pred_all_batches = y_np[:, 0, :]    # shape (bs, N)
squared_errors_Y = (Y_pred_all_batches - Y_true_all_batches)**2  # shape (bs, N)
mean_squared_error_Y = np.mean(squared_errors_Y, axis=0)         # shape (N,)
H = Y.shape[-1]
t_grid2 = np.linspace(0, 1, H)

         # shape (bs, N, N)
#squared_errors_Z = (z_np - Z[None, :, :])**2     #example 1
squared_errors_Z = (z_np - Z[:, :, :])**2 #example 2
mean_squared_error_Z = np.mean(squared_errors_Z, axis=0)        # shape (N, N)
N = Z.shape[1]
t = np.linspace(0, 1, N)
s = np.linspace(0, 1, N)
T_grid2, S_grid = np.meshgrid(t, s, indexing='ij')
masked_error_Z = np.ma.masked_where(S_grid < T_grid2, mean_squared_error_Z)



# ---- Plot 1: Y error over time ----
plt.figure(figsize=(8, 5))
plt.plot(t_grid2, mean_squared_error_Y, label="Mean squared error for Y")
plt.grid(True)
plt.xlabel('Time t')
#plt.ylabel('Squared error')
#plt.legend()
#plt.title('Mean Squared Error for Y(t)')
plt.show()


# ---- Plot 2: Z error surface ----
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(t_grid, s_grid, masked_error_Z, cmap='viridis', label= "Mean squared error for Z")
#ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('s')
ax.set_zlabel('Squared error')
#ax.set_title('Mean Squared Error Surface for Z(t,s)')
plt.show()



plt.figure(figsize=(8, 6))
plt.imshow(masked_error_Z, extent=[s_grid.min(), s_grid.max(), t_grid.min(), t_grid.max()],
           origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Mean squared error for Z')
plt.xlabel('s')
plt.ylabel('t')
#plt.title('Heatmap of Mean Squared Error for Z(t,s)')
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "Z_error_heatmap.png"))
plt.show()

