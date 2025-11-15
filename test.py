import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from numpy.ma.core import ones_like

from BSVIE import volterra_fbsde, Solver
from Evaluation import Result
from plot_generator import PlotGenerator


# =============== SELECT THESE BEFORE STARTING ===========
example_type = "nonlinear"  # select example. Options: "linear1", "linear2", "example1a"

run_name = "nonlinear_2025-11-15_13-21-08"

# =============== PATHS ===========
print("Working directory:", os.getcwd())
#new_folder = os.path.join(example_type, run_name)
new_folder= os.path.join(run_name)

project_dir = "/cfs/klemming/projects/supr/naiss2024-22-1707/DeepBSVIE"
if os.path.exists(project_dir): #we are on server
    path = os.path.join(project_dir, new_folder, "models")
    figures_path = os.path.join(project_dir, new_folder, "figures")
    # Set WandB cache and artifacts directories
    os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, "wandb_cache")
    os.environ['WANDB_ARTIFACTS_DIR'] = os.path.join(project_dir, "wandb_artifacts")
    # Make sure directories exist
    os.makedirs(os.environ['WANDB_CACHE_DIR'], exist_ok=True)
    os.makedirs(os.environ['WANDB_ARTIFACTS_DIR'], exist_ok=True)
else:
    project_dir= os.getcwd()
    figures_path = os.path.join(new_folder, "figures")
    path = os.path.join(new_folder, "models")
    #path = os.path.join(project_dir, new_folder, "models") #old code

#=============== DEVICE ===========
device =  "cpu"

#=============== CONFIG ===========
config_path = os.path.join(project_dir, new_folder, "config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"❌ Config file not found: {config_path}")
with open(config_path, "r") as f:
    config = json.load(f)

dim_x = config.get('dim_x')
dim_y = 1
dim_d = dim_x
N = config.get('N')
T = config.get('T')
x_0 = torch.ones(dim_x).to(device) * config.get('x0_scale')

equation = volterra_fbsde(
    x_0=x_0,
    mu_base=config.get('mu_base'),
    sig_base=config.get('sig_base'),
    lam=0.5, lam0=0.5, T=T,
    dim_x=dim_x, dim_y=dim_y, dim_d=dim_d,
    example_type=example_type,
    seed=config.get('seed', 42)
)

# =============== LOAD MODELS ===========
future_models_Y = {}
future_models_Z = {}
for n in range(N + 1):
    solver = Solver(
        equation,
        config['dim_h_Y'],
        config['dim_h_Z'],
        config['lr']
    )
    model_y_path = os.path.join(path, f"{example_type}_Y_{n}.pt")
    model_z_path = os.path.join(path, f"{example_type}_Z_{n}.pt")
    solver.modelY.load_state_dict(torch.load(model_y_path, map_location=device) ,strict=False)
    solver.modelZ.load_state_dict(torch.load(model_z_path, map_location=device) ,strict=False)
    solver.modelY.eval()
    solver.modelZ.eval()
    future_models_Y[n] = solver.modelY
    future_models_Z[n] = solver.modelZ


if example_type != "reflected":
# =============== GENERATE TEST PATHS ===========
    batch_size = 1000
    delta_t = equation.T / N
    result = Result(equation, example_type)
    W = result.gen_b_motion(batch_size, N)
    x = result.gen_x(batch_size, N, W)  # Shape: [batch_size, dim_x, N+1]

    # Convert to numpy for analytical computation
    x_np = x.cpu().numpy()
    times = np.linspace(0, equation.T, N+1)
    #times = np.linspace(0, equation.T, N)

    # ========== Y VALIDATION ==========
    # Compute analytical Y
    Y_analytical = result.analytical_Y(times, x_np[:, :, :])  # Shape: [batch_size, 1, N+1]
    Y_analytical_torch = torch.from_numpy(Y_analytical).float().to(device)

    Y_predicted = result.predict_Y(x,N, future_models_Y)
    # Compute Y errors
    mse_per_timestep = ((Y_predicted - Y_analytical_torch) ** 2 ).mean(dim=0).squeeze().cpu().numpy()
    rel_err_per_timestep = (((Y_predicted - Y_analytical_torch) ** 2).mean(dim=0) /
                            (Y_analytical_torch ** 2).mean(dim=0).clamp(min=1e-8)).squeeze().cpu().numpy()
    total_mse_y = mse_per_timestep.mean()
    total_rel_err_y = rel_err_per_timestep.mean()
    max_err_y = (Y_predicted - Y_analytical_torch).abs().max().item()


# ========== Z VALIDATION ==========
    # Initialize Z arrays: [batch_size, dim_y, dim_d, N, N]
    # We'll simplify to first component: [batch_size, dim_d, N, N]
    Z_predicted = result.predict_Z(x,N, future_models_Z)

    # Compute analytical Z
    Z_analytical = result.analytical_Z(times[:-1], np.zeros((batch_size, equation.dim_x, N, N)), x_np[:,:, :], equation.T)
    Z_analytical_torch = torch.from_numpy(Z_analytical).float().to(device)

    # For comparison, average over dim_x dimension to match dim_d
    # Z_analytical is [batch_size, dim_x, N, N], we need [batch_size, dim_d, N, N]

    # Compute Z errors (only for valid t <= s pairs)
    valid_mask = torch.triu(torch.ones((N, N), device=device), diagonal=0).bool()
    Z_diff = (Z_predicted - Z_analytical_torch) ** 2
    Z_diff_masked = Z_diff[:, :, valid_mask]
    absolute_err_Z = Z_diff_masked.mean().item()
    total_mse_z = Z_diff_masked.mean().item()
    Z_analytical_sq = (Z_analytical_torch ** 2)[:, :, valid_mask]
    total_rel_err_z = (Z_diff_masked.mean() / Z_analytical_sq.mean().clamp(min=1e-8)).item()
    max_err_z = (Z_predicted - Z_analytical_torch).abs().max().item()


    # =============== PRINT METRICS ===========
    print(f"\n{'=' * 70}")
    print(f"VALIDATION METRICS")
    print(f"{'=' * 70}")
    print(f"Y - Overall MSE:         {total_mse_y:.6e}")
    print(f"Y - Overall Rel Error:   {total_rel_err_y:.6f}")
    print(f"{'-' * 70}")
    print(f"Z - Overall MSE:         {total_mse_z:.6e}")
    print(f"Z - Overall Rel Error:   {total_rel_err_z:.6f}")
    print(f"{'=' * 70}\n")

    # =============== SUMMARIZE ===========
    y = Y_predicted.cpu().numpy()
    Y = Y_analytical
    mask = np.triu(np.ones((N, N)), 0)
    z = Z_predicted.cpu().numpy()*mask
    Z = Z_analytical*mask


    #absolute and relative errors Y and Z per time step
    err_Y = ((Y - y)**2).mean(axis=(0,1))
    rel_Y = (err_Y / np.maximum(Y**2, 1e-8).mean(axis=(0,1)))

    err_Z = ((Z - z) ** 2).mean(axis=(0,1))  # shape [N, N]
    rel_Z = err_Z / ((Z ** 2).mean(axis=(0,1))+ 1e-8) # shape [N, N]


    #ABSOLUTE AND RELATIVE ERRORS Y and Z
    squared_diff = (np.linalg.norm(Z - z, axis=1) ** 2) # (M, N, N)
    absolute_err_Z = np.mean(np.sum(squared_diff, axis=(1, 2))) * (delta_t ** 2)
    squared_Z = np.linalg.norm(Z, axis=1) ** 2  # (M, N, N)
    relative_error_Z = absolute_err_Z / (np.mean(np.sum(squared_Z, axis=(1, 2))) * (delta_t ** 2))

    squared_diff = np.linalg.norm(Y - y, axis=1) ** 2  # (M, N)
    absolute_err_Y = np.mean(np.sum(squared_diff, axis=-1) * delta_t)
    squared_Y = np.linalg.norm(Y, axis=1) ** 2
    relative_error_Y = absolute_err_Y / (np.mean(np.sum(squared_Y, axis=-1)) * delta_t)

    print("Err(Y):", absolute_err_Y)
    print("Rel Err(Y):", relative_error_Y)
    print("Err(Z):", absolute_err_Z)
    print("Rel Err(Z):", relative_error_Z)


    # =============== PLOTTING ===========
    plotter = PlotGenerator(
        figures_path,
        path,
        example_type="linear1"
    )
    plotter.plot_y_absolute_err(times, err_Y, err_Y.mean(), filename= "Y_absolute_error")
    plotter.plot_y_trajectories(times, Y_analytical, Y_predicted, 10, "Y_samples")
    plotter.plot_z_surfaces(Z_analytical, Z_predicted,  T, 0, "Z_surfaces")
    plotter.plot_z_error_heatmap( err_Z,  T, "Z_heatmap")
    plotter.plot_z_error_surface(err_Z, T, "Z_error_surface")
    timestep = 0
    plotter.plot_loss_single_timestep(timestep, f"loss_timestep_{timestep}")
    timestep = 50
    plotter.plot_loss_single_timestep(timestep, f"loss_timestep_{timestep}")
    timestep = 25
    plotter.plot_loss_single_timestep(timestep, f"loss_timestep_{timestep}")
    plotter.plot_z_fixed_t(times, Z, z,0, T, "Z_fixed_t")
    plotter.plot_z_fixed_s(times, Z, z,0, T, "Z_fixed_s")

else:

    batch_size = 1000
    delta_t = equation.T / N
    result = Result(equation, example_type)
    W = result.gen_b_motion(batch_size, N)
    x = result.gen_x(batch_size, N, W)  # Shape: [batch_size, dim_x, N+1]

    x_transposed = x.transpose(1, 2)
    Y_predicted = torch.zeros((batch_size, 1, N + 1), device=device)
    with torch.no_grad():
        for n in range(N + 1):
            if n in future_models_Y:
                x_n = x_transposed[:, n, :]
                Y_predicted[:, :, n] = torch.max(future_models_Y[n](N, n, x_n), 0.05*torch.ones((batch_size,dim_y), device=device))

    # Convert to numpy for analytical computation
    x_np = x.cpu().numpy()
    times = np.linspace(0, equation.T, N + 1)
    L_np = equation.barrier(times, x).detach().numpy()*np.ones_like(times)
    Z_predicted = torch.zeros((batch_size, equation.dim_d, N, N), device=device)
    with torch.no_grad():
        for n in range(N):
            if n in future_models_Z:
                x_n = x_transposed[:, n, :]
                m_indices = torch.arange(n, N, device=device)
                x_future = x_transposed[:, n:N, :]  # [batch_size, N-n, dim_x]

                # z_batch shape: [batch_size, N-n, dim_y, dim_d]
                z_batch = future_models_Z[n](N, n, x_n, m_indices, x_future)

                # Store in Z_predicted: take first dim_y component
                for idx, m in enumerate(range(n, N)):
                    Z_predicted[:, :, n, m] = z_batch[:, idx, 0, :]  # [batch_size, dim_d]



    # =============== PLOTTING ===========
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    n_samples = 3
    sample_indices = np.random.choice(batch_size, n_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        ax.plot(times, L_np[idx, :], 'C0--', lw=2, alpha=0.7,
                label='Barrier' if i == 0 else None)
        ax.plot(times, Y_predicted[idx, 0, :], linestyle='-',color='red',alpha=0.8,
                label='Predicted' if i == 0 else None)

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$Y(t, X_t)$')
    #ax.set_title(f'Y Sample Trajectories (n={n_samples})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    x = np.linspace(0, T, N)
    y = np.linspace(0, T, N)
    X, Y = np.meshgrid(x, y)
    mask = np.triu(np.ones((N, N)), k=0)

    fig = plt.figure(figsize=(18, 5))  # wide figure for horizontal layout

    for i, idx in enumerate(sample_indices):
        ax = fig.add_subplot(1, n_samples, i + 1, projection='3d')
        z_plot = Z_predicted[idx, 0, :, :]  # first dim_y component
        z_masked = np.ma.array(z_plot, mask=(1 - mask))

        surf = ax.plot_surface(X, Y, z_masked, cmap='viridis', alpha=0.8, edgecolor='none')
        ax.set_xlabel('$s$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$Z$')
        #ax.set_title(f'Sample {idx}')

    plt.tight_layout()
    plt.show()

    plotter = PlotGenerator(
        figures_path,
        path,
        example_type="linear1"
    )
    timestep = 0
    plotter.plot_loss_single_timestep(timestep, f"loss_timestep_{timestep}")
    timestep = 50
    plotter.plot_loss_single_timestep(timestep, f"loss_timestep_{timestep}")
    timestep = 25
    plotter.plot_loss_single_timestep(timestep, f"loss_timestep_{timestep}")

