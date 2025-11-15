import torch
import torch.nn as nn
import numpy as np
import os
import wandb
import json

from BSVIE import Solver

if torch.cuda.is_available() and torch.version.hip is not None:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


class Result():

    def __init__(self,  equation, example_type):
        self.equation = equation
        self.example_type = example_type

    def gen_b_motion(self, batch_size, N):
        delta_t = self.equation.T / N
        W = torch.randn(batch_size, self.equation.dim_d, N, device=device) * np.sqrt(delta_t)
        return W

    def gen_x(self, batch_size, N, W):
        delta_t = self.equation.T / N
        x = torch.zeros(batch_size, N + 1, self.equation.dim_x, device=device).reshape(-1, self.equation.dim_x, N + 1)
        x[:, :, 0] = self.equation.x_0.view(1, self.equation.dim_x).expand(x.size(0), -1)

        for i in range(N):
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1)
            x_current = x[:, :, i]
            drift = self.equation.b(delta_t * i, x_current) * delta_t
            diffusion = torch.matmul(self.equation.sigma(delta_t * i, x_current), w).reshape(-1, self.equation.dim_x)
            x[:, :, i + 1] = x_current + drift + diffusion

        return x

    def predict_Y(self, x, N, models_Y_dict):
        """
        Generate predicted Y values using trained models

        Args:
            x: Tensor of shape [batch_size, dim_x, N+1] - forward paths
            N: Number of time steps
            models_Y_dict: Dictionary mapping time indices to Y models

        Returns:
            Y_predicted: Tensor of shape [batch_size, 1, N+1]
        """
        batch_size = x.shape[0]
        device = x.device

        # Transpose x to [batch_size, N+1, dim_x] for model input
        x_transposed = x.transpose(1, 2)

        Y_predicted = torch.zeros((batch_size, 1, N + 1), device=device)

        with torch.no_grad():
            for n in range(N + 1):
                if n in models_Y_dict:
                    x_n = x_transposed[:, n, :]
                    Y_predicted[:, :, n] = models_Y_dict[n](N, n, x_n)

        return Y_predicted

    def predict_Z(self, x, N, models_Z_dict):
        """
        Generate predicted Z values using trained models

        Args:
            x: Tensor of shape [batch_size, dim_x, N+1] - forward paths
            N: Number of time steps
            models_Z_dict: Dictionary mapping time indices to Z models

        Returns:
            Z_predicted: Tensor of shape [batch_size, dim_x, N, N]
        """
        batch_size = x.shape[0]
        dim_x = x.shape[1]
        device = x.device

        # Transpose x to [batch_size, N+1, dim_x] for model input
        x_transposed = x.transpose(1, 2)

        Z_predicted = torch.zeros((batch_size, dim_x, N, N), device=device)

        with torch.no_grad():
            for n in range(N):
                if n in models_Z_dict:
                    x_n = x_transposed[:, n, :]
                    m_indices = torch.arange(n, N, device=device)
                    x_future = x_transposed[:, n:N, :]  # [batch_size, N-n, dim_x]

                    # z_batch shape: [batch_size, N-n, dim_y, dim_d]
                    z_batch = models_Z_dict[n](N, n, x_n, m_indices, x_future)

                    # Store in Z_predicted: take first dim_y component
                    for idx, m in enumerate(range(n, N)):
                        Z_predicted[:, :, n, m] = z_batch[:, idx, 0, :]  # [batch_size, dim_d]

        return Z_predicted

    def analytical_Y(self, times, x):
        """Compute analytical Y values"""
        if self.example_type == "linear1":
            factor1 = np.sin(np.pi * times) + (-np.cos(np.pi * self.equation.T) + np.cos(np.pi * times)) / np.pi
            factor2 = (x + np.exp(self.equation.T) - np.exp(times)).mean(axis=1, keepdims=True)
            return factor1 * factor2

        elif self.example_type == "linear2":
            mu_vec = self.equation.mu.cpu().numpy()[:, None]
            times_np = np.array(times)[None, :]
            exp_term = np.exp(-self.equation.lam * times_np) * np.exp(mu_vec * (self.equation.T - times_np))
            integral_term = self.equation.lam0 * (np.exp(mu_vec * (self.equation.T - times_np)) - 1) / mu_vec
            return np.mean(x * (exp_term + integral_term)[None, :, :], axis=1)

        elif self.example_type in ["example1a", "nonlinear"]:
            sum_x = np.sum(x, axis=1, keepdims=True)
            Y = times * np.sin(sum_x)
            return Y

        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

    def analytical_Z(self, times, z, x_np, T):
        """Compute analytical Z values"""
        batch_size, dim_x, N, _ = z.shape
        z_analytical = np.zeros_like(z)

        if self.example_type == "linear1":
            pi = np.pi
            for t_idx in range(N):
                t = times[t_idx]
                sin_t = np.sin(pi * t)
                for s_idx in range(t_idx, N):
                    s = times[s_idx]
                    r_grid = np.linspace(s, T, 200)
                    dr = r_grid[1] - r_grid[0]
                    sin_r = np.sin(pi * r_grid)
                    inner_r = (-np.cos(pi * T) + np.cos(pi * r_grid)) / pi
                    integrand = np.exp(-(r_grid - t)) * (sin_r + inner_r)
                    integral_val = np.trapz(integrand, r_grid)
                    z_scalar = 1 / dim_x * (sin_t + integral_val)
                    for i in range(dim_x):
                        z_analytical[:, i, t_idx, s_idx] = z_scalar

        elif self.example_type == "linear2":
            mu_vec = self.equation.mu.cpu().numpy()
            sig_vec = self.equation.sig.cpu().numpy()
            for t_idx in range(N):
                t = times[t_idx]
                for s_idx in range(t_idx, N):
                    s = times[s_idx]
                    exp_term = np.exp(-self.equation.lam * t) * np.exp(mu_vec * (self.equation.T - s))
                    integral_term = self.equation.lam0 * (np.exp(mu_vec * (T - s)) - 1) / mu_vec
                    z_analytical[:, :, t_idx, s_idx] = 1 / self.equation.dim_x * sig_vec * x_np[:, :, s_idx] * (
                                exp_term + integral_term)

        elif self.example_type == "example1a":
            sum_x = np.sum(x_np, axis=1)
            for t_idx in range(N):
                for s_idx in range(N):
                    cos_term = np.cos(sum_x[:, s_idx])[:, None]
                    z_analytical[:, :, t_idx, s_idx] = times[t_idx] * self.equation.sig_base * cos_term * np.ones(
                        (1, dim_x))

        elif self.example_type == "nonlinear":
            sum_x = np.sum(x_np, axis=1)
            sig_vec = self.equation.sig.cpu().numpy()
            for t_idx in range(N):
                for s_idx in range(N):
                    cos_term = np.cos(sum_x[:, s_idx])[:, None]
                    # sigma_x = self.equation.sig_base * x_np[:, :, s_idx]
                    sigma_x = x_np[:, :, s_idx] * sig_vec[None, :]  # [batch_size, dim_x]
                    z_analytical[:, :, t_idx, s_idx] = times[t_idx] * cos_term * sigma_x

        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

        return z_analytical

def validate_against_analytical(equation, example_type, future_models_Y, future_models_Z, N, save_dir,
                                device=device):
    """Validate using Result class analytical methods - both Y and Z"""

    print(f"\n{'=' * 70}")
    print(f"VALIDATING AGAINST ANALYTICAL SOLUTION")
    print(f"{'=' * 70}")

    batch_size = 1000

    # Create Result object (use any model, just need for path generation)
    result = Result( equation, example_type)

    # Generate test paths using Result's method
    W = result.gen_b_motion(batch_size, N)
    x = result.gen_x(batch_size, N, W)  # Shape: [batch_size, dim_x, N+1]

    # Convert to numpy for analytical computation
    x_np = x.cpu().numpy()
    times = np.linspace(0, equation.T, N + 1)

    # ========== Y VALIDATION ==========
    # Compute analytical Y using Result's method
    Y_analytical = result.analytical_Y(times, x_np)  # Shape: [batch_size, 1, N+1]

    # Convert back to torch for comparison
    Y_analytical_torch = torch.from_numpy(Y_analytical).float().to(device)
    if Y_analytical_torch.ndim == 2:
        Y_analytical_torch = Y_analytical_torch.unsqueeze(1)  # [batch_size, 1, N+1]

    # Predict using trained models
    # x is [batch_size, dim_x, N+1], need to transpose to [batch_size, N+1, dim_x]
    x_transposed = x.transpose(1, 2)
    Y_predicted = torch.zeros((batch_size, 1, N + 1), device=device)

    with torch.no_grad():
        for n in range(N + 1):
            if n in future_models_Y:
                x_n = x_transposed[:, n, :]
                Y_predicted[:, :, n] = future_models_Y[n](N, n, x_n)

    # Compute Y errors
    mse_per_timestep = ((Y_predicted - Y_analytical_torch) ** 2).mean(dim=0).squeeze().cpu().numpy()
    total_mse_y = mse_per_timestep.mean()

    # ========== Z VALIDATION ==========
    print(f"\nComputing Z predictions and analytical values...")

    # Initialize Z arrays: [batch_size, dim_y, dim_d, N, N]
    # We'll simplify to first component: [batch_size, dim_d, N, N]
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

    # Compute analytical Z
    # Need to create a temporary z array for analytical_Z method
    z_temp = np.zeros((batch_size, equation.dim_x, N, N))
    Z_analytical = result.analytical_Z(times[:-1], z_temp, x_np, equation.T)
    Z_analytical_torch = torch.from_numpy(Z_analytical).float().to(device)

    # For comparison, average over dim_x dimension to match dim_d
    # Z_analytical is [batch_size, dim_x, N, N], we need [batch_size, dim_d, N, N]
    if equation.dim_d == equation.dim_x:
        Z_analytical_compare = Z_analytical_torch
    else:
        # If dimensions don't match, average
        Z_analytical_compare = Z_analytical_torch.mean(dim=1, keepdim=True).expand(-1, equation.dim_d, -1, -1)

    # Compute Z errors (only for valid t <= s pairs)
    valid_mask = torch.triu(torch.ones((N, N), device=device), diagonal=0).bool()
    Z_diff = (Z_predicted - Z_analytical_compare) ** 2
    Z_diff_masked = Z_diff[:, :, valid_mask]

    total_mse_z = Z_diff_masked.mean().item()

    # Log to wandb
    if wandb.run is not None:
        # Y metrics
        wandb.log({
            'validation/Y_total_mse': total_mse_y,
            'validation/Z_total_mse': total_mse_z,
        })

        for n in range(len(mse_per_timestep)):
            wandb.log({
                f'validation/Y_mse_timestep_{n}': mse_per_timestep[n]            })

        # Plot Y sample trajectories
        n_samples = min(5, batch_size)
        sample_indices = torch.randperm(batch_size)[:n_samples].cpu().numpy()
        times_np = times

        Y_analytical_np = Y_analytical_torch.cpu().numpy()
        Y_predicted_np = Y_predicted.cpu().numpy()

        for i, idx in enumerate(sample_indices):
            wandb.log({
                f'Y_comparison_sample_{i}': wandb.plot.line_series(
                    xs=times_np,
                    ys=[Y_predicted_np[idx, 0, :], Y_analytical_np[idx, 0, :]],
                    keys=["Y_predicted", "Y_analytical"],
                    title=f"Sample {idx}: Analytical vs Predicted Y",
                    xname="Time"
                )
            })

        # Plot Z slices
        Z_predicted_np = Z_predicted.cpu().numpy()
        Z_analytical_np = Z_analytical_compare.cpu().numpy()
        s_grid = times[:-1]  # N timesteps

        # Choose a few samples for Z plotting
        z_sample_indices = sample_indices[:min(3, len(sample_indices))]

        # Fixed s, varying t (slices at different s values)
        s_indices = [N // 4, N // 2, 3 * N // 4] if N > 4 else [N // 2]
        s_indices = [s_idx for s_idx in s_indices if s_idx < N]

        for s_idx in s_indices:
            for i, sample_idx in enumerate(z_sample_indices):
                # Full-length arrays (fill zeros where invalid)
                Z_t_full = np.zeros(N)
                z_t_full = np.zeros(N)

                # Valid range: t <= s_idx
                valid_t = slice(0, s_idx + 1)
                # Average over dim_d dimension for visualization
                Z_t_full[valid_t] = Z_analytical_np[sample_idx, :, :s_idx + 1, s_idx].mean(axis=0)
                z_t_full[valid_t] = Z_predicted_np[sample_idx, :, :s_idx + 1, s_idx].mean(axis=0)

                wandb.log({
                    f"Z_fixed_s{s_idx}_sample{i}": wandb.plot.line_series(
                        xs=s_grid,
                        ys=[z_t_full, Z_t_full],
                        keys=["Z_predicted", "Z_analytical"],
                        title=f"Fixed s={s_grid[s_idx]:.3f} — Sample {sample_idx}",
                        xname="t"
                    )
                })

        # Fixed t, varying s (slices at different t values)
        t_indices = [N // 4, N // 2, 3 * N // 4] if N > 4 else [N // 2]
        t_indices = [t_idx for t_idx in t_indices if t_idx < N]

        for t_idx in t_indices:
            for i, sample_idx in enumerate(z_sample_indices):
                # Full-length arrays (fill zeros where invalid)
                Z_s_full = np.zeros(N)
                z_s_full = np.zeros(N)

                # Valid range: s >= t_idx
                valid_s = slice(t_idx, N)
                # Average over dim_d dimension for visualization
                Z_s_full[valid_s] = Z_analytical_np[sample_idx, :, t_idx, t_idx:].mean(axis=0)
                z_s_full[valid_s] = Z_predicted_np[sample_idx, :, t_idx, t_idx:].mean(axis=0)

                wandb.log({
                    f"Z_fixed_t{t_idx}_sample{i}": wandb.plot.line_series(
                        xs=s_grid,
                        ys=[z_s_full, Z_s_full],
                        keys=["Z_predicted", "Z_analytical"],
                        title=f"Fixed t={s_grid[t_idx]:.3f} — Sample {sample_idx}",
                        xname="s"
                    )
                })

    print(f"\n{'=' * 70}")
    print(f"VALIDATION METRICS")
    print(f"{'=' * 70}")
    print(f"Y - Overall MSE:         {total_mse_y:.6e}")
    print(f"{'-' * 70}")
    print(f"Z - Overall MSE:         {total_mse_z:.6e}")
    print(f"{'=' * 70}\n")

    return {
        'y_mse': total_mse_y,
        'y_mse_per_timestep': mse_per_timestep,
        'z_mse': total_mse_z,
    }
