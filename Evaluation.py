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

        # Cache integrands at each step j
        b_cache = []  # [batch_size, dim_x]
        sdw_cache = []  # [batch_size, dim_x]

        x = self.equation.x_0.expand(batch_size, -1).clone()
        x_all = torch.zeros(batch_size, self.equation.dim_x, N + 1, device=device)
        x_all[:, :, 0] = x

        for j in range(N):
            t_j = delta_t * j
            t_next = delta_t * (j + 1)
            w_j = W[:, :, j].reshape(-1, self.equation.dim_d, 1)

            b_cache.append(self.equation.b(t_j, x) * delta_t)
            sdw_cache.append(torch.matmul(self.equation.sigma(t_j, x), w_j).reshape(-1, self.equation.dim_x))

            # X(t_{j+1}) = x_0 + sum_{i=0}^{j} K(t_{j+1}, t_i) * [b_i*dt + sigma_i*dW_i]
            x_new = self.equation.x_0.expand(batch_size, -1).clone()
            for i in range(j + 1):
                t_i = delta_t * i
                k = self.equation.kernel(t_next, t_i)
                x_new = x_new + k * b_cache[i] + k * sdw_cache[i]

            x = x_new
            x_all[:, :, j + 1] = x

        return x_all

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
            T = self.equation.T
            factor1 = np.sin(np.pi * times) + (-np.cos(np.pi * self.equation.T) + np.cos(np.pi * times)) / np.pi
            x_mean = x.mean(axis=1)  # mean over dim_x -> (1000, 51)

            factor2 = np.exp(-(T - times)) * x_mean +  (np.exp(2 * times - T) - np.exp(T)) / 2

            return factor1 * factor2


        elif self.example_type == "linear2":
            mu_vec = self.equation.mu.cpu().numpy()[:, None]
            times_np = np.array(times)[None, :]
            exp_term = np.exp(-self.equation.lam * times_np) * np.exp(mu_vec * (self.equation.T - times_np))
            integral_term = self.equation.lam0 * (np.exp(mu_vec * (self.equation.T - times_np)) - 1) / mu_vec
            return np.mean(x * (exp_term + integral_term)[None, :, :], axis=1)[:, None,:]

        elif self.example_type in [ "nonlinear"]:
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
                    # Z^j(t,s) = exp(-(T-s))/d * (sin(pi*t) + (cos(pi*max(t,s)) - cos(pi*T))/pi)
                    # since s >= t in this loop, max(t,s) = s
                    cos_term = (np.cos(pi * s) - np.cos(pi * T)) / pi
                    z_scalar = np.exp(-(T - s)) / dim_x * (sin_t + cos_term)
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
                    #sigma_x = x_np[:, :, s_idx] * sig_vec[None, :]  # [batch_size, dim_x]
                    z_analytical[:, :, t_idx, s_idx] = times[t_idx] * cos_term * sig_vec[None,:] #sigma_x

        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

        return z_analytical

def validate_against_analytical(equation, example_type, future_models_Y, future_models_Z, N, save_dir,
                                device=device):

    print(f"\n{'=' * 70}")
    print(f"VALIDATING AGAINST ANALYTICAL SOLUTION")
    print(f"{'=' * 70}")

    batch_size = 1000

    result = Result( equation, example_type)

    W = result.gen_b_motion(batch_size, N)
    x = result.gen_x(batch_size, N, W)  # Shape: [batch_size, dim_x, N+1]

    x_np = x.cpu().numpy()
    times = np.linspace(0, equation.T, N + 1)

    print(times.shape)  # should be [batch, 1] or [1, N]
    print(x_np.shape)

    Y_analytical = result.analytical_Y(times, x_np)  # Shape: [batch_size, 1, N+1]
    Y_predicted = result.predict_Y(x, N, future_models_Y)  # [batch_size, 1, N+1]

    Y_predicted_np = Y_predicted.cpu().numpy()

    # Compute Y errors
    mse_per_timestep = ((Y_predicted_np - Y_analytical) ** 2).mean(axis=0).squeeze()
    total_mse_y = mse_per_timestep.mean()

    Z_predicted = result.predict_Z(x, N, future_models_Z)  # [batch_size, dim_x, N, N]
    Z_predicted_np = Z_predicted.cpu().numpy()

    # Compute analytical Z
    # Need to create a temporary z array for analytical_Z method
    z_temp = np.zeros((batch_size, equation.dim_x, N, N))
    Z_analytical = result.analytical_Z(times[:-1], z_temp, x_np, equation.T)

    # For comparison, average over dim_x dimension to match dim_d
    # Z_analytical is [batch_size, dim_x, N, N], we need [batch_size, dim_d, N, N]
    if equation.dim_d == equation.dim_x:
        Z_analytical_compare = Z_analytical
    else:
        # If dimensions don't match, average
        Z_analytical_avg = Z_analytical.mean(axis=1, keepdims=True)
        Z_analytical_compare = np.repeat(Z_analytical_avg, equation.dim_d, axis=1)

    # Compute Z errors (only for valid t <= s pairs)
    valid_mask = np.triu(np.ones((N, N), dtype=bool), k=0)
    Z_diff = (Z_predicted_np - Z_analytical_compare) ** 2
    Z_diff_masked = Z_diff[:, :, valid_mask]

    total_mse_z = Z_diff_masked.mean().item()


    if equation.USE_WANDB:
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

        for i, idx in enumerate(sample_indices):
            wandb.log({
                f'Y_comparison_sample_{i}': wandb.plot.line_series(
                    xs=times_np,
                    ys=[Y_predicted_np[idx, 0, :], Y_analytical[idx, 0, :]],
                    keys=["Y_predicted", "Y_analytical"],
                    title=f"Sample {idx}: Analytical vs Predicted Y",
                    xname="Time"
                )
            })

        # Plot Z slices
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
                Z_t_full[valid_t] = Z_analytical[sample_idx, :, :s_idx + 1, s_idx].mean(axis=0)
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
                Z_s_full[valid_s] = Z_analytical[sample_idx, :, t_idx, t_idx:].mean(axis=0)
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


    else:

        import matplotlib.pyplot as plt
        import os
        save_path = os.path.join(save_dir, "figures")
        os.makedirs(save_path, exist_ok=True)
        n_samples = min(5, batch_size)
        sample_indices = torch.randperm(batch_size)[:n_samples].cpu().numpy()

        times_np = times
        s_grid = times[:-1]


        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))
        for i, idx in enumerate(sample_indices):
            c = colors[i]
            # predicted = solid
            plt.plot(times_np, x_np[idx, 0, :],
                     color=c, linestyle='-', alpha=0.9)

        ########## Y
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_indices)))
        for i, idx in enumerate(sample_indices):
            c = colors[i]
            # predicted = solid
            plt.plot(times_np, Y_predicted_np[idx, 0, :],
                     color=c, linestyle='-', alpha=0.9)

            # analytical = dashed SAME color
            plt.plot(times_np, Y_analytical[idx, :],
                     color=c, linestyle='--', alpha=0.9)
        plt.title("Y: Predicted vs Analytical (all samples)")
        plt.xlabel("Time")
        plt.ylabel("Y")
        plt.plot([], [], '-', label="Predicted (solid)")
        plt.plot([], [], '--', label="Analytical (dashed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "Y_comparison_all_samples.pdf"))
        plt.show()
        plt.close()


        z_sample_indices = sample_indices[:min(3, len(sample_indices))]
        sample_idx = z_sample_indices[0]  # ONLY ONE SAMPLE
        s_indices = [N // 4, N // 2, 3 * N // 4] if N > 4 else [N // 2]
        s_indices = [s_idx for s_idx in s_indices if s_idx < N]

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(s_indices)))
        for i, s_idx in enumerate(s_indices):
            c = colors[i]
            s_val = s_grid[s_idx]
            Z_t_full = np.zeros(N)
            z_t_full = np.zeros(N)
            valid_t = slice(0, s_idx + 1)
            Z_t_full[valid_t] = Z_analytical[sample_idx, :, :s_idx + 1, s_idx].mean(axis=0)
            z_t_full[valid_t] = Z_predicted_np[sample_idx, :, :s_idx + 1, s_idx].mean(axis=0)
            # predicted solid
            plt.plot(s_grid, z_t_full, color=c, linestyle='-', label=f"s={s_val:.3f}")
            # analytical dashed same color
            plt.plot(s_grid, Z_t_full, color=c, linestyle='--')
        plt.title(f"Z fixed s (sample {sample_idx})")
        plt.xlabel("t")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "Z_fixed_s_ONE_SAMPLE.pdf"))
        plt.show()
        plt.close()

        # ======================================================

        # Z PLOTS - FIXED t (ALL SAMPLES IN SAME FIGURE)

        # ======================================================

        t_indices = [N // 4, N // 2, 3 * N // 4] if N > 4 else [N // 2]
        t_indices = [t_idx for t_idx in t_indices if t_idx < N]
        sample_idx = z_sample_indices[0]  # SAME SINGLE SAMPLE

        plt.figure(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(t_indices)))

        for i, t_idx in enumerate(t_indices):
            c = colors[i]
            t_val = s_grid[t_idx]

            Z_s_full = np.zeros(N)
            z_s_full = np.zeros(N)

            valid_s = slice(t_idx, N)

            Z_s_full[valid_s] = Z_analytical[sample_idx, :, t_idx, t_idx:].mean(axis=0)
            z_s_full[valid_s] = Z_predicted_np[sample_idx, :, t_idx, t_idx:].mean(axis=0)

            # predicted solid
            plt.plot(s_grid, z_s_full, color=c, linestyle='-', label=f"t={t_val:.3f}")

            # analytical dashed
            plt.plot(s_grid, Z_s_full, color=c, linestyle='--')

        plt.title(f"Z fixed t (sample {sample_idx})")
        plt.xlabel("s")

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "Z_fixed_t_ONE_SAMPLE.pdf"))
        plt.show()
        plt.close()

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        # ======================================================
        # Z SURFACE PLOT (PREDICTED VS ANALYTICAL)
        # ======================================================
        z_sample_indices = sample_indices[:min(2, len(sample_indices))]  # keep it readable
        t_grid = np.linspace(0, equation.T, N)
        s_grid = np.linspace(0, equation.T, N)
        T_mesh, S_mesh = np.meshgrid(t_grid, s_grid, indexing="ij")

        for sample_idx in z_sample_indices:
            Z_pred = Z_predicted_np[sample_idx, 0, :, :]
            Z_true = Z_analytical[sample_idx, 0, :, :]

            # mask invalid region t > s
            mask = T_mesh > S_mesh
            Z_pred = Z_pred.copy()
            Z_true = Z_true.copy()

            Z_pred[mask] = np.nan
            Z_true[mask] = np.nan

            fig = plt.figure(figsize=(11, 7))
            ax = fig.add_subplot(111, projection='3d')

            # predicted surface
            ax.plot_surface(
                T_mesh,
                S_mesh,
                Z_pred,
                cmap="viridis",
                alpha=0.7,
                linewidth=0,
                antialiased=True
            )

            # analytical surface (slightly transparent overlay)
            ax.plot_surface(
                T_mesh,
                S_mesh,
                Z_true,
                cmap="plasma",
                alpha=0.5,
                linewidth=0,
                antialiased=True
            )

            ax.set_title(f"Z surface: Predicted vs Analytical (sample {sample_idx})")
            ax.set_xlabel("t")
            ax.set_ylabel("s")
            ax.set_zlabel("Z")

            # optional: legend workaround (matplotlib 3D doesn't support it directly)
            import matplotlib.patches as mpatches
            pred_patch = mpatches.Patch(color="green", label="Predicted (viridis)")
            true_patch = mpatches.Patch(color="red", label="Analytical (plasma)")
            ax.legend(handles=[pred_patch, true_patch])

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"Z_surface_compare_{sample_idx}.pdf"))
            plt.show()
            plt.close()

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
