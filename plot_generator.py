import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import os
import json
import math


class PlotGenerator:

    def __init__(self, save_folder, path, example_type):
        """
            save_folder: Directory to save plots
            example_type: Prefix for saved files
        """
        self.save_folder = save_folder
        self.loss_path = path
        self.example_type = example_type
        self._set_plot_style()

    def _set_plot_style(self):
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'text.usetex': False,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 20,
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'figure.dpi': 150
        })

    def _save_figure(self, fig, filename):
        os.makedirs(self.save_folder, exist_ok=True)
        pdf_path = os.path.join(self.save_folder, f'{filename}.pdf')
        png_path = os.path.join(self.save_folder, f'{filename}.png')
        fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
        fig.savefig(png_path, bbox_inches='tight', dpi=300)

    # =========================================================================
    # Y APPROXIMATION PLOTS
    # =========================================================================

    def plot_y_absolute_err(self, times, mse_per_timestep,
                      total_mse_y, filename=None):
        """
        Plot MSE and relative error per timestep for Y.

        Args:
            times: Time array
            mse_per_timestep: MSE at each timestep
            rel_err_per_timestep: Relative error at each timestep
            total_mse_y: Mean MSE across all timesteps
            filename: Custom filename (optional)
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.plot(times[:], mse_per_timestep[:], 'o-', color='C0',
                label='MSE per timestep')
        ax.set_yscale('log')
        ax.set_ylabel('Error (log scale)')
        ax.set_xlabel('Time $t$')
        #ax.set_title('Y Errors Over Time')

        ax.axhline(total_mse_y, color='C0', linestyle='--', alpha=0.5,
                   label=f'Mean MSE={total_mse_y:.2e}')

        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = filename or f'{self.example_type}_Y_errors'
        self._save_figure(fig, fname)
        plt.show()
        plt.close(fig)

    def plot_y_errors(self, times, mse_per_timestep, rel_err_per_timestep,
                      total_mse_y, filename=None):
        """
        Plot MSE and relative error per timestep for Y.

        Args:
            times: Time array
            mse_per_timestep: MSE at each timestep
            rel_err_per_timestep: Relative error at each timestep
            total_mse_y: Mean MSE across all timesteps
            filename: Custom filename (optional)
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.plot(times[:], mse_per_timestep[:], 'o-', color='C0',
                label='MSE per timestep')
        ax.plot(times[:-1], rel_err_per_timestep[:-1], 's-', color='C1',
                label='Relative error')
        ax.set_yscale('log')
        ax.set_ylabel('Error (log scale)')
        ax.set_xlabel('Time $t$')
        #ax.set_title('Y Errors Over Time')

        ax.axhline(total_mse_y, color='C0', linestyle='--', alpha=0.5,
                   label=f'Mean MSE={total_mse_y:.2e}')

        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = filename or f'{self.example_type}_Y_errors'
        self._save_figure(fig, fname)
        plt.show()
        plt.close(fig)

    def plot_y_trajectories(self, times, Y_analytical, Y_predicted,
                            n_samples, filename=None):
        """
        Plot sample Y trajectories comparing analytical vs predicted.

        Args:
            times: Time array
            Y_analytical: Analytical solution [batch, dim, time]
            Y_predicted: Predicted solution [batch, dim, time]
            n_samples: Number of sample trajectories to plot
            filename: Custom filename (optional)
        """

        if hasattr(Y_analytical, 'cpu'):
            Y_analytical = Y_analytical.cpu().numpy()
        if hasattr(Y_predicted, 'cpu'):
            Y_predicted = Y_predicted.cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        batch_size = Y_analytical.shape[0]
        n_samples = min(n_samples, batch_size)
        sample_indices = np.random.choice(batch_size, n_samples, replace=False)

        for i, idx in enumerate(sample_indices):
            ax.plot(times, Y_analytical[idx, 0, :], 'C0--', lw=2, alpha=0.7,
                    label='Analytical' if i == 0 else None)
            ax.plot(times, Y_predicted[idx, 0, :], linestyle='-',color='red',alpha=0.8,
                    label='Predicted' if i == 0 else None)

        ax.set_xlabel('Time $t$')
        ax.set_ylabel('$Y(t, X_t)$')
        #ax.set_title(f'Y Sample Trajectories (n={n_samples})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = filename or f'{self.example_type}_Y_trajectories'
        self._save_figure(fig, fname)
        plt.show()
        plt.close(fig)

    # =========================================================================
    # Z APPROXIMATION PLOTS (SEPARATED)
    # =========================================================================

    def plot_z_surfaces(self, Z_analytical, Z_predicted, T, dimension, filename=None):
        """
        Plot 3D surface overlap of Z analytical vs predicted.

        Args:
            Z_analytical: Analytical Z [batch, dim, N, N]
            Z_predicted: Predicted Z [batch, dim, N, N]
            T: Terminal time
            dimension: dimension index to plot
            filename: Custom filename (optional)
        """

        if hasattr(Z_analytical, 'cpu'):
            Z_analytical = Z_analytical.cpu().numpy()
        if hasattr(Z_predicted, 'cpu'):
            Z_predicted = Z_predicted.cpu().numpy()

        # Pick random sample
        sample_idx = np.random.choice(Z_analytical.shape[0])
        Z_plot = Z_analytical[sample_idx, dimension, :, :]
        z_plot = Z_predicted[sample_idx, dimension, :, :]

        N = Z_plot.shape[0]
        mask = np.triu(np.ones_like(Z_plot), k=0)

        x = np.linspace(0, T, N)
        y = np.linspace(0, T, N)
        X, Y = np.meshgrid(x, y)

        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        Z_masked = np.ma.array(Z_plot, mask=(1 - mask))
        z_masked = np.ma.array(z_plot, mask=(1 - mask))

        ax.plot_surface(X, Y, Z_masked, cmap='plasma', alpha=0.7,  edgecolor='none',
                        label='Analytical')
        ax.plot_surface(X, Y, z_masked, cmap='viridis', alpha=0.7, edgecolor='none',
                        label='Predicted')
        ax.set_xlabel('$s$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$Z$')
        #ax.set_title(f'Z Surfaces component {dimension}: Analytical vs Predicted')

        plt.tight_layout()
        fname = filename or f'{self.example_type}_Z_surfaces'
        self._save_figure(fig, fname)
        plt.show()
        plt.close(fig)

    def plot_z_error_heatmap(self, abs_error, T, filename):

        # Move from torch to numpy if needed
        if hasattr(abs_error, 'cpu'):
            abs_error = abs_error.cpu().numpy()

        abs_plot = abs_error

        N = abs_plot.shape[0]
        mask = np.triu(np.ones_like(abs_plot), k=0)

        fig, ax = plt.subplots(figsize=(7, 5))

        # --- Absolute error ---
        abs_error_masked = np.ma.array(abs_plot, mask=(1 - mask))
        im = ax.imshow(abs_error_masked, aspect='auto', origin='lower',
                       cmap='YlOrRd', extent=[0, T, 0, T])

        ax.set_xlabel('$s$')
        ax.set_ylabel('$t$')
        #ax.set_title('Absolute Error')

        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        fname = filename or f'{self.example_type}_Z_abs_error_heatmap'
        self._save_figure(fig, fname)

        plt.show()
        plt.close(fig)

    def plot_z_error_surface(self, abs_error, T, filename):
        # Move from torch to numpy if needed
        if hasattr(abs_error, 'cpu'):
            abs_error = abs_error.cpu().numpy()

        abs_plot = abs_error
        N = abs_plot.shape[0]
        mask = np.triu(np.ones_like(abs_plot), k=0)  # upper-triangular mask

        # Create grid
        t = np.linspace(0, T, N)
        s = np.linspace(0, T, N)
        S, Tm = np.meshgrid(s, t)

        # Mask the lower-triangular region
        abs_error_masked = np.ma.array(abs_plot, mask=(1 - mask))

        # --- Plot ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Surface plot
        surf = ax.plot_surface(
            S, Tm, abs_error_masked,
            cmap='YlOrRd', edgecolor='none', linewidth=0,
            antialiased=True
        )

        ax.set_xlabel('$s$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('Absolute Error')
        #ax.set_title('Absolute Error Surface')

        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
        plt.tight_layout()

        fname = filename or f'{self.example_type}_Z_abs_error_surface'
        self._save_figure(fig, fname)

        plt.show()
        plt.close(fig)

    def plot_z_slices(self, times, Z_analytical, Z_predicted, dimension, T, filename=None):

        if hasattr(Z_analytical, 'cpu'):
            Z_analytical = Z_analytical.cpu().numpy()
        if hasattr(Z_predicted, 'cpu'):
            Z_predicted = Z_predicted.cpu().numpy()

        sample_idx = np.random.choice(Z_analytical.shape[0])
        Z_plot = Z_analytical[sample_idx, dimension, :, :]
        z_plot = Z_predicted[sample_idx, dimension, :, :]
        N = Z_plot.shape[0]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Fixed s slices
        ax = axes[0]
        s_indices = [N // 4, N // 2, 3 * N // 4]
        for s_idx in s_indices:
            if s_idx < N:
                valid_t = slice(0, s_idx + 1)
                ax.plot(times[:-1][valid_t], Z_plot[valid_t, s_idx],
                        '--', label=f's={times[s_idx]:.2f}', markersize=4)
                ax.plot(times[:-1][valid_t], z_plot[valid_t, s_idx],
                        'o-', alpha=0.7)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$Z(t,s)$')
        #ax.set_title('Fixed $s$ Slices')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Fixed t slices
        ax = axes[1]
        t_indices = [N // 4, N // 2, 3 * N // 4]
        for t_idx in t_indices:
            if t_idx < N:
                valid_s = slice(t_idx, N - 1)
                ax.plot(times[:-1][valid_s], Z_plot[t_idx, valid_s],
                        's-', label=f't={times[t_idx]:.2f}', markersize=4)
                ax.plot(times[:-1][valid_s], z_plot[t_idx, valid_s],
                        '--', alpha=0.7)
        ax.set_xlabel('$s$')
        ax.set_ylabel('$Z(t,s)$')
        #ax.set_title('Fixed $t$ Slices')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Diagonal elements
        ax = axes[2]
        diagonal_analytical = np.array([Z_plot[i, i] for i in range(N)])
        diagonal_predicted = np.array([z_plot[i, i] for i in range(N)])
        ax.plot(times[:-1], diagonal_analytical, 'k-', linewidth=2,
                label='Analytical')
        ax.plot(times[:-1], diagonal_predicted, 'C0--', linewidth=2,
                label='Predicted')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$Z(t,t)$')
        #ax.set_title('Diagonal Elements')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = filename or f'{self.example_type}_Z_slices'
        self._save_figure(fig, fname)
        plt.show()
        plt.close(fig)

    def plot_z_fixed_t(self, times, Z_analytical, Z_predicted, dimension, T, filename=None):

        if hasattr(Z_analytical, 'cpu'):
            Z_analytical = Z_analytical.cpu().numpy()
        if hasattr(Z_predicted, 'cpu'):
            Z_predicted = Z_predicted.cpu().numpy()

        sample_idx = np.random.choice(Z_analytical.shape[0])
        Z_plot = Z_analytical[sample_idx, dimension, :, :]
        z_plot = Z_predicted[sample_idx, dimension, :, :]
        N = Z_plot.shape[0]

        s_grid = np.linspace(0, T, N)  # Shape: (N,)
        t_indices = [5, 10, 20]
        colors = ['b', 'g', 'r', 'c', 'm']
        labels = [f"t = {np.round(s_grid[t_idx], 2)}" for t_idx in t_indices]

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, t_idx in enumerate(t_indices):
            # For fixed t = t_idx, we plot Z(t, s) for s >= t
            s_subgrid = s_grid[:]
            z_s = z_plot[t_idx, :]  # predicted Z(t, s)
            Z_s = Z_plot[t_idx, :]  # analytical Z(t, s)

            color = colors[i % len(colors)]
            label_t = np.round(s_grid[t_idx], 2)

            ax.plot(s_subgrid, z_s, color=color, linestyle='-', linewidth=2,
                    label=f'Predicted Z(t={label_t}, s)')
            ax.plot(s_subgrid, Z_s, color=color, linestyle='--', linewidth=2,
                    label=f'Analytical Z(t={label_t}, s)')

        ax.set_xlabel('$s$', fontsize=12)
        ax.set_ylabel('$Z(t,s)$', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = filename or f'{self.example_type}_Z_fixed_t_comparison'
        self._save_figure(fig, fname)
        plt.show()
        plt.close(fig)

    def plot_z_fixed_s(self, times, Z_analytical, Z_predicted, dimension, T, filename=None):

        if hasattr(Z_analytical, 'cpu'):
            Z_analytical = Z_analytical.cpu().numpy()
        if hasattr(Z_predicted, 'cpu'):
            Z_predicted = Z_predicted.cpu().numpy()

        sample_idx = np.random.choice(Z_analytical.shape[0])
        Z_plot = Z_analytical[sample_idx, dimension, :, :]
        z_plot = Z_predicted[sample_idx, dimension, :, :]
        N = Z_plot.shape[0]

        s_grid = np.linspace(0, T, N)  # Shape: (N,)
        s_indices = [5, 20, 30]
        colors = ['b', 'g', 'r', 'c', 'm']

        fig, ax = plt.subplots(figsize=(8, 6))

        for i, s_idx in enumerate(s_indices):
            # For fixed t = t_idx, we plot Z(t, s) for s >= t
            s_subgrid = s_grid[ :]
            z_s = z_plot[:, s_idx]  # predicted Z(t, s)
            Z_s = Z_plot[:, s_idx]  # analytical Z(t, s)

            color = colors[i % len(colors)]
            label_s = np.round(s_grid[s_idx], 2)

            ax.plot(s_subgrid, z_s, color=color, linestyle='-', linewidth=2,
                    label=f'Predicted Z(t, s={label_s})')
            ax.plot(s_subgrid, Z_s, color=color, linestyle='--', linewidth=2,
                    label=f'Analytical Z(t, s={label_s})')

        ax.set_xlabel('$s$', fontsize=12)
        ax.set_ylabel('$Z(t,s)$', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = filename or f'{self.example_type}_Z_fixed_s_comparison'
        self._save_figure(fig, fname)
        plt.show()
        plt.close(fig)

    # =========================================================================
    # LOSS PLOTS
    # =========================================================================

    def plot_loss_single_timestep(self, timestep, filename,
                                  ):

        loss_file = os.path.join(self.loss_path,
                                         f"loss_history_timestep_{timestep}.json")

        if not os.path.exists(loss_file):
            print(f"Warning: File not found: {loss_file}")
            return

        with open(loss_file, "r") as f:
            data = json.load(f)
        loss_data = np.array(data.get("loss_per_iteration", []))

        if len(loss_data) == 50:
            print(f"Warning: No loss data for timestep {timestep}")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if timestep == 50:
            loss_data = loss_data[:1000]

        iterations = np.arange(1, len(loss_data) + 1)


        ax.semilogy(iterations, loss_data, color='C0', lw=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        #
        #ax.set_title(f'Timestep {timestep}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = filename or f'{self.example_type}_loss_timestep_{timestep}'
        self._save_figure(fig, fname)
        plt.show()
        plt.close(fig)

