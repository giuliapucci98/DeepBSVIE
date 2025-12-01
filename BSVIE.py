import torch
import torch.nn as nn
import numpy as np
import os
import wandb
import json

if torch.cuda.is_available() and torch.version.hip is not None:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


class volterra_fbsde():
    def __init__(self, x_0, mu_base, sig_base, lam, lam0, T, dim_x, dim_y, dim_d, example_type, seed=42):
        self.x_0 = x_0
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d
        self.mu_base = mu_base
        self.sig_base = sig_base
        self.lam = lam
        self.lam0 = lam0
        self.example_type = example_type

        i = torch.linspace(-1, 1, dim_x, device=device)
        self.mu = self.mu_base * (1 + 0.3 * i)
        self.sig = self.sig_base * (1 + 0.2 * i)
        # self.mu = self.mu_base * torch.ones(dim_x, device=device) # constant mu
        # self.sig = self.sig_base * torch.ones(dim_x, device=device) # constant sigma


    def b(self, t, x):
        if self.example_type == "linear1":
            return torch.zeros_like(x)  # Simple Brownian Motion
        if self.example_type in ["linear2", "reflected", "nonlinear"]:
            #return self.mu * x  # GBM
            return self.mu 
        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

    def sigma(self, t, x):
        batch_size, dim_x = x.shape
        if self.example_type in ["linear2", "reflected", "nonlinear"]:
            sig_matrix = torch.diag(self.sig).unsqueeze(0).expand(batch_size, -1, -1)
            #return sig_matrix * x.unsqueeze(-1)  # scale by x
            return sig_matrix
        elif self.example_type in ["linear1"]:
            sig_matrix = (torch.eye(dim_x, device=x.device) * self.sig_base).unsqueeze(0).expand(batch_size, -1, -1)  # constant sigma matrix
            return sig_matrix
        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

    def f_vectorized(self, t_n, s_array, x_batch, y_batch, z_batch):
        '''        t_n: scalar - current time
        s_array: [num_steps] - array of future times
        x_batch: [batch_size, num_steps, dim_x] - future states
        y_batch: [batch_size, num_steps, dim_y] - future y values
        z_batch: [batch_size, num_steps, dim_y, dim_d] - future z values
        Returns: [batch_size, num_steps, dim_y] '''
        if self.example_type == "linear1":
            s_expanded = s_array.view(1, -1, 1)
            indicator = (s_expanded >= t_n).to(y_batch.dtype)
            exp_term = torch.exp(-(s_expanded - t_n))
            term1 = exp_term * indicator * y_batch
            term2 = torch.exp(s_expanded) * z_batch.sum(dim=-1)
            return term1 + term2

        elif self.example_type == "linear2":
            return self.lam0 * x_batch.mean(dim=2, keepdim=True)

        elif self.example_type == "nonlinear":
            sum_x = x_batch.sum(dim=-1, keepdim=True)  # [batch_size, num_steps, 1]
            sigma_x = self.sig.view(1, 1, -1)# * x_batch  # x_batch: [B, S, dim_x], -> [B, S, dim_x, 1]
            norm_sigma_x_sq = (sigma_x.squeeze(-1) ** 2).sum(dim=-1, keepdim=True)  # [B, S, 1]
            term1 = 0.5 * t_n * torch.sin(sum_x) * norm_sigma_x_sq
            # term2: mu^T * sigma^{-1} * Z
            # mu / sigma: [dim_x]
            mu_over_sigma = (self.mu / self.sig).view(1, 1, 1, -1)  # [1,1,1,dim_x]
            # z_batch: [B, S, dim_y, dim_d], dim_d should match dim_x
            term2 = (mu_over_sigma * z_batch).sum(dim=-1)  # sum over last dim, [B,S,dim_y]
            return term1 - term2

        elif self.example_type == "reflected":
            return torch.zeros_like(y_batch)

        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

    def g(self, t, x):
        if self.example_type == "linear1":
            return np.sin(np.pi * t) * (x.sum(dim=-1, keepdim=True) / self.dim_x)
        elif self.example_type == "linear2":
            return np.exp(-self.lam0 * t) * (x.mean(dim=1, keepdim=True))
        elif self.example_type == "nonlinear":
            sum_x = x.sum(dim=-1, keepdim=True)
            return t * torch.sin(sum_x).expand(-1, self.dim_y)
        elif self.example_type == "reflected":
            discount = 1 / (1 + (self.T - t))
            return discount * torch.relu(x.mean(dim=-1, keepdim=True) - 1.0)

        else:
            raise ValueError(f"Unknown example_type: {self.example_type}")

    def barrier(self, t, x):
        return 0.05 * torch.ones(x.size(0), self.dim_y, device=x.device)


class NN_Y(nn.Module):
    def __init__(self, equation, dim_h):
        super(NN_Y, self).__init__()

        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y)

        self.equation = equation

    def forward(self, N, n, x):
        """
        x: [batch_size, dim_x] - state at time n
        Returns: [batch_size, dim_y]
        """

        def phi(x):
            h1 = torch.tanh(self.linear1(x))
            h2 = torch.tanh(self.linear2(h1))
            h3 = torch.tanh(self.linear3(h2))
            # h4 = torch.tanh(self.linear4(h3 + h1))
            return self.linear4(h3)  # [bs,dy] -> [bs,dy]

        delta_t = self.equation.T / N
        x_nor = x

        inpt = torch.cat((x_nor, torch.ones(x.size(0), 1, device=device) * delta_t * n), 1)
        y = phi(inpt)
        return y


class NN_Z(nn.Module):
    """
            xt: [batch_size, dim_x] - current state at time n
            m_indices: [num_steps] - timestep indices [n, n+1, ..., N-1]
            xs: [batch_size, num_steps, dim_x] - future states
            Returns: [batch_size, num_steps, dim_y, dim_d]
    """

    def __init__(self, equation, dim_h):
        super(NN_Z, self).__init__()

        self.linear1 = nn.Linear(2 * equation.dim_x + 2, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y * equation.dim_d)

        self.equation = equation

    def forward(self, N, n, xt, m_indices, xs):

        def phi(x):
            h1 = torch.tanh(self.linear1(x))
            h2 = torch.tanh(self.linear2(h1))
            h3 = torch.tanh(self.linear3(h2))
            return self.linear4(h3)  # [bs  # [bs,dy*dd] -> [bs,dy*dd]

        # xt: [batch_size, dim_x] - current state at time n
        # m_indices: [num_steps] - timestep indices[n, n + 1, ..., N - 1]
        # xs: [batch_size, num_steps, dim_x] - future states
        # Returns: [batch_size, num_steps, dim_y, dim_d]

        delta_t = self.equation.T / N
        batch_size = xt.shape[0]
        num_steps = m_indices.shape[0]

        xt_nor = xt
        xs_nor = xs

        # Expand xt to match all timesteps: [batch_size, num_steps, dim_x]
        xt_expanded = xt_nor.unsqueeze(1).expand(-1, num_steps, -1)
        # Flatten for network: [batch_size * num_steps, dim_x]
        xt_flat = xt_expanded.reshape(-1, self.equation.dim_x)
        xs_flat = xs_nor.reshape(-1, self.equation.dim_x)

        # Time encodings: [batch_size * num_steps, 1]
        t_n = torch.full((batch_size * num_steps, 1), delta_t * n, device=device)
        t_m_values = delta_t * m_indices  # [num_steps]
        t_m = t_m_values.unsqueeze(0).expand(batch_size, -1).reshape(-1, 1)

        # Single forward pass through network
        inpt = torch.cat((xt_flat, t_n, xs_flat, t_m), 1)
        z = phi(inpt).reshape(batch_size, num_steps, self.equation.dim_y, self.equation.dim_d)

        return z


class EarlyStoppingChecker:
    def __init__(self, patience, min_iterations, rel_tolerance, window_size,
                 low_loss_threshold=0.001, low_loss_patience=30):
        self.patience = patience  # How long to wait after best loss
        self.min_iterations = min_iterations  # Train at least this many iterations
        self.rel_tolerance = rel_tolerance  # Minimum improvement to count as "better"
        self.window_size = window_size  # Smooth loss over this many iterations
        self.low_loss_threshold = low_loss_threshold  # Stop if loss stays below this
        self.low_loss_patience = low_loss_patience  # Iterations to stay below threshold

        self.loss_history = []
        self.best_loss = float('inf')
        self.best_iteration = 0
        self.low_loss_counter = 0  # Count iterations below threshold

    def check(self, iteration, current_loss):
        self.loss_history.append(current_loss)

        # Always train at least min_iterations
        if iteration < self.min_iterations:
            return False

        # Check if loss is below threshold and stable
        if current_loss < self.low_loss_threshold:
            self.low_loss_counter += 1
            if self.low_loss_counter >= self.low_loss_patience:
                print(
                    f"Early stopping: Loss stayed below {self.low_loss_threshold} for {self.low_loss_patience} iterations")
                return True
        else:
            # Reset counter if loss goes above threshold
            self.low_loss_counter = 0

        # Calculate smoothed loss (average of last 'window_size' losses)
        if len(self.loss_history) >= self.window_size:
            smoothed_loss = np.mean(self.loss_history[-self.window_size:])

            # Is this smoothed loss better than our best?
            if smoothed_loss < self.best_loss * (1 - self.rel_tolerance):
                self.best_loss = smoothed_loss
                self.best_iteration = iteration

            # Stop if no improvement for 'patience' iterations
            if iteration - self.best_iteration >= self.patience:
                print(f"Early stopping: No improvement for {self.patience} iterations")
                return True

        return False

class Solver:

    def __init__(self, equation, dim_h_Y, dim_h_Z, lr, weight_decay=1e-5):
        self.equation = equation
        self.dim_h_Y = dim_h_Y
        self.dim_h_Z = dim_h_Z
        self.modelY = NN_Y(equation, dim_h_Y).to(device)
        self.modelZ = NN_Z(equation, dim_h_Z).to(device)

        self.optimizer = torch.optim.AdamW(
            list(self.modelY.parameters()) + list(self.modelZ.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        self.base_lr = lr

    def gen_forward_path(self, batch_size, N, start_n):
        delta_t = self.equation.T / N
        num_steps = N - start_n

        x_paths = torch.zeros(batch_size, num_steps + 1, self.equation.dim_x, device=device)
        w_increments = torch.zeros(batch_size, num_steps, self.equation.dim_d, 1, device=device)

        if start_n == 0:
            x = self.equation.x_0.expand(batch_size, -1).clone()
        else:
            x = self.equation.x_0.expand(batch_size, -1).clone()
            for i in range(start_n):
                w = torch.randn(batch_size, self.equation.dim_d, 1, device=device) * np.sqrt(delta_t)
                t_current = delta_t * i
                drift = self.equation.b(t_current, x) * delta_t
                diffusion = torch.matmul(self.equation.sigma(t_current, x), w).reshape(-1, self.equation.dim_x)
                x = x + drift + diffusion

        x_paths[:, 0, :] = x

        for i in range(num_steps):
            w = torch.randn(batch_size, self.equation.dim_d, 1, device=device) * np.sqrt(delta_t)
            w_increments[:, i, :, :] = w

            t_current = delta_t * (start_n + i)
            drift = self.equation.b(t_current, x) * delta_t
            diffusion = torch.matmul(self.equation.sigma(t_current, x), w).reshape(-1, self.equation.dim_x)
            x = x + drift + diffusion

            x_paths[:, i + 1, :] = x

        return x_paths, w_increments

    def volterra_loss(self, x_paths, w_increments, n, y, z_batch, N, future_models_Y, reflected):
        delta_t = self.equation.T / N
        t_n = delta_t * n
        batch_size = y.shape[0]
        num_steps = N - n

        if n == N:
            # Use x_{N-1}, which is at index 0 in x_paths for this iteration
            x_terminal = x_paths[:, 0, :]  # NOT x_paths[:, -1, :]
            if reflected:
                barrier_val = self.equation.barrier(t_n,
                                                    x_terminal)  # it's ok as the barrier is constant, we only need y for final dimension
                terminal_val = torch.max(self.equation.g(t_n, x_terminal), barrier_val)
            else:
                terminal_val = self.equation.g(t_n, x_terminal)
            # loss = torch.mean((y - terminal_val) ** 2)
            loss = torch.mean((y - terminal_val).norm(2, dim=1))

            return loss, {
                'y_mean': y.mean().item(),
                'y_std': y.std().item()
            }

        # For n < N-1: Backward BSDE equation
        # We need Y values at future timesteps for the f function
        y_batch = torch.zeros(batch_size, num_steps, self.equation.dim_y, device=device)
        y_batch[:, 0, :] = y  # Y_n at the current timestep

        # Fill in future Y values using pretrained models
        if num_steps > 1:
            with torch.no_grad():
                for idx in range(1, num_steps):
                    m = n + idx
                    if m in future_models_Y:
                        x_m = x_paths[:, idx, :]
                        y_batch[:, idx, :] = future_models_Y[m](N, m, x_m)

        # Compute terminal value at time T (which is x_paths[:, -1, :] = x_N)
        x_T = x_paths[:, -1, :]
        if reflected:
            barrier_val = self.equation.barrier(t_n,
                                                x_T)  # it's ok as the barrier is constant, we only need y for final dimension
            terminal_val = torch.max(self.equation.g(t_n, x_T), barrier_val)
        else:
            terminal_val = self.equation.g(t_n, x_T)
        # Vectorized f computation over all future timesteps
        s_array = delta_t * torch.arange(n, N, device=device)
        x_future = x_paths[:, :-1, :]  # [batch, num_steps, dim_x]
        f_vals = self.equation.f_vectorized(t_n, s_array, x_future, y_batch, z_batch)

        # Integral approximations
        integral_f = (f_vals * delta_t).sum(dim=1)  # [batch, dim_y]
        z_dot_w = torch.matmul(z_batch, w_increments).squeeze(-1)  # [batch, num_steps, dim_y]
        integral_z = z_dot_w.sum(dim=1)  # [batch, dim_y]

        # BSDE equation: Y_n = g(T, X_T) + ∫_n^T f(...) ds - ∫_n^T Z dW
        estimate = terminal_val + integral_f - integral_z
        # loss = torch.mean((y - estimate) ** 2)
        loss = torch.mean((y - estimate).norm(2, dim=1))

        return loss, {
            'estimate_mean': estimate.mean().item(),
            'estimate_std': estimate.std().item(),
            'y_mean': y.mean().item(),
            'y_std': y.std().item()
        }

    def train_step(self, batch_size, N, n, itr, multiplier, future_models_Y, reflected, use_scheduler,
                   early_stop_threshold, factor, patience, grad_clip, max_grad_norm, verbose=True):

        '''early_stopper = EarlyStoppingChecker(
            patience=100,
            min_iterations=50,
            rel_tolerance=0.001,
            window_size=10,
            low_loss_threshold=0.001,  # Your threshold
            low_loss_patience=30  # Your patience
        )'''

        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                # self.optimizer, mode='min', factor=0.5, patience=30, threshold=1e-4, min_lr=1e-6
                self.optimizer, mode='min', factor=factor, patience=patience, threshold = 1e-4,  min_lr=1e-6
            )

        history = {
            'loss': [], 'y_mean': [], 'y_std': [], 'z_mean': [], 'z_std': [],
            'gradient_norm': [], 'learning_rate': []
        }

        is_terminal = (n == N)
        step_type = "TERMINAL" if is_terminal else "BACKWARD"

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Training timestep n={n} ({step_type} STEP)")
            print(f"Time: t={self.equation.T * n / N:.3f}, Future models: {len(future_models_Y)}")
            print(f"{'=' * 60}")

        if n == N:
            max_iterations = itr * multiplier
        else:
            max_iterations = itr

        for i in range(max_iterations):
            x_paths, w_increments = self.gen_forward_path(batch_size, N, n)
            x_n = x_paths[:, 0, :]

            y = self.modelY(N, n, x_n)

            if n < N:
                m_indices = torch.arange(n, N, device=device)
                x_future = x_paths[:, :-1, :]
                z_batch = self.modelZ(N, n, x_n, m_indices, x_future)
            else:
                z_batch = torch.zeros(batch_size, N, device=device)

            loss, metrics = self.volterra_loss(x_paths, w_increments, n, y, z_batch, N, future_models_Y, reflected)

            self.optimizer.zero_grad()
            loss.backward()

            total_norm = sum(p.grad.data.norm(2).item() ** 2
                             for p in self.modelY.parameters() if p.grad is not None)
            total_norm += sum(p.grad.data.norm(2).item() ** 2
                              for p in self.modelZ.parameters() if p.grad is not None)
            grad_norm = total_norm ** 0.5

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    list(self.modelY.parameters()) + list(self.modelZ.parameters()),
                    max_norm=max_grad_norm
                )

            self.optimizer.step()

            if use_scheduler:
                scheduler.step(loss.item())

            current_lr = self.optimizer.param_groups[0]['lr']

            history['loss'].append(loss.item())
            history['gradient_norm'].append(grad_norm)
            history['y_mean'].append(metrics.get('y_mean', y.mean().item()))
            history['y_std'].append(metrics.get('y_std', y.std().item()))
            history['z_mean'].append(z_batch.mean().item())
            history['z_std'].append(z_batch.std().item())
            history['learning_rate'].append(current_lr)

            actual_iterations = len(history['loss'])

            wandb_log = {
                f'losses/loss_{n}': loss.item(),
                f'timestep_{n}/gradient_norm': grad_norm,
                f'timestep_{n}/learning_rate': current_lr,
                f'timestep_{n}/y_mean': metrics.get('y_mean', y.mean().item()),
                f'timestep_{n}/y_std': metrics.get('y_std', y.std().item()),
                f'timestep_{n}/z_mean': z_batch.mean().item(),
                f'timestep_{n}/z_std': z_batch.std().item(),
                f'timestep_{n}/iteration': i,
                f'timestep_{n}/actual_iterations': actual_iterations,
                f'timestep_{n}/stopped_early': actual_iterations < max_iterations,
                f'summary/iterations_used_n_{n}': actual_iterations

            }

            if not is_terminal:
                wandb_log.update({
                    f'timestep_{n}/integral_z_mean': metrics.get('integral_z_mean', 0),
                    f'timestep_{n}/estimate_mean': metrics.get('estimate_mean', 0),
                })

            wandb.log(wandb_log)

            if verbose and (i % 100 == 0 or i < 10):
                print(f"Iter {i:4d}: loss={loss.item():.6e} | "
                      f"grad={grad_norm:.4e} | lr={current_lr:.2e}")

            '''
            if n < N:
                if early_stopper.check(i, loss.item()):
                    if verbose:
                        print(f"\n✓ Early stopped at iteration {i}")
                        print(f"  Best loss was at iteration {early_stopper.best_iteration}")
                        print(f"  Final loss: {loss.item():.6e}")
                    break'''

        return history, {'iterations': actual_iterations}


def full_backward_training(example_type, config, equation, save_dir, reflected):
    """Train the ENTIRE BSDE from N backwards to 0"""
    print(f"\n{'#' * 70}")
    print(f"# FULL BACKWARD TRAINING: {example_type}")
    print(f"{'#' * 70}\n")

    os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    future_models_Y = {}
    future_models_Z = {}

    N = config.get('N')
    T = config.get('T')

    iteration_counts = []

    for n in range(N, -1, -1):
        print(f"\n{'=' * 70}")
        print(f"TRAINING TIMESTEP n={n}/{N} (t={T * n / N:.3f})")
        print(f"{'=' * 70}")

        lr = config.get('lr', 1e-2)
        if n < N:
            lr = lr * config.get('lr_decay', 0.8) ** (N - n)

        solver = Solver(
            equation,
            dim_h_Y=config.get('dim_h_Y', 64),
            dim_h_Z=config.get('dim_h_Z', 80),
            lr=lr,
            weight_decay=config.get('weight_decay', 1e-5)
        )

        if n < N:
            solver.modelY.load_state_dict(
                torch.load(os.path.join(save_dir, f'{example_type}_Y_{n + 1}.pt'))
            )
            if n < N - 1:
                solver.modelZ.load_state_dict(
                    torch.load(os.path.join(save_dir, f'{example_type}_Z_{n + 1}.pt'))
                )
            # print(f"✓ Warm start from n={n + 1}")

        history, info = solver.train_step(
            batch_size=config.get('batch_size', 1024),
            N=N,
            n=n,
            itr=config.get('itr', 2000), multiplier=config.get('multiplier', 1),
            future_models_Y=future_models_Y, reflected=reflected,
            use_scheduler=config.get('use_scheduler', True),
            early_stop_threshold=config.get('early_stop_threshold', 1e-4),
            factor = config.get('factor', 0.5),
            patience = config.get('patience', 50),
            grad_clip=config.get('grad_clip', True),
            max_grad_norm=config.get('max_grad_norm', 1.0),
            verbose=True
        )

        torch.save(solver.modelY.state_dict(),
                   os.path.join(save_dir, f'{example_type}_Y_{n}.pt'))
        torch.save(solver.modelZ.state_dict(),
                   os.path.join(save_dir, f'{example_type}_Z_{n}.pt'))

        solver.modelY.eval()
        solver.modelZ.eval()
        future_models_Y[n] = solver.modelY
        future_models_Z[n] = solver.modelZ

        all_results[n] = {
            'history': history,
            'info': info,
            'final_loss': history['loss'][-1]
        }

        iteration_counts.append(all_results[n]['info']['iterations'])

        # --- Save losses and lr to JSON file ---
        loss_data = {
            "timestep": n,
            "loss_per_iteration": history["loss"],
            "learning_rate": history["learning_rate"]
        }

        json_path = os.path.join(save_dir, f"loss_history_timestep_{n}.json")
        with open(json_path, "w") as f:
            json.dump(loss_data, f, indent=4)
        print(f"✓ Saved loss history for timestep {n} to {json_path}")

        wandb.log({
            f'summary/timestep_{n}_final_loss': history['loss'][-1],
            f'summary/timestep_{n}_iterations': info['iterations']
        })

        print(f"\n Final loss: {history['loss'][-1]:.6e}")
        print(f"Iterations used: {info['iterations']}")

        wandb.log({
            'summary/total_iterations': sum(iteration_counts),
            'summary/avg_iterations_per_timestep': np.mean(iteration_counts),
            'summary/max_iterations_used': max(iteration_counts),
            'summary/min_iterations_used': min(iteration_counts),
        })

    return all_results, equation





