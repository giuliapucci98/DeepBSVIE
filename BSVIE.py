import setuptools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from collections import deque
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class volterra_fbsde():
    def __init__(self, x_0, mu, sig, lam, lam0,K, T, dim_x, dim_y, dim_d):
        self.x_0 = x_0
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d
        self.mu = mu
        self.sig= sig
        self.lam = lam
        self.lam0 = lam0
        self.K = K

    def b(self, t, x):
        #return mu * torch.ones_like(x) # ABM
        return self.mu*x #GBM
        #return torch.zeros_like(x)  # Simple Brownian Motion

    def sigma(self, t, x):
        #return sig * torch.ones_like(x).unsqueeze(-1) #ABM
        return self.sig * x.view(-1, 1, 1)  # GBM
        #return torch.ones_like(x).unsqueeze(-1)  # brownian motion

    def f(self, t, s, x, y, z):
        #t = torch.as_tensor(t, dtype=y.dtype, device=y.device)
        #s = torch.as_tensor(s, dtype=y.dtype, device=y.device)
        #indicator = (s >= t).to(y.dtype)
        ##return torch.exp(-(s - t)) * indicator * y + z.squeeze(-1) #LINEAR EXAMPLE 1
        #return torch.exp(-(s - t)) * indicator * y + z.squeeze(-1)*torch.exp(s) #LINEAR EXAMPLE 1.1

        #return 0.5*x #linear example 2

        #EXAMPLE WITH BARRIERS
        return torch.zeros_like(x)


    def g(self, t, x):
        #return x*np.sin(np.pi * t) #LINEAR EXAMPLE 1
        #return np.exp(-self.lam* t) * x #LINEAR EXAMPLE 2

        #EXAMPLE WITH BARRIERS
        discount = 1/(1+(self.T-t))
        return discount*torch.relu(x-self.K)


    def barrier(self, t, x):
        #discount = 1 / (1 + (self.T - t))
        #return torch.relu(x-self.K)
        return 0.1*torch.ones_like(x) #to be fixed for multi dimension


class NN_Y(nn.Module):
    def __init__(self, equation, dim_h):
        super(NN_Y, self).__init__()
        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        #self.linear4 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y)
        self.bn1 = nn.BatchNorm1d(dim_h)
        self.bn2 = nn.BatchNorm1d(dim_h)
        self.bn3 = nn.BatchNorm1d(dim_h)
        self.equation = equation

    def forward(self, N, n, x):

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/(sd + 0.0001)

        def phi(x):
            x = torch.tanh(self.linear1(x))
            x = torch.tanh(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return self.linear4(x) #[bs,dy] -> [bs,dy]

        delta_t = self.equation.T / N
        x_nor = standardize(x)
        #if n!=0:
        #    x_nor = standardize(x)

        inpt = torch.cat((x_nor, torch.ones(x.size()[0], 1, device=device) * delta_t * n), 1)
        y = phi(inpt)
        return y



class NN_Z(nn.Module):
    def __init__(self, equation, dim_h):
        super(NN_Z, self).__init__()
        self.linear1 = nn.Linear( 2*equation.dim_x + 2 , dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        #self.linear4 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y * equation.dim_d)
        self.bn1 = nn.BatchNorm1d(dim_h)
        self.bn2 = nn.BatchNorm1d(dim_h)
        self.bn3 = nn.BatchNorm1d(dim_h)
        self.equation = equation

    def forward(self, N, n, xt, m, xs):
        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/(sd + 0.0001)
        def phi(x):
            x = torch.tanh(self.linear1(x))
            x = torch.tanh(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return self.linear4(x) #[bs,dy*dd] -> [bs,dy*dd]

        delta_t = self.equation.T / N
        xt_nor = standardize(xt)
        xs_nor = standardize(xs)

        inpt = torch.cat((xt_nor, torch.ones(xt.size()[0], 1, device=device) * delta_t * n, xs_nor, torch.ones(xt.size()[0], 1, device=device) * delta_t * m  ), 1)
        z = phi(inpt).reshape(-1,self.equation.dim_y,self.equation.dim_d)
        return z


class BSDEsolver():
    def __init__(self, equation, dim_h_Y, dim_h_Z, modelY, modelZ, lr,coeff):
        self.modelY = modelY
        self.modelZ = modelZ
        self.equation = equation
        self.optimizer = torch.optim.Adam(list(self.modelY.parameters()) + list(self.modelZ.parameters()),
                                          lr * coeff)
        self.dim_h_Y = dim_h_Y
        self.dim_h_Z = dim_h_Z


    def volterra_loss(self, x_path, n, y, z_dict, w_dict, N, future_models_Y):
        delta_t = self.equation.T / N
        t_n = delta_t * n

        # Terminal condition: g(t_n, X_T)
        x_T = x_path[N]  # X(T)
        #terminal_val = torch.max(self.equation.g(t_n, x_T), self.equation.barrier(t_n,x_T)) # g(t, X_T)
        terminal_val = torch.max(self.equation.g(t_n, x_T), self.equation.barrier(t_n,x_T)) # g(t, X_T)


        integral_f = torch.zeros_like(y)
        integral_z = torch.zeros_like(y)

        for m in range(n,  N):
            s= delta_t*m
            x_s = x_path[m]

            # Get y(s) for all s >= t_n
            if m == n:
                # At current time step, use the predicted y
                y_s = y
            else:
                # For future time steps use trained models
                if m in future_models_Y:
                    y_s = torch.max(future_models_Y[m](N, m, x_s), self.equation.barrier(s,x_s))

            # Z(t, s)
            z_ts = z_dict[m]
            f_val = self.equation.f(t_n, s, x_s, y_s, z_ts)
            integral_f += f_val * delta_t

            w_s = w_dict[m]
            integral_z += torch.matmul(z_ts, w_s).reshape(-1, self.equation.dim_y)

        estimate = terminal_val + integral_f - integral_z

        dist = (y - estimate).norm(2, dim=1)
        return torch.mean(dist)


    def gen_forward_path(self, batch_size, N, start_n):

        delta_t = self.equation.T / N
        x_paths = {}
        w_dict = {}

        if start_n == 0:
            x = self.equation.x_0 + torch.zeros(batch_size, self.equation.dim_x, device=device, requires_grad=True)
        else:
            x = self.equation.x_0 + torch.zeros(batch_size, self.equation.dim_x, device=device, requires_grad=True)
            for i in range(start_n):
                w = torch.randn(batch_size, self.equation.dim_d, 1, device=device) * np.sqrt(delta_t)
                x = x + self.equation.b(delta_t * i, x) * delta_t + \
                    torch.matmul(self.equation.sigma(delta_t * i, x), w).reshape(-1, self.equation.dim_x)

        x_paths[start_n] = x

        for i in range(start_n, N):
            w = torch.randn(batch_size, self.equation.dim_d, 1, device=device) * np.sqrt(delta_t)
            w_dict[i] = w

            x_next = x + self.equation.b(delta_t * i, x) * delta_t + \
                     torch.matmul(self.equation.sigma(delta_t * i, x), w).reshape(-1, self.equation.dim_x)

            x_paths[i + 1] = x_next
            x = x_next

        return x_paths, w_dict


    def train(self, batch_size, N, n, itr, path, multiplyer):
            loss_n = []
            delta_t = self.equation.T / N

            future_models_Y = {}
            future_models_Z = {}

            if n != N - 1:
                for future_n in range(n + 1, N ):
                    mod_Y = NN_Y(self.equation, self.dim_h_Y).to(device)
                    mod_Y.load_state_dict(torch.load(os.path.join(path, f"Y_state_dict_{future_n}")))
                    mod_Y.eval()
                    future_models_Y[future_n] = mod_Y

                    mod_Z = NN_Z(self.equation, self.dim_h_Z).to(device)
                    mod_Z.load_state_dict(torch.load(os.path.join(path, f"Z_state_dict_{future_n}")))
                    mod_Z.eval()
                    future_models_Z[future_n] = mod_Z

            if n >= N - 2:
                itr_actual = multiplyer * itr
            else:
                itr_actual = itr

            for i in range(itr_actual):
                flag = True
                while flag:
                    try:
                        x_paths, w_dict = self.gen_forward_path(batch_size, N, n)
                        flag = any(torch.isnan(x_paths[key]).any() for key in x_paths)
                    except:
                        flag = True

                x_n = x_paths[n]
                y = self.modelY(N, n, x_n)

                # Compute Z(t_n, s) for all s >= t_n
                z_dict = {}
                for m in range(n, N):
                    x_m = x_paths[m]
                    z_dict[m] = self.modelZ(N, n, x_n, m, x_m)

                loss = self.volterra_loss(x_paths, n, y, z_dict, w_dict, N, future_models_Y)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                loss_n.append(float(loss))

            return loss_n, y

class BSDEiter():
    def __init__(self, equation, dim_h_Y, dim_h_Z):
        self.equation = equation
        self.dim_h_Y = dim_h_Y
        self.dim_h_Z = dim_h_Z


    def train_whole(self, batch_size, N, path, itr, multiplier):
        loss_data = []


        for n in range(N-1,-1,-1):
            print(f"Training for n={n}")
            lr = 0.001
            coeff = 1
            if n < N-2:
                #lr = 0.0005
                coeff = 1
            #print("time "+ str(n))
            modY = NN_Y(self.equation, self.dim_h_Y).to(device)
            modZ = NN_Z(self.equation, self.dim_h_Z).to(device)
            bsde_solver = BSDEsolver(self.equation, self.dim_h_Y, self.dim_h_Z, modY, modZ, lr, coeff)
            if n != N - 1:
                bsde_solver.modelY.load_state_dict(torch.load(os.path.join(path, f"Y_state_dict_{n + 1}")))
                bsde_solver.modelZ.load_state_dict(torch.load(os.path.join(path, f"Z_state_dict_{n + 1}")))
                bsde_solver.optimizer.load_state_dict(torch.load(os.path.join(path, f"state_dict_opt_{n + 1}")))

            loss_n, y = bsde_solver.train(batch_size, N, n, itr, path, multiplier)
            loss_data.append(loss_n)
            torch.save(bsde_solver.modelY.state_dict(), os.path.join(path,"Y_state_dict_" + str(n)))
            torch.save(bsde_solver.modelZ.state_dict(), os.path.join(path, "Z_state_dict_" + str(n)))
            torch.save(bsde_solver.optimizer.state_dict(), os.path.join(path, "state_dict_opt_" + str(n)))

        return loss_data, y














class Result():
    def __init__(self, modelY, modelZ, equation):
        self.modelY = modelY
        self.modelZ = modelZ
        self.equation = equation

    def gen_b_motion(self, batch_size, N):
        delta_t = self.equation.T / N
        W = torch.randn(batch_size, self.equation.dim_d, N, device=device) * np.sqrt(delta_t)

        return W



    def gen_x(self, batch_size, N, W):
        delta_t = self.equation.T / N
        x = self.equation.x_0 + torch.zeros(batch_size, (N+1) * self.equation.dim_x, device=device).reshape(-1,self.equation.dim_x, N+1) #[bs,dx,N]
        for i in range(N):
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1)
            x[:,:,i+1] = x[:,:,i] + self.equation.b(delta_t * i, x[:,:,i]) * delta_t + torch.matmul(self.equation.sigma(delta_t * i, x[:,:,i]),w).reshape(-1, self.equation.dim_x)
        #return torch.exp(x)
        return x

    def predict(self, N, batch_size, x, path):
        delta_t = self.equation.T / N
        ys = torch.zeros(batch_size, self.equation.dim_y, N+1)
        zs = torch.zeros(batch_size, self.equation.dim_y, self.equation.dim_d, N, N)

        for n in range(N):
            self.modelY.load_state_dict(torch.load(os.path.join(path, f"Y_state_dict_{n}"),  map_location=torch.device('cpu')), strict=False)
            self.modelZ.load_state_dict(torch.load(os.path.join(path, f"Z_state_dict_{n}" ),  map_location=torch.device('cpu')), strict=False)

            y = torch.max(self.modelY(N, n, x[:, :, n]) , self.equation.barrier(delta_t*n, x[:, :, n] )) # [bs, dy]
            ys[:, :, n] = y

            for s_idx in range(n, N-1):
                z = self.modelZ(N, n, x[:, :, n], s_idx, x[:, :, s_idx])  # [bs, dy, dd]
                zs[:, :, :, n, s_idx] = z

        # Terminal condition for Y
        ys[:, :, N] = torch.max(self.equation.g(delta_t * N, x[:, :, N]), self.equation.barrier(delta_t*N, x[:, :, N] )) # [
        return ys, zs




