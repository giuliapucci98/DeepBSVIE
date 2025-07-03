import torch
import numpy as np
import os
import time
import json


from BSVIE import volterra_fbsde
from BSVIE import BSDEiter
from BSVIE import NN_Y
from BSVIE import NN_Z


if torch.cuda.is_available() and torch.version.hip is not None:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


#check the working directory
print("Working directory:", os.getcwd())

new_folder_flag = True
new_folder = "barrier/"

path = os.path.join(new_folder, "state_dicts")

if new_folder_flag:
    os.makedirs(path, exist_ok=True)


#network parameters
network_params = {"dim_x": 1,"dim_y": 1,"dim_d": 1,"dim_h_Y": 11, "dim_h_Z": 11,"N": 50,"itr": 500,"batch_size": 2**13,"multiplier": 1}
dim_x, dim_y, dim_d, dim_h_Y, dim_h_Z, N, itr, batch_size, multiplier =  network_params.values()

#BSVIE parameters
BSVIE_params = {"T": 1.0, "x0_value": 1.0, "mu": 0.07, "sig": 0.1, "lam":0.05, "lam0":0.5, "K": 1.0}
T, x0_value, mu, sig, lam, lam0, K = BSVIE_params.values()

dt=T/N
x_0 = torch.ones(dim_x, device = device)*x0_value

all_params = {**network_params, **BSVIE_params}

with open(os.path.join(path, "parameters.json"), "w") as f:
    json.dump(all_params, f, indent=2)

equation = volterra_fbsde(x_0, mu, sig, lam, lam0, K, T, dim_x, dim_y, dim_d)

bsde_itr = BSDEiter(equation, dim_h_Y, dim_h_Z)


start_time = time.time()
loss, y = bsde_itr.train_whole(batch_size, N, path, itr, multiplier)
end_time = time.time()
elapsed_time = (end_time - start_time) / 60  # convert seconds to minutes
print(f"Elapsed time: {elapsed_time:.4f} minutes")

with open(os.path.join(path, "loss.json"), 'w') as f:
    json.dump(loss, f, indent=2)