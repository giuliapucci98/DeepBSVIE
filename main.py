import torch
import os
import time
from datetime import datetime
import json

# =============== SELECT THESE BEFORE STARTING ===========
example_type = "nonlinear"  # select example. Options: "linear1", "linear2", "example1a"
reflected = False

run_name = example_type + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#=============== PATH SETUP ====================
print("Working directory:", os.getcwd())
new_folder_flag = True
new_folder = run_name
project_dir = "/cfs/klemming/projects/supr/naiss2024-22-1707/DeepBSVIE"
if os.path.exists(project_dir): #we are on server
    path = os.path.join(project_dir, new_folder, "models")
    # Set WandB cache and artifacts directories
    os.environ['WANDB_CACHE_DIR'] = os.path.join(project_dir, "wandb_cache")
    os.environ['WANDB_ARTIFACTS_DIR'] = os.path.join(project_dir, "wandb_artifacts")
    # Make sure directories exist
    os.makedirs(os.environ['WANDB_CACHE_DIR'], exist_ok=True)
    os.makedirs(os.environ['WANDB_ARTIFACTS_DIR'], exist_ok=True)
else:
    project_dir= os.getcwd()
    path = os.path.join(new_folder, "models")
if new_folder_flag:
    os.makedirs(path, exist_ok=True)
print("State dicts will be saved in:", path)


import wandb

from BSVIE import volterra_fbsde, Solver, full_backward_training
from Evaluation import validate_against_analytical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# Set parameters depending on the example type
if example_type == "linear1":
    mu_base, sig_base, x0_scale = 0.0, 1.0, 0.0
elif example_type == "linear2":
    mu_base, sig_base, x0_scale = 0.1, 0.5, 1.0
elif example_type == "nonlinear":
    mu_base, sig_base, x0_scale = 0.1, 0.3, 1.0
elif example_type == "reflected":
    mu_base, sig_base, x0_scale = 0.1, 0.2, 1.0
else:
    raise ValueError(f"Unknown example_type: {example_type}")

config = {
    'dim_x': 5,
    'N': 50,
    'T': 1.0,
    'mu_base': mu_base,
    'sig_base': sig_base,
    'x0_scale': x0_scale,
    'dim_h_Y': 40,
    'dim_h_Z': 80,
    'batch_size': 2**12,
    'itr': 500,
    'multiplier': 3,
    'lr': 1e-2,
    'lr_decay': 0.995,
    'factor_lr_decay': 0.5,
    'patience_lr_decay': 50,
    'weight_decay': 1e-5,
    'use_scheduler': True,
    'early_stop_threshold': 1e-5,
    'seed': 42,
    'example_type': example_type,
    'grad_clip': True,
    'max_grad_norm': 1.0,
}

config_path = os.path.join(project_dir, new_folder, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)
print(f"✅ Config saved to: {config_path}")


wandb.init(
    project="bsde-volterra-solver",
    name=run_name,
    config=config,
    dir=path,
    tags=[example_type, f"N={config['N']}", "GPU"],
    reinit=True
)

print(f"\n\n{'#' * 70}")
print(f"# STARTING FULL TRAINING FOR {example_type}")
print(f"# RUN NAME: {run_name}")
print(f"{'#' * 70}")

dim_x = config.get('dim_x')
dim_y = 1
dim_d = dim_x
N = config.get('N')
T = config.get('T')
x_0 = torch.ones(dim_x).to(device) * config.get('x0_scale', 1.0)
equation = volterra_fbsde(
    x_0=x_0,
    mu_base=config.get('mu_base', 0.1),
    sig_base=config.get('sig_base', 0.3),
    lam=0.5, lam0=0.5, T=T,
    dim_x=dim_x, dim_y=dim_y, dim_d=dim_d,
    example_type=example_type,
    seed=config.get('seed', 42),
)
start_time = time.time()

all_results, equation = full_backward_training(
    example_type=example_type,
    config=config,
    equation=equation,
    save_dir=path, reflected=reflected
)
end_time = time.time()
elapsed_time = (end_time - start_time) / 60  # convert seconds to minutes
print(f"Elapsed time: {elapsed_time:.4f} minutes")

wandb.log({"training/total_time_minutes": elapsed_time})

future_models_Y = {}
future_models_Z = {}
for n in sorted(all_results.keys()):
    solver = Solver(
        equation,
        config['dim_h_Y'],
        config['dim_h_Z'],
        config['lr'],
        config['multiplier']
    )
    model_y_path = os.path.join(path, f"{example_type}_Y_{n}.pt")
    model_z_path = os.path.join(path, f"{example_type}_Z_{n}.pt")
    solver.modelY.load_state_dict(torch.load(model_y_path))
    solver.modelZ.load_state_dict(torch.load(model_z_path))
    solver.modelY.eval()
    solver.modelZ.eval()
    future_models_Y[n] = solver.modelY
    future_models_Z[n] = solver.modelZ

validation_metrics = validate_against_analytical(
    equation, example_type,
    future_models_Y, future_models_Z, config['N'], path
)

wandb.finish()
