#!/bin/bash -l
#SBATCH -A naiss2025-22-794
#SBATCH -J myjob
#SBATCH -p gpu
#SBATCH -t 00:30:00
#SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
#SBATCH --output=output.log
#SBATCH --error=error.log


source /pdc/software/23.12/eb/software/anaconda3/2024.02-1-cpeGNU-23.12/etc/profile.d/conda.sh
conda activate myenv

ml PDCOLD/23.12
ml  singularity/4.1.1-cpeGNU-23.12



singularity exec --rocm -B /cfs/klemming /pdc/software/resources/sing_hub/rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1 python3 train.py



