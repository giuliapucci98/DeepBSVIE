#!/bin/bash -l
#SBATCH -A naiss2025-22-794
#SBATCH -J myjob
#SBATCH -p main
#SBATCH -t 00:10:00
#SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
#SBATCH --output=output.log
#SBATCH --error=error.log

module load python/3.10

source /pdc/software/23.12/eb/software/anaconda3/2024.02-1-cpeGNU-23.12/etc/profile.d/conda.sh
conda activate myenv

ml PDCOLD/23.12
ml  singularity/4.1.1-cpeGNU-23.12

export PYTHONPATH=/cfs/klemming/home/p/pucci/.conda/envs/myenv/lib/python3.10/site-packages:$PYTHONPATH

singularity exec --rocm -B /cfs/klemming /pdc/software/resources/sing_hub/rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1 python3 test.py

