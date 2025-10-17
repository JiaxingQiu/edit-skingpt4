#!/bin/bash
#SBATCH --job-name=ft_y3_skingpt_io
#SBATCH -p gpu
#SBATCH --gres=gpu:1  
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH -t 2:00:00
#SBATCH -A cama
#SBATCH --export=NONE
#SBATCH --output=log/slurm-%x.out
#SBATCH --error=log/slurm-%x.err

set -euo pipefail

module purge
module load cuda/11.8.0        
source "$HOME/miniconda3/etc/profile.d/conda.sh"   
conda activate skingpt4

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd /scratch/jq2uw/edit-skingpt4
python3 main_ft.py --model_type gpt_io --eval_target y3