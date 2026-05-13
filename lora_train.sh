#!/bin/bash
#SBATCH --job-name=gs_hyperspectral
#SBATCH --account=gs_hyperspectral
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.log

#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --mem=20G
#SBATCH --time=03:00:00

# Activate conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate sam3ju

# Move to project directory
cd /home/hensemberk/dev/ia_project

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run training
python train_finetune.py 

echo "End time: $(date)"