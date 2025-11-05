#!/bin/bash
#SBATCH --job-name=rbm_opt_gpu
#SBATCH --output=logs/rbm_opt_%A_%a.out
#SBATCH --error=logs/rbm_opt_%A_%a.err
# SBATCH --array=1-$(wc -l < H_idx_list.txt)
#SBATCH --time=0:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod    
#SBATCH --qos=boost_qos_dbg
#SBATCH --mem=8G
#SBATCH --account=ict_25_cmsp

# Load any required modules here
module load cuda
module load python
module load openmpi

source ../.venv/bin/activate

export $OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Create logs directory if it doesn't exist
mkdir -p logs

# Get the H_idx from the file based on array task ID
# H_IDX=$(sed -n "${SLURM_ARRAY_TASK_ID}p" H_idx_list.txt)

# Set environment variables for GPU usage
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Go to the Netket directory
cd ../Netket

# Run the Python script with the index
# python run_coup_optimal_basis.py $H_IDX
python run_coup_optimal_basis.py 10