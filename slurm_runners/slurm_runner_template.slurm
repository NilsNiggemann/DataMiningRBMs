#!/bin/bash
#SBATCH --job-name=rbm_opt_gpu
#SBATCH --output=logs/rbm_opt_%A_%a.out
#SBATCH --error=logs/rbm_opt_%A_%a.err
#SBATCH --array=1-470%2
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=boost_usr_prod    
#SBATCH --qos=boost_qos_dbg
# SBATCH --qos=boost_qos_lprod
#SBATCH --mem=10G
#SBATCH --account=ICT25_CMSP_0

# Load any required modules here
module load cuda/12.3
module load python
module load openmpi
unset LD_LIBRARY_PATH
source ../.venv/bin/activate

echo "Module Load steps compelete"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
# Create logs directory if it doesn't exist
mkdir -p logs

# echo "testing cuda from python"
# python -c 'import jax; print(jax.devices())'
# echo "python step should have printed jax.devices"

# Get the H_idx from the file based on array task ID
H_IDX=$(sed "${SLURM_ARRAY_TASK_ID}q;d" H_idx_list.txt)


# Go to the Netket directory
cd ../Netket

# Run the Python script with the index
# python run_coup_unrotated_basis.py $H_IDX
python run_coup_optimal_basis.py $H_IDX --output_folder ../data/extended_data_optimal_basis_rbm
# python run_coup_optimal_basis.py 10
#
echo "DONE, exiting from slurm........"
