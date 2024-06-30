#!/bin/bash
#SBATCH --account=training2426
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus

# 1， 2， 4，  8，  16， 32，  64，  128， 256
# 4， 8， 16， 32， 64， 128， 256， 512， 1024

# Without this, srun does not inherit cpus-per-task from sbatch.
echo "----------------------------------"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

## TODO:
# 1. Set MASTER_ADDR and MASTER_PORT
# 2. Set GPUS_PER_NODE

echo "Job id: $SLURM_JOB_ID"
source env/activate.sh 

## TODO:
# Export TORCH_LOGS for fsdp

## TODO:
# Replace this line with the distributed training launch script
srun --cpu_bind=none python to_distributed_training.py