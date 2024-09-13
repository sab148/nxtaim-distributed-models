#!/bin/bash
#SBATCH --account=training2435
#SBATCH --nodes=1   ## TODO: Increase this number when you use DDP or FSDP
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dc-gpu

#SBATCH --reservation=training2435

# Without this, srun does not inherit cpus-per-task from sbatch.
echo "----------------------------------"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

## TODO:
# 1. Set MASTER_ADDR and MASTER_PORT

echo "Job id: $SLURM_JOB_ID"
source env/activate.sh 

## TODO:
# Export TORCH_LOGS for FSDP

## TODO:
# Replace this line with the distributed training launch script
srun --cpu_bind=none python to_distributed_training.py