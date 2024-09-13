#!/bin/bash
#SBATCH --account=training2435
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=dc-gpu

#SBATCH --reservation=training2435

# Without this, srun does not inherit cpus-per-task from sbatch.
echo "----------------------------------"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

echo "Job id: $SLURM_JOB_ID"
source env/activate.sh 

srun --cpu_bind=none python train.py