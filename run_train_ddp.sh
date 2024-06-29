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
# so processes know who to talk to
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
export MASTER_PORT=7010
export GPUS_PER_NODE=4

echo "MASTER_ADDR:MASTER_PORT=""$MASTER_ADDR":"$MASTER_PORT"
echo "----------------------------------"


echo "Job id: $SLURM_JOB_ID"
source env/activate.sh 


srun --cpu_bind=none bash -c "torchrun \
    --nnodes=$SLURM_NNODES \
    --rdzv_backend c10d \
    --nproc_per_node=gpu \
    --rdzv_id $RANDOM \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
    train_ddp.py "