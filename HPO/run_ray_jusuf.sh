#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=ray_hpo_nxtaim
#SBATCH --account=training2426
#SBATCH --output=ray_hpo_RAND.out
#SBATCH --error=ray_hpo_RAND.err
#SBATCH --partition=gpus
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --exclusive

# load relevant modules
ml --force purge
ml Stages/2024 GCCcore/.12.3.0 Python/3.11.3 

ml NCCL/default-CUDA-12 PyTorch/2.1.2 torchvision/0.16.2 JUBE/2.6.1

# activate local python environment
source ray_tune_env/bin/activate

# make sure the nodes use infiniband for communication
export NCCL_SOCKET_IFNAME="ib0"

# python script and arguments
COMMAND="tune_cifar.py --scheduler RAND --num-samples 10  --max-iterations 10 --num-workers 1 --data-dir /p/scratch/training2426/cifar10/data "

echo $COMMAND
echo "NUM NODES: ${SLURM_JOB_NUM_NODES}"

# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES="0"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

num_gpus=1

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1

####### this part is taken from the ray example slurm script #####
set -x

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# get the head node
head_node=${nodes_array[0]}

port=8674

export ip_head="$head_node"i:"$port"
export head_node_ip="$head_node"i

# launch the head node
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &

worker_num=$((SLURM_JOB_NUM_NODES - 1))

# launch the worker nodes
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" --node-ip-address="$node_i"i --redis-password='5241590000000000' \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
done

echo "Ready"

# run python script
python -u $COMMAND
