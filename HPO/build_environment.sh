ml --force purge
ml Stages/2024 GCCcore/.12.3.0 Python/3.11.3 

ml NCCL/default-CUDA-12 PyTorch/2.1.2 torchvision/0.16.2

python3 -m venv ray_tune_env

source ray_tune_env/bin/activate

pip3 install ray==2.31.0 ray[tune]==2.31.0

deactivate