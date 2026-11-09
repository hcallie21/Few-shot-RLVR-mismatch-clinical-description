#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=90G
#SBATCH --cpus-per-task=12
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <saved_hf_token_name> <image_root>"
    exit 1
fi
echo "huggingface authorization token name: $1"
echo "image root: $2"
set -eo pipefail
# Set thread configuration
cd "$SLURM_SUBMIT_DIR"
conda activate trl-flash
pip cache purge
module load CUDA/12.8
module load cuDNN/9.12.0/CUDA-12
module load gcc/13.2.0
module load cmake/3.30.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME=$CUDA_HOME"
LIGER_KERNEL_IMPL=cutedsl
export OMP_NUM_THREADS=1  # Prevent system overload
export CUDA_LAUNCH_BLOCKING=1

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export GPU_ID=${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}
echo ${GPU_ID}


echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

nvidia-smi -L

hf auth list
hf auth switch --token-name "$1"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} torchrun --standalone --nproc_per_node=2 \
    m27b_all_lin_grpo_reinforcement_CRO_1500.py \
      --image_root "$2" \
    > ./training_output.txt 2>&1
