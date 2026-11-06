#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=90G
#SBATCH --cpus-per-task=12
set -eo pipefail
cd /data/houstoncm/Contradictions/exhaustive_hp_tuning/try_best_combos/80_10_10_CRO_OLLAMA_BETA_01_m27b
source $MYCONDA
conda activate trl-flash
pip cache purge
module load CUDA/12.8
module load cuDNN/9.12.0/CUDA-12
module load gcc/13.2.0
module load cmake/3.30.2
export PATH=$CUDA_HOME/bin:$PATH

# module load CUDA/12.1 gcc
echo "CUDA_HOME=$CUDA_HOME"
LIGER_KERNEL_IMPL=cutedsl

# Three workers, each with a complete BF16 model on its own 80 GB GPU.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset CUDA_LAUNCH_BLOCKING
unset DEBUG_MODE
export LD_LIBRARY_PATH="/data/houstoncm/conda/envs/trl-flash/lib:${LD_LIBRARY_PATH:-}"

# Retain Slurm's CUDA_VISIBLE_DEVICES mapping; GPU_ID does not select GPUs.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total --format=csv
hf auth switch --token-name callie_read

TASK_DIR=/data/houstoncm/Contradictions/exhaustive_hp_tuning/try_best_combos/80_10_10_CRO_OLLAMA_BETA_01_m27b
TASK_PY="${TASK_DIR}/m27b_all_lin_grpo_inference_CRO_1500_mult_gpu.py"
TASK_RUN_DIR=$(mktemp -d "${TASK_DIR}/inference__XXXXXX")
TASK_LOG="${TASK_RUN_DIR}/inference.log"
TASK_BATCH_SIZE=${INFERENCE_BATCH_SIZE:-2}
echo "Progress log: ${TASK_LOG}"
echo "Worker CSVs: ${TASK_RUN_DIR}"

# torchrun starts the workers directly; no DDP wrapping or gradient sync is needed.
# Explicit world size prevents accidentally processing the whole dataset per GPU.
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    "${TASK_PY}" --workers 3 --run-dir "${TASK_RUN_DIR}" \
    --batch-size "${TASK_BATCH_SIZE}" --threads 2 --max-new-tokens 1500 \
    > "${TASK_LOG}" 2>&1

# Runs only after all workers succeed; validates coverage and restores input order.
python "${TASK_PY}" --merge-results --workers 3 --run-dir "${TASK_RUN_DIR}" \
    >> "${TASK_LOG}" 2>&1
echo "Finished. Final CSV: ${TASK_DIR}/m27b_all_lin_grpo_inference_reasoning_cos_repetition_overlong_1500.csv"
