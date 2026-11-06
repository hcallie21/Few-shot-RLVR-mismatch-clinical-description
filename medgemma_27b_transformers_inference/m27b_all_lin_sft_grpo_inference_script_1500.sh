#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=90G
#SBATCH --cpus-per-task=12
set -eo pipefail

cd /data/houstoncm/Contradictions/
source $MYCONDA
conda activate trl
# module load CUDA/12.1 gcc
echo "CUDA_HOME=$CUDA_HOME"

# Three workers, each with a complete BF16 model on its own 80 GB GPU.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset CUDA_LAUNCH_BLOCKING
unset DEBUG_MODE
export LD_LIBRARY_PATH="/data/houstoncm/conda/envs/trl/lib:${LD_LIBRARY_PATH:-}"

# Retain Slurm's CUDA_VISIBLE_DEVICES mapping; GPU_ID does not select GPUs.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=name,memory.total --format=csv
hf auth switch --token-name callie_read

TASK_DIR=/data/houstoncm/Contradictions/exhaustive_hp_tuning/control_inf_transformers
TASK_PY="${TASK_DIR}/m27b_all_lin_inference_1500_mult_gpu_base_or_adapter.py"
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
echo "Finished. Final CSV: ${TASK_DIR}/m27b_control_inf_transformers_max_tokens_1500.csv"
