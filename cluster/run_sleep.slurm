#!/bin/bash
#SBATCH -p docencia                  # UPV DSIC queue for jobs < 24 hours
#SBATCH --gres=shard:2               # Request 2 logical GPUs (~16GB VRAM on NVIDIA L40S 48GB)
#SBATCH --cpus-per-task=4            # 4 CPU threads for data prep and GGUF quantization
#SBATCH --time=01:00:00              # 1 hour walltime (typical cycle takes ~8 minutes)
#SBATCH --job-name=continual_sleep
#SBATCH -o logs/sleep_%j.log

# ==============================================================================
# Continual Sleep Engine — SLURM Nocturnal Consolidation Job
# Target Host: tensor.dsic.upv.es (4x NVIDIA L40S 48GB GDDR6 ECC, CUDA 12.6)
# ==============================================================================
set -euo pipefail

echo "============================================================"
echo "[SLURM] Starting Continual Sleep Cycle on $(hostname)"
echo "[SLURM] Job ID: ${SLURM_JOB_ID:-local} | Date: $(date -u)"
echo "============================================================"

# 1. Verify disk quota safety (excluding mounted network shares W and asigDSIC)
echo "[Check] Current $HOME usage:"
du -sh --exclude=asigDSIC --exclude=W "$HOME" || true

# 2. Activate isolated Python virtual environment (CUDA 12.6 / cu126)
VENV_PATH="${VENV_PATH:-$HOME/venv/sleep_env}"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
fi

# 3. Inspect allocated GPU shard
nvidia-smi

# 4. Run full nocturnal pipeline (Rehearsal -> Mixer -> LoRA Merge -> Safety Gate)
CYCLE_ID="cycle_$(date -u +%Y%m%d_%H%M%S)"
python -m cluster.run_pipeline --cycle-id "$CYCLE_ID" --mode hpc_gpu

# 5. If Safety Gate approved the checkpoint, compile to Q4_K_M GGUF for Edge RTX 3060
if [[ -f "data/artifacts/audit_${CYCLE_ID}_APPROVED.flag" ]]; then
    echo "[SLURM] Safety Quality Gate PASSED. Compiling merged FP16 checkpoint to GGUF Q4_K_M..."
    bash cluster/quantize_gguf.sh "data/merged_models/checkpoint_${CYCLE_ID}" "data/artifacts/latest_model.gguf"
else
    echo "[SLURM] Safety Quality Gate REJECTED candidate checkpoint. Aborting GGUF compilation (Rollback active)."
    exit 2
fi

echo "[SLURM] Nocturnal Sleep Cycle $CYCLE_ID finished successfully."
