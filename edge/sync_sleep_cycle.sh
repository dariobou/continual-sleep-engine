#!/usr/bin/env bash
# ==============================================================================
# Continual Sleep Engine — Edge-to-HPC Synchronization Script
# Connects local WSL2 (RTX 3060) with UPV DSIC Cluster (tensor.dsic.upv.es)
# Requires SSH alias 'tensor-upv' configured in ~/.ssh/config
# ==============================================================================
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-tensor-upv}"
REMOTE_DIR="${REMOTE_DIR:-~/continual-sleep-engine}"
LOCAL_BUFFER="data/wake_buffer.jsonl"
LOCAL_ARTIFACTS="data/artifacts"

usage() {
    echo "Usage: $0 {push-wake|trigger-sleep|pull-gguf|check-quota}"
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

ACTION="$1"

case "$ACTION" in
    check-quota)
        echo "[Edge -> UPV] Checking disk usage on tensor.dsic.upv.es (excluding mounted W and asigDSIC)..."
        ssh "$REMOTE_HOST" 'du -sh --exclude=asigDSIC --exclude=W $HOME && sinfo'
        ;;

    push-wake)
        if [[ ! -f "$LOCAL_BUFFER" ]]; then
            echo "[Error] No wake buffer found at $LOCAL_BUFFER"
            exit 1
        fi
        echo "[Edge -> UPV] Syncing wake_buffer.jsonl and codebase to $REMOTE_HOST:$REMOTE_DIR..."
        rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*.gguf' \
            ./ "$REMOTE_HOST:$REMOTE_DIR/"
        echo "[Success] Wake buffer uploaded."
        ;;

    trigger-sleep)
        echo "[Edge -> UPV] Submitting SLURM sleep job to 'docencia' partition (--gres=shard:2)..."
        ssh "$REMOTE_HOST" "cd $REMOTE_DIR && mkdir -p logs && sbatch cluster/run_sleep.slurm"
        ssh "$REMOTE_HOST" 'squeue -u $USER'
        ;;

    pull-gguf)
        mkdir -p "$LOCAL_ARTIFACTS"
        echo "[UPV -> Edge] Downloading consolidated GGUF and Audit Report..."
        rsync -avz "$REMOTE_HOST:$REMOTE_DIR/data/artifacts/" "$LOCAL_ARTIFACTS/"
        if [[ -f "$LOCAL_ARTIFACTS/latest_model.gguf.sha256" ]]; then
            echo "[Edge] Verifying SHA256 integrity of downloaded GGUF..."
            (cd "$LOCAL_ARTIFACTS" && sha256sum -c latest_model.gguf.sha256)
            echo "[Edge] Registering updated model in local Ollama server..."
            cat <<EOF > "$LOCAL_ARTIFACTS/Modelfile"
FROM ./latest_model.gguf
PARAMETER temperature 0.3
SYSTEM "You are a reliable, safety-aligned personal engineering assistant."
EOF
            (cd "$LOCAL_ARTIFACTS" && ollama create sleep-assistant:latest -f Modelfile)
            echo "[Wake Up Complete] Model 'sleep-assistant:latest' is live on RTX 3060."
        else
            echo "[Warning] No verified latest_model.gguf.sha256 found. Did the Safety Gate abort deployment?"
            exit 1
        fi
        ;;

    *)
        usage
        ;;
esac
