#!/bin/bash

# See speedrun.sh for more comments

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/home/featurize/data/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/ikm_mini_results"
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}
log "=============================================="
log "Mini IKM sft Training"
log "=============================================="
    

TAG="mini_ikm_d20"

python -u -m scripts.chat_sft --device_type="cuda" --device_batch_size=2 --model_tag="${TAG}" --eval_every=400 --num_epochs=1 2>&1 | tee "$RESULTS_DIR/${TAG}_sft_train.log"

log "=============================================="
log "IKM_mini sft training Complete!"
log "=============================================="


