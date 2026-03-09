#!/bin/bash


# bash runs/speedrun.sh
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh


export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="./.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 9.5)
python -m scripts.base_train -- --depth=24 --target-param-data-ratio=9.5 --device-batch-size=8 --fp8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
python -m scripts.base_eval -- --device-batch-size=8
