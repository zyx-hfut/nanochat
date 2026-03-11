#!/bin/bash


# bash runs/speedrun.sh
# screen -L -Logfile runs/part2.log -S part2 bash runs/part2.sh

export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/mnt/data/nanochat/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

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
python -m scripts.base_train --depth=20 --target-param-data-ratio=9.5 --device-batch-size=16 --fp8 --run=$WANDB_RUN --sample-every=500
# evaluate the model: CORE metric, BPB on train/val, and draw samples
python -m scripts.base_eval --device-batch-size=16
