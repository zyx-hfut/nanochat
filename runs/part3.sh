#!/bin/bash


# bash runs/speedrun.sh
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="./.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it


# run SFT and eval the model
python -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
python -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
