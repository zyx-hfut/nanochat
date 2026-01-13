#!/bin/bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/home/featurize/data/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate


python -m scripts.chat_cli -i sft



