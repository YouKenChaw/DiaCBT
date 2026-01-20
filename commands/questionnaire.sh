#!/bin/bash

export PYTHONPATH=(dirname "$0")/..:${PYTHONPATH:-}
export OPENAI_API_KEY="your_gpt_key"
export OPENAI_API_BASE="base"

python ./code/scripts/questionnaire.py \
    --dataset C2D2 \
    --agent ours \
    --backbone ours \
    --questionnaire CTRS \
    --npc_backbone gpt-4o