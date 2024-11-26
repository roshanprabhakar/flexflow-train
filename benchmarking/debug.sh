#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="JackFram/llama-160m"
NGPUS=1

python ../inference/utils/download_hf_model.py $MODEL_NAME

export LEGION_BACKTRACE=1

./inference/incr_decoding/incr_decoding \
    -ll:cpu 16 -ll:gpu $NGPUS -ll:util 16 \
    -ll:fsize 20000 -ll:zsize 10000 \
    --fusion \
    -llm-model $MODEL_NAME \
    -prompt ../benchmarking/test.json \
    -tensor-parallelism-degree $NGPUS \
    -log-file ../inference/output/test.out \
    -output-file ../inference/output/test.json \
    --max-requests-per-batch 1 --max-tokens-per-batch 3000 --max-sequence-length 3000

