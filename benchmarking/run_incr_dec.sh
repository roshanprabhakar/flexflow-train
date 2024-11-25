#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"


./inference/incr_decoding/incr_decoding \
    -ll:cpu 16 -ll:gpu 8 -ll:util 16 \
    -ll:fsize 20000 -ll:zsize 30000 \
    --fusion \
    -llm-model meta-llama/Llama-3.1-8B-Instruct \
    -prompt ../benchmarking/test.json \
    -tensor-parallelism-degree 8 \
    -log-file ../inference/output/test.out \
    -output-file ../inference/output/test.json \
    --max-requests-per-batch 1 --max-tokens-per-batch 3000 --max-sequence-length 3000

