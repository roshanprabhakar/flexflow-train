#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

reset
make -j
source set_python_envs.sh

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
partition_name="FEATURE_EXTRACTION"
matching_strategy="linear_token_path"
max_tree_depth=10
max_spec_factor=1.0

export CUDA_VISIBLE_DEVICES=7
export LEGION_BACKTRACE=1
FSIZE=36000
ZSIZE=30000
CSIZE=100000

echo "Running partition ${partition_name} with model ${model_name} and "
python ../inference/utils/download_hf_model.py --half-precision-only $model_name

rm /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}_sd.out || true

./inference/suffix_decoding/suffix_decoding \
    -ll:gpu 1 -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree 1 \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
    --fusion \
    --max-sequence-length 2000 \
    --max-requests-per-batch 1 \
    --max-tokens-per-batch 1024 \
    --max-output-length 900 \
    --matching-strategy $matching_strategy \
    --max-tree-depth $max_tree_depth \
    --max-spec-factor $max_spec_factor \
    --disable-online-tree-update \
    -llm-model $model_name \
    -trace /usr/suffix-tree-decoding/trace/flexflow/cortex_ff_FEATURE_EXTRACTION.json \
    -trace-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_ff_${partition_name}_sd.json \
    -output-file /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}_sd.out \
    -csv-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}_sd.csv \
    -target-partition ${partition_name} \
    --verbose