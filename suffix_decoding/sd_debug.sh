#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

reset
make -j
source set_python_envs.sh

matching_strategy="dynamic_token_tree"
max_tree_depth=16
max_spec_factor=1.0

# export CUDA_VISIBLE_DEVICES=1,2,3,5
export LEGION_BACKTRACE=1

########### LLAMA 70B on 8 GPUs ###########
model_name="meta-llama/Meta-Llama-3-70B-Instruct"
partition_name="FEATURE_EXTRACTION"
NGPUS=8
FSIZE=32000
ZSIZE=200000
CSIZE=200000

########### LLAMA 8B on 8 GPUs ###########
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
partition_name="SQL_FANOUT1"
NGPUS=1
FSIZE=30000
ZSIZE=100000
CSIZE=100000

echo "Running partition ${partition_name} with model ${model_name} and "
python ../inference/utils/download_hf_model.py --half-precision-only $model_name

rm usr/FlexFlow/inference/output/cortex_${partition_name}_sd.out || true

./inference/suffix_decoding/suffix_decoding \
    -ll:gpu $NGPUS -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree $NGPUS \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
    --fusion \
    --max-sequence-length 4700 \
    --max-requests-per-batch 1 \
    --max-tokens-per-batch 1024 \
    --max-output-length 900 \
    --matching-strategy $matching_strategy \
    --max-tree-depth $max_tree_depth \
    --max-spec-factor $max_spec_factor \
    -llm-model $model_name \
    -trace /usr/suffix-tree-decoding/trace/flexflow/cortex_ff_SQL_FANOUT1.json \
    -trace-output-path usr/FlexFlow/inference/output/cortex_ff_${partition_name}_sd.json \
    -output-file usr/FlexFlow/inference/output/cortex_${partition_name}_sd.out \
    -csv-output-path usr/FlexFlow/inference/output/cortex_${partition_name}_sd.csv \
    -target-partition ${partition_name}