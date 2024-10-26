#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

reset

make -j
source set_python_envs.sh

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
small_model_name="Felladrin/Llama-160M-Chat-v1"
partition_name="FEATURE_EXTRACTION"

export LEGION_BACKTRACE=1
FSIZE=30000
ZSIZE=30000
CSIZE=100000

echo "Running partition ${partition_name} with model ${model_name} and small model ${small_model_name}"
python ../inference/utils/download_hf_model.py --half-precision-only $model_name $small_model_name

rm /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}.out || true

./inference/suffix_decoding/specinfer \
    -ll:gpu 8 -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree 8 \
    -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
    --fusion \
    -ll:force_kthreads \
    --max-sequence-length 2000 \
    --max-requests-per-batch 1 \
    --max-tokens-per-batch 1024 \
    --max-output-length 900 \
    --max-tree-depth 4 \
    --expansion-degree 3 \
    -llm-model $model_name \
    -ssm-model $small_model_name \
    -trace /usr/FlexFlow/build/_deps/suffix_decoding-src/trace/cortex.json \
    -trace-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_ff_${partition_name}.json \
    -output-file /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}.out \
    -csv-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}.csv \
    -target-partition ${partition_name} \
    --verbose