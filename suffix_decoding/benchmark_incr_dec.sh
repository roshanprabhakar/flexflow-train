#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

make -j
source set_python_envs.sh

model_name=meta-llama/Meta-Llama-3-70B-Instruct
NGPUS=8
FSIZE=70000
ZSIZE=200000
CSIZE=200000
MAX_SEQ_LEN=7000
tokens_per_batch=1024

# comment these lines in for debugging
model_name=meta-llama/Meta-Llama-3-8B-Instruct
FSIZE=32000
ZSIZE=30000
CSIZE=60000

partitions=(
    # QUESTION_SUGGESTION
    # CATEGORIZATION
    FEATURE_EXTRACTION
    # SQL_FANOUT1
    # SQL_FANOUT2
    # SQL_FANOUT3
    # SQL_COMBINE
)
batch_sizes=(
    8
    # 16
)

# download all models and small models
python ../inference/utils/download_hf_model.py --half-precision-only $model_name
export LEGION_BACKTRACE=1

for i in "${!partitions[@]}"; do
    for j in "${!batch_sizes[@]}"; do
        partition_name=${partitions[$i]}
        batch_size=${batch_sizes[$j]}
        
        echo "Running partition ${partition_name} with model ${model_name}, batch size ${batch_size}, and tokens per batch ${tokens_per_batch}"
        # create model name version where "/" is replaced with "-"
        model_name_=$(echo $model_name | tr / -)
        rm /home/yak/goliaro/FlexFlow/inference/output/cortex_incr_dec_${partition_name}_${model_name_}_${batch_size}.csv || true
        rm /home/yak/goliaro/FlexFlow/inference/output/cortex_incr_dec_${partition_name}_${model_name_}_${batch_size}.out || true

        time ./inference/suffix_decoding/incr_dec \
            -ll:gpu $NGPUS -ll:cpu 4 -ll:util 4 \
            -tensor-parallelism-degree $NGPUS \
            -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
            --fusion \
            --max-sequence-length $MAX_SEQ_LEN \
            --max-requests-per-batch $batch_size \
            --max-tokens-per-batch $tokens_per_batch \
            --max-output-length 900 \
            -llm-model $model_name \
            -trace /home/yak/goliaro/suffix-tree-decoding/trace/llama70b/cortex.json \
            -trace-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_incr_dec_${partition_name}.json \
            -output-file /home/yak/goliaro/FlexFlow/inference/output/cortex_incr_dec_${partition_name}_${model_name_}_${batch_size}.out \
            -csv-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_incr_dec_${partition_name}_${model_name_}_${batch_size}.csv \
            -target-partition ${partition_name}
    done
done