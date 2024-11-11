#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

make -j install

model_name=meta-llama/Llama-3.1-70B-Instruct
NGPUS=8
FSIZE=36000
ZSIZE=200000
CSIZE=200000
MAX_SEQ_LEN=7000
tokens_per_batch=1024

batch_sizes=(
    16
    8
)

dataset_fp="../wildchat/sharegpt.json"
partition_name="all"

export LEGION_BACKTRACE=1

for j in "${!batch_sizes[@]}"; do
    batch_size=${batch_sizes[$j]}
    
    echo "Running dataset ${dataset_fp} with model ${model_name}, batch size ${batch_size}, and tokens per batch ${tokens_per_batch}"
    # create model name version where "/" is replaced with "-"
    model_name_=$(echo $model_name | tr / -)
    log_fp="/usr/FlexFlow/inference/output/${dataset_fp}_incr_dec_offline_${model_name_}_${batch_size}.log"
    output_fp="/usr/FlexFlow/inference/output/${dataset_fp}_incr_dec_offline_${model_name_}_${batch_size}.json"
    metrics_fp="/usr/FlexFlow/inference/output/${dataset_fp}_incr_dec_offline_${model_name_}_${batch_size}.csv"
    rm $metrics_fp $output_fp $log_fp || true

    time ./inference/suffix_decoding/incr_dec \
        -ll:gpu $NGPUS -ll:cpu 16 -ll:util 16 \
        -tensor-parallelism-degree $NGPUS \
        -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
        --fusion \
        --max-sequence-length $MAX_SEQ_LEN \
        --max-requests-per-batch $batch_size \
        --max-tokens-per-batch $tokens_per_batch \
        --max-output-length 1024 \
        -llm-model $model_name \
        -trace ${dataset_fp} \
        -trace-output-path ${output_fp} \
        -output-file ${log_fp} \
        -csv-output-path $metrics_fp \
        -target-partition ${partition_name}
done