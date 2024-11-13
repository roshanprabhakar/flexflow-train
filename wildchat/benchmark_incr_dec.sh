#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

# export BUILD_TYPE=Release
# ../config/config.linux
make -j install

model_name=meta-llama/Llama-3.1-70B-Instruct
NGPUS=8
NCPUS=16
FSIZE=36000
ZSIZE=170000
CSIZE=60000


# model_name=meta-llama/Llama-3.1-8B-Instruct
# NGPUS=8
# FSIZE=36000
# ZSIZE=30000
# CSIZE=100000



MAX_SEQ_LEN=7000
tokens_per_batch=1024

batch_sizes=(
    16
    8
)

dataset_name="sharegpt"
dataset_fp="../wildchat/${dataset_name}.json"
partition_name="all"

export LEGION_BACKTRACE=1

# python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='meta-llama/Llama-3.1-70B-Instruct', allow_patterns='*.safetensors', max_workers=30)"
# python ../inference/utils/download_hf_model.py --half-precision-only $model_name --refresh-cache

for j in "${!batch_sizes[@]}"; do
    batch_size=${batch_sizes[$j]}
    
    echo "Running dataset ${dataset_fp} with model ${model_name}, batch size ${batch_size}, and tokens per batch ${tokens_per_batch}"
    # create model name version where "/" is replaced with "-"
    model_name_=$(echo $model_name | tr / -)
    log_fp="/usr/FlexFlow/inference/output/incr_dec_offline_${model_name_}_${batch_size}_${dataset_name}.log"
    output_fp="/usr/FlexFlow/inference/output/incr_dec_offline_${model_name_}_${batch_size}_${dataset_name}.json"
    metrics_fp="/usr/FlexFlow/inference/output/incr_dec_offline_${model_name_}_${batch_size}_${dataset_name}.csv"
    rm $metrics_fp $output_fp $log_fp || true

    time ./inference/suffix_decoding/incr_dec \
        -ll:gpu $NGPUS -ll:cpu $NCPUS -ll:util $NCPUS \
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