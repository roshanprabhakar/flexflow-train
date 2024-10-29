#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

make -j
source set_python_envs.sh

model_name=meta-llama/Meta-Llama-3-70B-Instruct
small_model_name=meta-llama/Meta-Llama-3-8B-Instruct
NGPUS=8
FSIZE=70000
ZSIZE=200000
CSIZE=200000
MAX_SEQ_LEN=2000
tokens_per_batch=1024

# # comment these lines in for debugging
# model_name=meta-llama/Meta-Llama-3-8B-Instruct
# FSIZE=30000
# ZSIZE=90000
# CSIZE=120000

partitions=(
    QUESTION_SUGGESTION
    CATEGORIZATION
    FEATURE_EXTRACTION
    SQL_FANOUT1
    SQL_FANOUT2
    SQL_FANOUT3
    SQL_COMBINE
)
batch_sizes=(
    # 1
    8
    # 16
)

max_tree_depths=(
    # 4
    8
    # 16
)
expansion_degrees=(
    # 1
    3
    # 4
)

# download all models and small models
python ../inference/utils/download_hf_model.py --half-precision-only $model_name $small_model_name
export LEGION_BACKTRACE=1

for i in "${!partitions[@]}"; do
    for j in "${!batch_sizes[@]}"; do
    for k in "${!max_tree_depths[@]}"; do
    for l in "${!expansion_degrees[@]}"; do
        partition_name=${partitions[$i]}
        batch_size=${batch_sizes[$j]}
        max_tree_depth=${max_tree_depths[$k]}
        expansion_degree=${expansion_degrees[$l]}
        
        echo "Running partition ${partition_name} with model ${model_name}, ssm ${ssm}, batch size ${batch_size}, and tokens per batch ${tokens_per_batch} with max tree depth ${max_tree_depth} and expansion degree ${expansion_degree}"
        # create model name version where "/" is replaced with "-"
        model_name_=$(echo $model_name | tr / -)
        small_model_name_=$(echo $small_model_name | tr / -)
        rm /home/yak/goliaro/FlexFlow/inference/output/cortex_specinfer_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${max_tree_depth}_${expansion_degree}.csv || true
        rm /home/yak/goliaro/FlexFlow/inference/output/cortex_specinfer_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${max_tree_depth}_${expansion_degree}.out || true

        time ./inference/suffix_decoding/specinfer \
            -ll:gpu $NGPUS -ll:cpu 4 -ll:util 4 \
            -tensor-parallelism-degree $NGPUS \
            -ll:fsize $FSIZE -ll:zsize $ZSIZE -ll:csize $CSIZE \
            --fusion \
            --max-sequence-length $MAX_SEQ_LEN \
            --max-requests-per-batch $batch_size \
            --max-tokens-per-batch $tokens_per_batch \
            --max-output-length 900 \
            --max-tree-depth ${max_tree_depth} \
            --expansion-degree ${expansion_degree} \
            -llm-model $model_name \
            -ssm-model $small_model_name \
            -trace /home/yak/goliaro/suffix-tree-decoding/trace/llama70b/cortex.json \
            -trace-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_ff_speciner_${partition_name}.json \
            -output-file /home/yak/goliaro/FlexFlow/inference/output/cortex_specinfer_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${max_tree_depth}_${expansion_degree}.out \
            -csv-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_specinfer_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${max_tree_depth}_${expansion_degree}.csv \
            -target-partition ${partition_name}
    done
    done
    done
done