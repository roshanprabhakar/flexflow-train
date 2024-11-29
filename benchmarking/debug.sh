#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# PROMPT="../benchmarking/test.json"
PROMPT="/usr/FlexFlow/inference/prompt/peft.json"
MODEL_NAME="JackFram/llama-160m"
PEFT_MODEL_NAME="goliaro/llama-160m-lora"
NGPUS=1
NCPUS=4

reset
make -j install

# python ../inference/utils/download_hf_model.py $MODEL_NAME
python ../inference/utils/download_peft_model.py $PEFT_MODEL_NAME

mkdir -p ../inference/prompt
echo '["Two things are infinite: "]' > ../inference/prompt/peft.json
echo '["“Two things are infinite: the universe and human stupidity; and I'\''m not sure about the universe.”"]' > ../inference/prompt/peft_dataset.json
# Create output folder
mkdir -p ../inference/output


# export LEGION_BACKTRACE=1
export FF_DEBG_NO_WEIGHTS=1

export CUDA_VISIBLE_DEVICES=1

# gdb -ex run --args ./inference/incr_decoding/incr_decoding \
#     -ll:cpu $NCPUS -ll:gpu $NGPUS -ll:util $NCPUS \
#     -ll:fsize 20000 -ll:zsize 10000 \
#     -llm-model $MODEL_NAME --verbose \
#     -prompt $PROMPT \
#     -tensor-parallelism-degree $NGPUS \
#     -log-file ../inference/output/test.out \
#     -output-file ../inference/output/test.json \
#     --max-requests-per-batch 1 --max-tokens-per-batch 3000 --max-sequence-length 3000

#--verbose -lg:prof 1 -lg:prof_logfile prof_%.gz \

gdb -ex run --args ./inference/peft/peft \
    -ll:cpu 4 -ll:gpu $NGPUS -ll:util 4 \
    -ll:fsize 20000 -ll:zsize 10000 \
    --fusion \
    -llm-model $MODEL_NAME \
    -enable-peft -peft-model $PEFT_MODEL_NAME \
    -finetuning-dataset /usr/FlexFlow/inference/prompt/peft_dataset.json \
    -tensor-parallelism-degree $NGPUS \
    -output-file ../inference/output/test.json \
    --max-requests-per-batch 1 --max-tokens-per-batch 3000 --max-sequence-length 3000

# -lg:prof 1 -lg:prof_logfile prof_%.gz --verbose --inference-debugging \
## -prompt /usr/FlexFlow/inference/prompt/peft.json \