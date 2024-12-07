#!/bin/bash

# Target directory
TARGET_DIR="/root/.cache/flexflow/weights/m4-ai/tinymistral-6x248m/half-precision"

# Loop through files containing "layer_"
for file in "$TARGET_DIR"/*layer_*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//layer_/layer.}"
  fi
done

# Loop through files containing "_input"
for file in "$TARGET_DIR"/*_input*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//_input/.input}"
  fi
done

# Loop through files containing ".mlp."
for file in "$TARGET_DIR"/*_mlp_*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//.mlp./.mlp.}"
  fi
done

for file in "$TARGET_DIR"/*_post*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//_post/.post}"
  fi
done


for file in "$TARGET_DIR"/*_block*; do
  if [[ -f "$file" ]]; then
    mv "$file" "${file//_block/.block}"
  fi
done
