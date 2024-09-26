#!/bin/bash

# Llama benchmark script.
# Usage:
# ./benchmark-llama.sh <device-id> <iree-benchmark-path> <vmfb-path> <weights-path> <function-name> <inputs-dir>

set -euo pipefail

readonly DEVICE_ID="$1"

readonly IREE_BENCHMARK="$(realpath "$2")"
if [ ! -f "$IREE_BENCHMARK" ] ; then
  echo "Specified iree-benchmark-module binary not found: ${IREE_BENCHMARK}"
  exit 1
fi

readonly VMFB_PATH="$(realpath "$3")"
if [ ! -f "$VMFB_PATH" ] ; then
  echo "Specified vmfb path not found: ${VMFB_PATH}"
  exit 1
fi

readonly WEIGHTS_PATH="$(realpath "$4")"
if [ ! -f "$WEIGHTS_PATH" ] ; then
  echo "Specified weights path not found: ${WEIGHTS_PATH}"
  exit 1
fi

readonly FUNCTION_NAME="$5"

readonly INPUTS_DIR="$(realpath "$6")"
if [ ! -d "$INPUTS_DIR" ] ; then
  echo "Specified input directory path not found: ${INPUTS_DIR}"
  exit 1
fi

shift 6

set -x

# llama 8B inputs
ROCR_VISIBLE_DEVICES="$DEVICE_ID" "$IREE_BENCHMARK" \
  --device=hip://"$DEVICE_ID" \
  --hip_use_streams=true \
  --hip_allow_inline_execution=true \
  --device_allocator=caching \
  --module="$VMFB_PATH" \
  --parameters=model="$WEIGHTS_PATH" \
  --function="$FUNCTION_NAME" \
  --input=4x16xsi64=@"$INPUTS_DIR"/tokens.npy \
  --input=4xsi64=@"$INPUTS_DIR"/seq_lens.npy \
  --input=4x1xsi64=@"$INPUTS_DIR"/seq_block_ids.npy \
  --input=128x1048576xf16=@"$INPUTS_DIR"/cache_state_f16.npy \
  --benchmark_repetitions=3