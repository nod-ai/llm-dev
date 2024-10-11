#!/bin/bash

# Usage: PATH=/path/to/iree/build/tools:$PATH ./benchmark-unet.sh <N> <weights-path> <prefill-inputs-dir>

set -xeu

if (( $# != 1 && $# != 2 && $# != 3)); then
  echo "usage: $0 <hip-device-id> <weights-path> <prefill-inputs-dir>"
  exit 1
fi

ROCR_VISIBLE_DEVICES=$1 iree-benchmark-module \
  --device=hip://$1 \
  --hip_use_streams=true \
  --hip_allow_inline_execution=true \
  --device_allocator=caching \
  --module=$PWD/llama.vmfb \
  --parameters=model=$2 \
  --function=prefill_bs4 \
  --input=@$3/tokens.npy \
  --input=@$3/seq_lens.npy \
  --input=@$3/seq_block_ids.npy \
  --input=@$3/cache_state_f16.npy \
  --benchmark_repetitions=3

  # prefill input sizes
  # --input=4x16xsi64 \
  # --input=4xsi64 \
  # --input=4x1xsi64 \
  # --input=128x2621440xf16 (for llama3-70b) \