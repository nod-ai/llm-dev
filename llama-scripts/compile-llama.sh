#!/bin/bash

# Base llama compilation script.
# Usage:
# ./compile-llama.sh <iree-compile-path> <gfxip> <input mlir> -o <output vmfb>

set -euo pipefail

readonly IREE_COMPILE="$(realpath "$1")"
if [ ! -f "$IREE_COMPILE" ] ; then
  echo "Specified iree-compile binary not found: ${IREE_COMPILE}"
  exit 1
fi

readonly CHIP="$2"

readonly INPUT="$(realpath "$3")"
if [ ! -f "$INPUT" ] ; then
  echo "Input mlir file not found: ${INPUT}"
  exit 1
fi

shift 3

set -x

"$IREE_COMPILE" "$INPUT" \
    --iree-hal-target-backends=rocm \
    --iree-hip-target="$CHIP" \
    "$@"
