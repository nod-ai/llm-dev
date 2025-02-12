# Llama 8B FP8
Branch: `users/dan-garvey/enable_custom_fp8_matmul`

Irpa file: `/sharedfile/llama3_8b_fp8.irpa` on SharkMi300X

## Attention and Activation dtype configs
| Attention-dtype | Activation-dtype | Eager Prefill | Eager Decode | IREE Prefill | IREE Decode | Tracy Profile | Comments|
|-----------------|------------------|---------------|--------------|--------------|-------------|---------------|---------|
|bfloat16 | bfloat16 | PASS | PASS | FAIL (compile) | FAIL (compile) | | | |
|float8_e4m3fnuz | bfloat16 |  | | | | |  |

## Eager mode:
```
python -m sharktank.examples.paged_llm_v1 \
  "The capitol of Texas is" \
  --irpa-file=/sharedfile/llama3_8b_fp8.irpa \
  --tokenizer-config-json=/sharedfile/tokenizer_config.json \
  --attention-kernel=torch \
  --activation-dtype=bfloat16 \
  --save_intermediates_path="2_11" \
  --attention-dtype=bfloat16 \
  --fake-quant \
  --use-hf
```

## Export IR:
```
python3 -m sharktank.examples.export_paged_llm_v1 --irpa-file=/sharedfile/llama3_8b_fp8.irpa \
--output-mlir=fp8_dan1.mlir \
--output-config=config1.json \
--bs=1 --attention-kernel torch \
--attention-dtype=float8_e4m3fnuz --activation-dtype=bfloat16
```

## Compile:
```
../iree-build-no-trace/tools/iree-compile fp8_dan1.mlir \
  --iree-hip-target=gfx942 \
  -o=fp8_dan1.vmfb \
  --iree-hal-target-device=hip \
  --iree-dispatch-creation-enable-aggressive-fusion=true \
  --iree-global-opt-propagate-transposes=true \
  --iree-opt-aggressively-propagate-transposes=true \
  --iree-opt-data-tiling=false \
  --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' \
  --iree-hal-indirect-command-buffers=true \
  --iree-stream-resource-memory-model=discrete \
  --iree-hal-memoization=true \
  --iree-opt-strip-assertions
```

## Run:
```
../iree-build-no-trace/tools/iree-run-module \
--hip_use_streams=true \
--module=fp8_dan1.vmfb \
--parameters=model=/sharedfile/llama3_8b_fp8.irpa \
--device=hip://4 \
--function=prefill_bs1 \
--input=1x32xi64=@/sharedfile/prefill/prefill_token_ids_1_32.bin \
--input=1xi64=@/sharedfile/prefill/prefill_seq_lens_1.bin \
--input=1x1xi64=@/sharedfile/prefill/prefill_seq_block_ids_1_1.bin \
--input=128x2097152xf8E4M3FNUZ=@/sharedfile/prefill/prefill_cache_state_128_2097152.bin
```
