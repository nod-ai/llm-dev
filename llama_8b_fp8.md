# Llama 8B FP8
Irpa file: `/sharedfile/llama3_8b_fp8.irpa` on SharkMi300X

## Attention and Activation dtype configs
| Attention-dtype | Activation-dtype | Eager Prefill | Eager Decode | IREE Prefill | IREE Decode | Tracy Profile | Comments|
|-----------------|------------------|---------------|--------------|--------------|-------------|---------------|---------|
|bfloat16 | bfloat16 | PASS | PASS | PASS w/ no extra flags | TBD | | Compile w/o `--iree-dispatch-creation-enable-aggressive-fusion=true` |
|float8_e4m3fnuz | bfloat16 | [FAIL](https://gist.github.com/aviator19941/fe1f129557632896a8fabf573c973b5b) | FAIL | | | |

## Numerics:

Instructions to run perplexity tests for 8B fp8 model (non-fp8 attention)

Eager ppl:

```
python -m sharktank.evaluate.perplexity_torch \
  --irpa-file=fp8_attn.irpa \
  --tokenizer-config-json=/sharedfile/tokenizer_config.json \
  --attention-kernel=torch \
  --attention-dtype=bfloat16 \
  --activation-dtype=bfloat16 \
  --use-hf \
  --fake-quant\
  --num-prompts=4 \
  --device='cuda:0'
```

IREE ppl:
```
python -m sharktank.evaluate.perplexity_iree \
  --irpa-file=/sharedfile/llama3_8b_fp8.irpa \
  --tokenizer-config-json=/sharedfile/tokenizer_config.json \
  --attention-kernel torch \
  --iree-hal-target-device=hip \
  --iree-hip-target=gfx942 \
  --attention-dtype=bfloat16 \
  --activation-dtype=bfloat16 \
  --kv-cache-dtype=float8_e4m3fnuz \
  --use-hf \
  --num-prompts=4 \
  --iree-device='hip://0'
```

## Eager mode:
```
python -m sharktank.examples.paged_llm_v1 \
  "The capitol of Texas is" \
  --irpa-file=/sharedfile/llama3_8b_fp8.irpa \
  --tokenizer-config-json=/sharedfile/tokenizer_config.json \
  --attention-kernel=torch \
  --activation-dtype=bfloat16 \
  --attention-dtype=bfloat16 \
  --use-hf
```

## Export IR:
```
python3 -m sharktank.examples.export_paged_llm_v1 --irpa-file=/sharedfile/llama3_8b_fp8.irpa \
--output-mlir=fp8_dan1.mlir \
--output-config=config1.json \
--bs=1 --attention-kernel torch \
--attention-dtype=float8_e4m3fnuz \
--activation-dtype=bfloat16 \
--use-hf
```

## Compile:
Minimal flags:
```
../iree-build-no-trace/tools/iree-compile fp8_dan1.mlir \
  --iree-hip-target=gfx942 \
  -o=fp8_dan1.vmfb \
  --iree-hal-target-device=hip
```

Additional flags:
```
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
