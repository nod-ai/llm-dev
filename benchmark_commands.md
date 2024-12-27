### Unsharded

### 8B-FP16-Prefill

```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=/data/llama3.1/weights/8b/fp16/llama3.1_8b_fp16.irpa \
  --output-mlir=8b_fp16_prefill.mlir \
  --output-config=8b_fp16_prefill.json \
  --skip-decode

iree-compile 8b_fp16_prefill.mlir \
  --iree-hip-target=gfx942 \
  -o=prefill_8b_unsharded.vmfb \
  --iree-hal-target-device=hip \
  --iree-dispatch-creation-enable-aggressive-fusion=true \
  --iree-global-opt-propagate-transposes=true \
  --iree-opt-aggressively-propagate-transposes=true \
  --iree-opt-data-tiling=false \
  --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' \
  --iree-hal-indirect-command-buffers=true \
  --iree-stream-resource-memory-model=discrete \
  --iree-hip-legacy-sync=false \
  --iree-hal-memoization=true \
  --iree-opt-strip-assertions



ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
iree-benchmark-module \
  --hip_use_streams=true \
  --module=prefill_8b_unsharded.vmfb \
  --parameters=model=/data/llama3.1/weights/8b/fp16/llama3.1_8b_fp16.irpa \
  --device=hip://4 \
  --function=prefill_bs4 \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/tokens.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/seq_lens.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/seq_block_ids.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32/cs_f16.npy --benchmark_repetitions=8
```

### 8B-FP16-Decode

```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=/home/sai/temp_dir_by_dhiraj/gitRepo/llama_end_to_end/export/meta-llama-3.1-8b-instruct.f16.gguf \
  --output-mlir=8b_fp16_decode_unsharded.mlir \
  --output-config=8b_fp16_decode_unsharded.json

iree-compile 8b_fp16_decode_unsharded.mlir \
  --iree-hip-target=gfx942 \
  -o=8b_fp16_decode_unsharded.vmfb \
  --iree-hal-target-device=hip \
  --iree-dispatch-creation-enable-aggressive-fusion=true \
  --iree-global-opt-propagate-transposes=true \
  --iree-opt-aggressively-propagate-transposes=true \
  --iree-opt-data-tiling=false \
  --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' \
  --iree-hal-indirect-command-buffers=true \
  --iree-stream-resource-memory-model=discrete \
  --iree-hip-legacy-sync=false \
  --iree-hal-memoization=true \
  --iree-opt-strip-assertions 

ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \
iree-benchmark-module \
--hip_use_streams=true \
  --module=8b_fp16_decode_unsharded.vmfb \
  --parameters=model=/data/llama3.1/weights/8b/fp16/llama3.1_8b_fp16.irpa \
  --device=hip://4 \
  --function=decode_bs4 \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/next_tokens.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/seq_lens.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/start_positions.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/seq_block_ids.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32/cs_f16.npy --benchmark_repetitions=8

```

### 70B-FP16-Prefill

```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=/data/llama3.1/weights/70b/fp16/llama3.1_70b_f16.irpa \
  --output-mlir=prefill_70b_unsharded.mlir \
  --output-config=prefill_70b_unsharded.json \
  --skip-decode

iree-compile prefill_70b_unsharded.mlir \
  --iree-hip-target=gfx942 \
  -o=prefill_70b_unsharded.vmfb \
  --iree-hal-target-device=hip \
  --iree-dispatch-creation-enable-aggressive-fusion=true \
  --iree-global-opt-propagate-transposes=true \
  --iree-opt-aggressively-propagate-transposes=true \
  --iree-opt-data-tiling=false \
  --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' \
  --iree-hal-indirect-command-buffers=true \
  --iree-stream-resource-memory-model=discrete \
  --iree-hip-legacy-sync=false \
  --iree-hal-memoization=true \
  --iree-opt-strip-assertions 


ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
iree-benchmark-module \
  --hip_use_streams=true \
  --module=prefill_70b_unsharded.vmfb \
  --parameters=model=/data/llama3.1/weights/70b/fp16/llama3.1_70b_f16.irpa \
  --device=hip://4 \
  --function=prefill_bs4 \
  --input=@/data/llama3.1/weights/70b/prefill_args_bs4_128_stride_32/tokens.npy \
  --input=@/data/llama3.1/weights/70b/prefill_args_bs4_128_stride_32/seq_lens.npy \
  --input=@/data/llama3.1/weights/70b/prefill_args_bs4_128_stride_32/seq_block_ids.npy \
  --input=@/data/llama3.1/weights/70b/prefill_args_bs4_128_stride_32/cs_f16.npy --benchmark_repetitions=8
```

### 70B-FP16-Decode-Unsharded

```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=/data/llama3.1/weights/70b/fp16/llama3.1_70b_f16.irpa \
  --output-mlir=70b_fp16_decode_unsharded.mlir \
  --output-config=70b_fp16_decode_unsharded.json

iree-compile 70b_fp16_decode_unsharded.mlir \
  --iree-hip-target=gfx942 \
  -o=70b_fp16_decode_unsharded.vmfb \
  --iree-hal-target-device=hip \
  --iree-dispatch-creation-enable-aggressive-fusion=true \
  --iree-global-opt-propagate-transposes=true \
  --iree-opt-aggressively-propagate-transposes=true \
  --iree-opt-data-tiling=false \
  --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' \
  --iree-hal-indirect-command-buffers=true \
  --iree-stream-resource-memory-model=discrete \
  --iree-hip-legacy-sync=false \
  --iree-hal-memoization=true \
  --iree-opt-strip-assertions 

	ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \
iree-benchmark-module \
--hip_use_streams=true \
  --module=70b_fp16_decode_unsharded.vmfb \
  --parameters=model=/data/llama3.1/weights/70b/fp16/llama3.1_70b_f16.irpa \
  --device=hip://4 \
  --function=decode_bs4 \
  --input=@/data/llama3.1/weights/70b/decode_args_bs4_128_stride_32/next_tokens.npy \
  --input=@/data/llama3.1/weights/70b/decode_args_bs4_128_stride_32/seq_lens.npy \
  --input=@/data/llama3.1/weights/70b/decode_args_bs4_128_stride_32/start_positions.npy \
  --input=@/data/llama3.1/weights/70b/decode_args_bs4_128_stride_32/seq_block_ids.npy \
  --input=@/data/llama3.1/weights/70b/decode_args_bs4_128_stride_32/cs_f16.npy --benchmark_repetitions=8
```



