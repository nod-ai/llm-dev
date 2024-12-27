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

### Sharded

### 8B-FP16-Prefill-Sharded-TP8

```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.irpa \
  --output-mlir=8b_prefill_sharded.mlir \
  --output-config=8b_prefill_sharded.json \
  --skip-decode


iree-compile \
  8b_prefill_sharded.mlir \
  --iree-hip-target=gfx942 \
  -o=8b_prefill_sharded.vmfb \
  --iree-hal-target-device="hip[0]" \
  --iree-hal-target-device="hip[1]" \
  --iree-hal-target-device="hip[2]" \
  --iree-hal-target-device="hip[3]" \
  --iree-hal-target-device="hip[4]" \
  --iree-hal-target-device="hip[5]" \
  --iree-hal-target-device="hip[6]" \
  --iree-hal-target-device="hip[7]" \
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
  --module=8b_prefill_sharded.vmfb \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank0.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank1.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank2.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank3.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank4.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank5.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank6.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank7.irpa \
  --device=hip://0 \
  --device=hip://1 \
  --device=hip://2 \
  --device=hip://3 \
  --device=hip://4 \
  --device=hip://5 \
  --device=hip://6 \
  --device=hip://7 \
  --function=prefill_bs4 \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/tokens.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/seq_lens.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/seq_block_ids.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy \
  --input=@/data/llama3.1/weights/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy --benchmark_repetitions=8
```

### 8B-FP16-Decode-Sharded

```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.irpa \
  --output-mlir=8b_decode_sharded.mlir \
  --output-config=8b_decode_sharded.json \


iree-compile \
  8b_decode_sharded.mlir \
  --iree-hip-target=gfx942 \
  -o=8b_decode_sharded.vmfb \
  --iree-hal-target-device="hip[0]" \
  --iree-hal-target-device="hip[1]" \
  --iree-hal-target-device="hip[2]" \
  --iree-hal-target-device="hip[3]" \
  --iree-hal-target-device="hip[4]" \
  --iree-hal-target-device="hip[5]" \
  --iree-hal-target-device="hip[6]" \
  --iree-hal-target-device="hip[7]" \
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
  --module=8b_decode_sharded.vmfb \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank0.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank1.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank2.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank3.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank4.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank5.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank6.irpa \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank7.irpa \
  --device=hip://0 \
  --device=hip://1 \
  --device=hip://2 \
  --device=hip://3 \
  --device=hip://4 \
  --device=hip://5 \
  --device=hip://6 \
  --device=hip://7 \
  --function=decode_bs4 \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/next_tokens.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/seq_lens.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/start_positions.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/seq_block_ids.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy \
  --input=@/data/llama3.1/weights/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy --benchmark_repetitions=8
```
