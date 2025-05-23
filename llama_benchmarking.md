# Benchmark Llama 3.1 Unsharded (TP1)
In order to benchmark Llama 3.1 prefill and decode, you will need these artifacts for unsharded (TP=1) benchmarks:

1. irpa file(s)
2. IR
3. prefill numpy inputs
4. decode numpy inputs

## 0. Set up venv
a. Clone `shark-ai`:
```
git clone https://github.com/nod-ai/shark-ai.git
```

b. Set up env:
https://github.com/nod-ai/shark-ai/blob/main/docs/developer_guide.md#setup-a-venv


## 1. Get the unsharded irpa files
Create a SAS token in Azure:
- Go to the [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/overview) storage account in the Azure portal
- In the `Security + networking` dropdown, click `Shared access signature`
- Under `Allowed resource types` select Service, Container, and Object
- Scroll down to the bottom and select `Generate SAS and connection string`
- Scroll down and Copy the SAS token
- Replace [Add your SAS token here] (including the [ and ]) by SAS token string in instructions below 

```
azcopy copy \
'https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/8b_f16.irpa?[Add SAS token here]' \
'8b_fp16.irpa'
```

If you have trouble accessing `sharkblobs`, you can copy the 8b f16 unsharded irpa file from the `SharkMi300x` machine:
```
scp nod@10.23.233.219:/data/llama3.1/weights/8b/fp16/llama3.1_8b_fp16.irpa 8b_fp16.irpa
```

N.B.: Weight files for Llama-3.1-Instruct-<8B/70B/405B> (GGUF) are located on the `SharkMi300X` machine at:
```
/shark-dev/<8b/70b/405b>/instruct/weights/llama3_<8b/70b/405b>_instruct_fp16.gguf
```

## 2. Generate the IR
a. To generate the IR for prefill only:
```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=8b_fp16.irpa \
  --output-mlir=8b_fp16_prefill_nondecomposed.mlir \
  --output-config=8b_fp16_prefill_nondecomposed.json \
  --skip-decode
```

To generate the IR for both prefill + decode (remove the `--skip-decode` flag):
```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=8b_fp16.irpa \
  --output-mlir=8b_fp16_nondecomposed.mlir \
  --output-config=8b_fp16_nondecomposed.json
```

## 3. Get the numpy inputs

Get the 8b f16 tp1 unsharded prefill numpy inputs: [get_8b_fp16_tp1_prefill_inputs.sh](https://gist.github.com/aviator19941/380acabc77aeb4749fac14262e17db69)

Get the 8b f16 tp1 unsharded decode numpy inputs: [get_8b_fp16_tp1_decode_inputs.sh](https://gist.github.com/aviator19941/5f7db8ada6a4a95efe1d9a7975fed276)

## 4. Compile command
This command compiles the full IR (both prefill + decode) into a vmfb.

```
../iree-build-no-trace/tools/iree-compile 8b_fp16_prefill_nondecomposed.mlir \
  --iree-hip-target=gfx942 \
  -o=prefill_8b.vmfb \
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
## 5. Benchmark command
In order to benchmark prefill, make sure you specify the function as `prefill_bs{batch_size}` and specify the 4 inputs using the numpy files in 
`prefill_args_bs4_128_stride_32`.

Prefill benchmark command:

```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-benchmark-module \
  --hip_use_streams=true \
  --module=prefill_8b.vmfb \
  --parameters=model=8b_fp16.irpa \
  --device=hip://4 \
  --function=prefill_bs4 \
  --input=@prefill_args_bs4_128_stride_32/tokens.npy \
  --input=@prefill_args_bs4_128_stride_32/seq_lens.npy \
  --input=@prefill_args_bs4_128_stride_32/seq_block_ids.npy \
  --input=@prefill_args_bs4_128_stride_32/cs_f16.npy \
  --benchmark_repetitions=3
```

In order to benchmark decode, make sure you specify the function as `decode_bs{batch_size}` and specify the 5 inputs using the numpy files in `decode_args_bs4_128_stride_32`.

Decode benchmark command:

```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \
  ../iree-build-no-trace/tools/iree-benchmark-module \
  --hip_use_streams=true \
  --module=8b_fp16_nondecomposed_32.vmfb \
  --parameters=model=8b_fp16.irpa \
  --device=hip://4 \
  --function=decode_bs4 \
  --input=@decode_args_bs4_128_stride_32/next_tokens.npy \
  --input=@decode_args_bs4_128_stride_32/seq_lens.npy \
  --input=@decode_args_bs4_128_stride_32/start_positions.npy \
  --input=@decode_args_bs4_128_stride_32/seq_block_ids.npy \
  --input=@decode_args_bs4_128_stride_32/cs_f16.npy \
  --benchmark_repetitions=3
```

## 6. Get tracy file
Build IREE with runtime tracing and tracy:
```
cmake -G Ninja -B ../iree-build-trace   -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo   \
-DIREE_ENABLE_ASSERTIONS=ON   -DCMAKE_C_COMPILER=clang   \
-DCMAKE_CXX_COMPILER=clang++   -DIREE_ENABLE_RUNTIME_TRACING=ON   \
-DIREE_BUILD_TRACY=ON   -DIREE_ENABLE_LLD=ON   \
-DIREE_BUILD_PYTHON_BINDINGS=ON   \
-DPython3_EXECUTABLE="$(which python3)"  \
-DIREE_TARGET_BACKEND_CUDA=OFF -DIREE_HAL_DRIVER_HIP=ON \
-DIREE_TARGET_BACKEND_ROCM=ON .

cmake --build ../iree-build-trace
```

Compile with trace:
```
../iree-build-trace/tools/iree-compile \
  ../SHARK-Platform/8b_fp16_prefill_nondecomposed.mlir \
  --iree-hip-target=gfx942 \
  -o=prefill_8b.vmfb \
  --iree-hal-target-device=hip \
  --iree-dispatch-creation-enable-aggressive-fusion=true \
  --iree-global-opt-propagate-transposes=true \
  --iree-opt-aggressively-propagate-transposes=true \
  --iree-opt-data-tiling=false \
  --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' \
  --iree-hal-indirect-command-buffers=true \
  --iree-stream-resource-memory-model=discrete \
  --iree-hal-memoization=true \
  --iree-opt-strip-assertions \
  --iree-hal-executable-debug-level=3 \
  --iree-hal-dump-executable-sources-to=dump
```

Run `iree-run-module` with `TRACY_NO_EXIT=1`:
```
TRACY_NO_EXIT=1 \
  ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-trace/tools/iree-run-module \
  --hip_use_streams=true \
  --module=prefill_8b.vmfb \
  --parameters=model=8b_fp16.irpa \
  --device=hip://4 \
  --function=prefill_bs4 \
  --input=@prefill_args_bs4_128_stride_32/tokens.npy \
  --input=@prefill_args_bs4_128_stride_32/seq_lens.npy \
  --input=@prefill_args_bs4_128_stride_32/seq_block_ids.npy \
  --input=@prefill_args_bs4_128_stride_32/cs_f16.npy
```

Open another terminal and run this command to capture the tracy file:
```
../iree-build-trace/tracy/iree-tracy-capture -f -o prefill_8b.tracy
```

# Benchmark Llama 3.1 TP8 Sharded
## 1. Set up TP>1 sharded artifacts (optional)
Given a non-sharded irpa file, if you want to create your own TP8 sharded irpa files use this command:
```
python3 -m sharktank.examples.sharding.shard_llm_dataset \
  --irpa-file 405b_fp16.irpa \
  --output-irpa 405b_fp16_tp8.irpa \
  --tensor-parallelism-size 8
```

## 2. Download sharded irpa files
Create a SAS token in Azure:
Follow instructions [here](https://github.com/nod-ai/llm-dev/edit/main/llama_benchmarking.md#1-get-the-unsharded-irpa-files).

The sharded irpa files for 405b have already been generated and stored. In order to download them, use this command:
```
azcopy copy \
  'https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/tp8?[Add SAS token here]' \
  '405b_tp8_irpa' --recursive
```

### X-1 machine
```
8b weights: /shark-dev/data/llama3.1/weights/8b/fp16/tp8/
70b weights: /shark-dev/data/llama3.1/weights/70b/fp16/tp8/
405b weights: /shark-dev/data/llama3.1/weights/405b/fp16/tp8/
```

## 3. Generate the sharded IR

To generate the sharded IR use the unranked sharded irpa file:
### Export 8b sharded IR
```
python -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.irpa \
  --output-mlir=8b_instruct_tp8.mlir \
  --output-config=8b_instruct_tp8.json
```

### Export 405b sharded IR

You need to use the unranked sharded irpa file to generate the full sharded IR. In order to generate IR
for only prefill, add the `--skip-decode` flag to the command:

```
python3 -m sharktank.examples.export_paged_llm_v1 \
  --bs=4 \
  --irpa-file=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.irpa \
  --output-mlir=405b_f16_tp8.mlir \
  --output-config=405b_f16_tp8.json
```

## 4. Get the TP8 sharded numpy inputs:

Get the 405b f16 tp8 prefill numpy inputs: [get_405b_tp8_prefill_inputs.sh](https://gist.github.com/aviator19941/97323fee3524d193c0dff2653d6a2a86)

Get the 405b f16 tp8 decode numpy inputs: [get_405b_tp8_decode_inputs.sh](https://gist.github.com/aviator19941/a874d3cc03649abbfecc5dac27c62eda)

### 8b-Instruct, 70b-Instruct, 405b-Instruct TP8 numpy inputs:
X-1:
```
/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8
/shark-dev/8b/decode_args_bs4_128_stride_32_tp8
/shark-dev/70b/prefill_args_bs4_128_stride_32_tp8
/shark-dev/70b/decode_args_bs4_128_stride_32_tp8
/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8
/shark-dev/405b/decode_args_bs4_128_stride_32_tp8
```

## 5. Build IREE for tracy profiling

Build IREE with runtime tracing and tracy:
```
cmake -G Ninja -B ../iree-build-trace   -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo   \
-DIREE_ENABLE_ASSERTIONS=ON   -DCMAKE_C_COMPILER=clang   \
-DCMAKE_CXX_COMPILER=clang++   -DIREE_ENABLE_RUNTIME_TRACING=ON   \
-DIREE_BUILD_TRACY=ON   -DIREE_ENABLE_LLD=ON   \
-DIREE_BUILD_PYTHON_BINDINGS=ON   \
-DPython3_EXECUTABLE="$(which python3)"  \
-DIREE_TARGET_BACKEND_CUDA=OFF -DIREE_HAL_DRIVER_HIP=ON \
-DIREE_TARGET_BACKEND_ROCM=ON .

cmake --build ../iree-build-trace
```
## 5b. Compile sharded IR

### Compile 8b tp8
```
iree-compile \
    8b_instruct_tp8.mlir \
    --iree-hip-target=gfx942 \
    -o=8b_instruct_tp8.vmfb \
    --iree-hal-target-device=hip[0] \
    --iree-hal-target-device=hip[1] \
    --iree-hal-target-device=hip[2] \
    --iree-hal-target-device=hip[3] \
    --iree-hal-target-device=hip[4] \
    --iree-hal-target-device=hip[5] \
    --iree-hal-target-device=hip[6] \
    --iree-hal-target-device=hip[7] \
    --iree-dispatch-creation-enable-aggressive-fusion=true \
    --iree-global-opt-propagate-transposes=true \
    --iree-opt-aggressively-propagate-transposes=true \
    --iree-opt-data-tiling=false \
    --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' \
    --iree-stream-resource-memory-model=discrete \
    --iree-hal-indirect-command-buffers=true \
    --iree-hal-memoization=true \
    --iree-opt-strip-assertions
```

### Compile 405b tp8
```
~/iree-build-trace/tools/iree-compile --compile-to=input  \
artifacts/405b_f16_prefill_tp8.mlir  \
-o artifacts/405b_f16_prefill_tp8.iree.mlir

~/iree-build-trace/tools/iree-compile  \
artifacts/405b_f16_prefill_tp8.iree.mlir  \
--iree-hip-target=gfx942  \
--iree-hal-target-device=hip[0]  \
--iree-hal-target-device=hip[1]  \
--iree-hal-target-device=hip[2]  \
--iree-hal-target-device=hip[3]  \
--iree-hal-target-device=hip[4]  \
--iree-hal-target-device=hip[5]  \
--iree-hal-target-device=hip[6]  \
--iree-hal-target-device=hip[7]  \
--iree-dispatch-creation-enable-aggressive-fusion=true     \
--iree-global-opt-propagate-transposes=true  \
--iree-opt-aggressively-propagate-transposes=true     \
--iree-opt-data-tiling=false  \
--iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))'     \
--iree-hal-indirect-command-buffers=true  \
--iree-stream-resource-memory-model=discrete  \
--iree-hal-memoization=true  \
--iree-opt-strip-assertions \
--iree-hal-executable-debug-level=3 \
--iree-hal-dump-executable-sources-to=dump \
--mlir-print-debuginfo \
-o=artifacts/prefill_405b_tp8_tracy.vmfb
```

## 4b. iree-run-module (optional)

### 8b tp8 prefill
```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-run-module   \
  --hip_use_streams=true   \
  --module=8b_instruct_tp8.vmfb   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank0.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank1.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank2.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank3.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank4.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank5.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank6.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank7.irpa   \
  --device=hip://0   \
  --device=hip://1   \
  --device=hip://2   \
  --device=hip://3   \
  --device=hip://4   \
  --device=hip://5   \
  --device=hip://6   \
  --device=hip://7   \
  --function=prefill_bs4   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/tokens.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/seq_lens.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
  --benchmark_repetitions=3
```
### 8b tp8 decode
```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-run-module   \
  --hip_use_streams=true   \
  --module=8b_instruct_tp8.vmfb   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank0.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank1.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank2.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank3.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank4.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank5.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank6.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank7.irpa   \
  --device=hip://0   \
  --device=hip://1   \
  --device=hip://2   \
  --device=hip://3   \
  --device=hip://4   \
  --device=hip://5   \
  --device=hip://6   \
  --device=hip://7   \
  --function=decode_bs4   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/next_tokens.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/seq_lens.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/start_positions.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
  --benchmark_repetitions=3
```

### 405b tp8 prefill
Adapt as per model as your artifacts names, following example is for 405B TP8 sharded run:

```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-run-module   \
  --hip_use_streams=true   \
  --module=405b_instruct_tp8.vmfb   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank0.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank1.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank2.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank3.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank4.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank5.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank6.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank7.irpa   \
  --device=hip://0   \
  --device=hip://1   \
  --device=hip://2   \
  --device=hip://3   \
  --device=hip://4   \
  --device=hip://5   \
  --device=hip://6   \
  --device=hip://7   \
  --function=prefill_bs4   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/tokens.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/seq_lens.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
  --benchmark_repetitions=3
```
### 405b tp8 decode
```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-run-module   \
  --hip_use_streams=true   \
  --module=405b_f16_tp8.vmfb   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank0.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank1.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank2.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank3.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank4.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank5.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank6.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank7.irpa   \
  --device=hip://0   \
  --device=hip://1   \
  --device=hip://2   \
  --device=hip://3   \
  --device=hip://4   \
  --device=hip://5   \
  --device=hip://6   \
  --device=hip://7   \
  --function=decode_bs4   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/next_tokens.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/seq_lens.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/start_positions.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
  --benchmark_repetitions=3
```

## 6. Benchmark sharded vmfb 

### 8b tp8 prefill
```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-benchmark-module   \
  --hip_use_streams=true   \
  --module=export/8b_instruct_tp8.vmfb   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank0.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank1.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank2.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank3.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank4.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank5.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank6.irpa   \
  --parameters=model=/shark-dev/8b/instruct/weights/tp8/llama3.1_8b_instruct_fp16_tp8.rank7.irpa   \
  --device=hip://0   \
  --device=hip://1   \
  --device=hip://2   \
  --device=hip://3   \
  --device=hip://4   \
  --device=hip://5   \
  --device=hip://6   \
  --device=hip://7   \
  --function=prefill_bs4   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/tokens.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/seq_lens.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
  --input=@/shark-dev/8b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
  --benchmark_repetitions=3
```
### 8b tp8 decode
```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-benchmark-module   \
  --hip_use_streams=true   \
  --module=export/8b_instruct_tp8.vmfb   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank0.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank1.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank2.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank3.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank4.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank5.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank6.irpa   \
  --parameters=model=/data/llama3.1/weights/8b/fp16/tp8/llama3.1_8b_fp16_tp8_parameters.rank7.irpa   \
  --device=hip://0   \
  --device=hip://1   \
  --device=hip://2   \
  --device=hip://3   \
  --device=hip://4   \
  --device=hip://5   \
  --device=hip://6   \
  --device=hip://7   \
  --function=decode_bs4   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/next_tokens.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/seq_lens.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/start_positions.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
  --input=@/shark-dev/8b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
  --benchmark_repetitions=3
```

### 405b tp8 prefill
```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-benchmark-module   \
  --hip_use_streams=true   \
  --module=405b_f16_tp8.vmfb   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank0.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank1.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank2.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank3.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank4.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank5.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank6.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank7.irpa   \
  --device=hip://0   \
  --device=hip://1   \
  --device=hip://2   \
  --device=hip://3   \
  --device=hip://4   \
  --device=hip://5   \
  --device=hip://6   \
  --device=hip://7   \
  --function=prefill_bs4   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/tokens.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/seq_lens.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
  --benchmark_repetitions=3
```
### 405b tp8 decode
```
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-benchmark-module   \
  --hip_use_streams=true   \
  --module=405b_f16_tp8.vmfb   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank0.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank1.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank2.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank3.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank4.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank5.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank6.irpa   \
  --parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank7.irpa   \
  --device=hip://0   \
  --device=hip://1   \
  --device=hip://2   \
  --device=hip://3   \
  --device=hip://4   \
  --device=hip://5   \
  --device=hip://6   \
  --device=hip://7   \
  --function=decode_bs4   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/next_tokens.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/seq_lens.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/start_positions.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
  --input=@/shark-dev/405b/decode_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy \
  --benchmark_repetitions=3
```

## 7. Collect Tracy Profile
Run tracy profile collection, replace the IP address and port as per your case or choice:

```
TRACY_PORT=8086 TRACY_NO_EXIT=1 ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \
~/iree-build-trace/tools/iree-run-module -run-module  --hip_use_streams=true \
--module=artifacts/prefill_405b_tp8_tracy.vmfb  \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.irpa   \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank0.irpa   \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank1.irpa   \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank2.irpa   \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank3.irpa   \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank4.irpa   \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank5.irpa   \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank6.irpa   \
--parameters=model=/shark-dev/data/llama3.1/weights/405b/fp16/tp8/llama3.1_405b_fp16_tp8_parameters.rank7.irpa   \
--device=hip://0  --device=hip://1  --device=hip://2  --device=hip://3   \
--device=hip://4  --device=hip://5  --device=hip://6  --device=hip://7   \
--function=prefill_bs4  --input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/tokens.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/seq_lens.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/seq_block_ids.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_0.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_1.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_2.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_3.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_4.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_5.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_6.npy   \
--input=@/shark-dev/405b/prefill_args_bs4_128_stride_32_tp8/cs_f16_shard_7.npy  
```
You can replace iree-run-module by iree-benchmark-module and add  --benchmark_repetitions=3 to run benchmark instead.

And in a separate terminal run following (-s 35 colelcts for 35 seconds, you can remove it and try, at present I see that 405B cannot colelct more than 35 seconds):
```
 ~/iree-build-trace/tracy/iree-tracy-capture -f -a 172.19.26.15 -p 8086 -o llama3.1_405b_tp8_fp16_prefill.tracy -s 35
```

## 8. Build/install Tracy Profile Viewer

You can install [Windows Tracy profile viewer](https://github.com/wolfpld/tracy/releases/download/v0.11.1/windows-0.11.1.zip) on your laptop or try building the tracy-profiler on Ubuntu 22.04 and use that. For Ubuntu 22.04, you will need to make sure your machine has right packages by doing following:
```
sudo apt update
sudo apt install libdbus-1-dev libegl1-mesa-dev libxkbcommon-dev wayland-protocols libwayland-egl1-mesa libwayland-dev
```
Then follow instructions on [build tracy profiler](https://iree.dev/developers/performance/profiling-with-tracy/#building-tracy-from-source)
Following worked for my setup with ~/iree is where iree source code is checked out for me:
```
cd ~/iree/third_party/tracy
cmake -B profiler/build -S profiler -DCMAKE_BUILD_TYPE=Release
cmake --build profiler/build --parallel --config Release
```
## 9. View tracy file

You can view trace using:
```
~/iree/third-party/trcay/profiler/build/tracy-profiler .\llama3.1_405b_tp8_fp16_prefill.tracy
```
Or say if you installed the windows version on your laptop, you can scp your .tracy from MI300X server and view as below assuming you uninstalled the windows release of tracy at c:\iree-tracy. An example command to follow is below:
```
scp 172.19.26.15:/home/kudeepak/llama3.1/405B/llama3.1_405b_tp8_fp16_prefill.tracy .
C:\iree-tracy\tracy-profiler.exe .\llama3.1_405b_tp8_fp16_prefill.tracy
```

