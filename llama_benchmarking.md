# How to benchmark Llama 3.1
In order to benchmark Llama 3.1 prefill and decode, you will need these artifacts for sharded (Tensor Parallel (TP)=8) or unsharded (TP=1):

1. irpa file(s)
2. IR
3. numpy inputs

## 1. Get the irpa files
Create a SAS token in Azure:
- Go to the [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/overview) storage account in the Azure portal
- In the `Security + networking` dropdown, click `Shared access signature`
- Under `Allowed resource types` select Service, Container, and Object
- Scroll down to the bottom and select `Generate SAS and connection string`
- Scroll down and Copy the SAS token

Unsharded:

```
azcopy copy 'https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/8b_f16.irpa?[Add SAS token here]' '8b_f16.irpa'
```

If you have trouble accessing `sharkblobs`, you can copy the 8b f16 unsharded irpa file from the `SharkMi300x` machine:
```
scp nod@10.23.233.219:/data/llama3.1/weights/8b/fp16/llama3.1_8b_fp16.irpa 8b_f16.irpa
```

Sharded - If you want to create your own tp8 sharded irpa files use this command:
```
python3 -m sharktank.examples.sharding.shard_llm_dataset --irpa-file 8b_fp16.irpa --output-irpa 8b_fp16_tp8.irpa --tensor-parallelism-size 8
```

Larger sharded irpa files (e.g. 70b, 405b) will be stored in `sharkblobs` soon. Otherwise, you can copy the 70b/405b f16 sharded irpa files from the `SharkMi300x` machine (long copy time):
```
scp nod@10.23.233.219:/data/llama3.1/weights/405b/fp16/tp8/* .
```


## 2. Generate the IR
a. Clone `shark-ai`:
```
git clone https://github.com/nod-ai/shark-ai.git
```

b. Set up env:
https://github.com/nod-ai/shark-ai/blob/main/docs/developer_guide.md#setup-a-venv

c. Generate the IR for prefill only (remove `--skip-decode` flag for both prefill + decode).

Unsharded:

```
python3 -m sharktank.examples.export_paged_llm_v1 --bs=4 --irpa-file=8b_f16.irpa --output-mlir=8b_f16_prefill_nondecomposed.mlir --output-config=8b_f16_prefill_nondecomposed.json --attention-kernel=torch --skip-decode
```

Sharded - You need to use the unranked sharded irpa file to generate the sharded IR:

```
python3 -m sharktank.examples.export_paged_llm_v1 --bs=4 --irpa-file=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.irpa --output-mlir=405b_f16_prefill_tp8_nondecomposed.mlir --output-config=405b_f16_prefill_tp8_nondecomposed.json --attention-kernel=torch --skip-decode
```

## 3. Get the numpy inputs

Get the 8b f16 tp1 unsharded numpy inputs: [get_8b_f16_tp1_numpy_inputs.sh](https://gist.github.com/aviator19941/380acabc77aeb4749fac14262e17db69)

Get the 8b f16 tp8 sharded numpy inputs: [get_8b_f16_tp8_numpy_inputs.sh](https://gist.github.com/aviator19941/9b3cd6511347e57671b7ff1da7c80bfa)

## 4. Compile command
Unsharded:

`
../iree-build-no-trace/tools/iree-compile 8b_f16_prefill_nondecomposed.mlir --iree-hip-target=gfx942 -o=prefill_8b.vmfb --iree-hal-target-device=hip --iree-dispatch-creation-enable-aggressive-fusion=true --iree-global-opt-propagate-transposes=true --iree-opt-aggressively-propagate-transposes=true --iree-opt-data-tiling=false --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' --iree-hal-indirect-command-buffers=true --iree-stream-resource-memory-model=discrete --iree-hip-legacy-sync=false --iree-hal-memoization=true --iree-opt-strip-assertions
`

Sharded:

`
../iree-build-no-trace/tools/iree-compile 405b_f16_prefill_tp8_nondecomposed.mlir --iree-hip-target=gfx942 -o=prefill_405b_tp8.vmfb --iree-hal-target-device=hip[0] --iree-hal-target-device=hip[1] --iree-hal-target-device=hip[2] --iree-hal-target-device=hip[3] --iree-hal-target-device=hip[4] --iree-hal-target-device=hip[5] --iree-hal-target-device=hip[6] --iree-hal-target-device=hip[7] --iree-dispatch-creation-enable-aggressive-fusion=true --iree-global-opt-propagate-transposes=true --iree-opt-aggressively-propagate-transposes=true --iree-opt-data-tiling=false --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))' --iree-hal-indirect-command-buffers=true --iree-stream-resource-memory-model=discrete --iree-hip-legacy-sync=false --iree-hal-memoization=true --iree-opt-strip-assertions
`

## 5. Benchmark command
Unsharded:

`
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ../iree-build-no-trace/tools/iree-benchmark-module \
  --hip_use_streams=true \
  --device_allocator=caching \
  --module=prefill_8b.vmfb \
  --parameters=model=8b_fp16.irpa \
  --device=hip://4 \
  --function=prefill_bs4 \
  --input=@/data/llama-3.1/weights/8b/prefill_args_bs4_128/random_tokens.npy \
  --input=@/data/llama-3.1/weights/8b/prefill_args_bs4_128/seq_lens.npy \
  --input=@/data/llama-3.1/weights/8b/prefill_args_bs4_128/seq_block_ids.npy \
  --input=@/data/llama-3.1/weights/8b/prefill_args_bs4_128/cs_f16.npy \
  --benchmark_repetitions=3
`

Sharded:

`
ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ../iree-build-no-trace/tools/iree-run-module --hip_use_streams=true --device_allocator=caching --module=prefill_405b_tp8.vmfb --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.irpa --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.rank0.irpa --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.rank1.irpa --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.rank2.irpa --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.rank3.irpa --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.rank4.irpa --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.rank5.irpa --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.rank6.irpa --parameters=model=/shark-dev/405b/llama3.1_405b_fp16_tp8_parameters.rank7.irpa --device=hip://0 --device=hip://1 --device=hip://2 --device=hip://3 --device=hip://4 --device=hip://5 --device=hip://6 --device=hip://7 --function=prefill_bs4 --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/random_tokens.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/seq_lens.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/seq_block_ids.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/cs_f16_shard_0.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/cs_f16_shard_1.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/cs_f16_shard_2.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/cs_f16_shard_3.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/cs_f16_shard_4.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/cs_f16_shard_5.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/cs_f16_shard_6.npy --input=@/data/llama3.1/weights/405b/prefill_args_bs4_128/cs_f16_shard_7.npy
`
