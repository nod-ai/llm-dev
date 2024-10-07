
# Introduction
This file provides status of readiness of halo models like llama3, grok1 etc. on MI3xx 

# Glossary
TPn: Tensor Parallel using n GPUs

# Caveats
- Do not use CPX mode for MI300 as memory is 1/8th and you will run out of resources

# Current status

|Models | compile | inference (SPX mode) | tracy |
|---|---|---|---|
|llama3-8b-FP16| PASS | prefill (1746 ms), decode (71.8 ms), [commands](https://gist.github.com/aviator19941/f10b5b7a7c3975de4363450b4d7ec68f) | [prefill](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama8b_f16_prefill.tracy) [decode](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama8b_f16_decode.tracy) |
|llama3-8b-Q4_1| PASS | prefill (1817 ms), decode (57.3 ms), [commands](https://gist.github.com/aviator19941/f10b5b7a7c3975de4363450b4d7ec68f) | [prefill](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama8b_q4_1_prefill_v2.tracy) [decode](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama8b_q4_1_decode_v2.tracy) |
|llama3-8b-Q4_k| PASS | | |
|llama3-70b-Q4_1| PASS | prefill (3543 ms), decode (213 ms), [commands](https://gist.github.com/aviator19941/79ee5afc39c225ec7469030320014fa3) | [prefill](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama70b_q4_1_prefill.tracy) [decode](https://sharkpublic.blob.core.windows.net/sharkpublic/avi/llama70b_q4_1_decode.tracy) |
|llama2-7b-FP8| [FAIL](https://github.com/iree-org/iree/issues/18367)| | |
|grok-1-Q4_1| PASS | FAIL, out of memory | [prefill](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1-rocm-prefill.tracy) [decode](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1-rocm-decode.tracy) |

# Tasks and Issues
task      | owner      | status | next actions
:-------: | :--------: |:-------: | :------:
Sharded LLaMa | boian | In progress | Landing first sharded tests
Export/Compile LLaMa | kyle | blocked on `torch.aten.complex` | rob is authoring fix
LLaMa 8 prefill comparison | rob | layerwise comparison for prefill is normal | handing off tooling to Avi
LLaMa 8 decode comparison | avi | still investigating cause of numeric issue | reuse rob's tooling to investigate
FP8 quantized model | dan | finishing results from quark | following up with Giuseppe on new `fp8 quantization
Model evaluation tooling | archana | working on perplexity script | update on progress / blockers

# Goals

- [ ] Attention Compiler Work
  - [ ] Dynamic sequence length
  - [ ] Causal Masking
  - [ ] Flex attention compilation
- [ ] LLaMa 8b prefill and decode
  - [x] validated numerically correct 
  - [ ] export
  - [ ] compiled
  - [ ] benchmarked
  - [ ] replicate for larger variants
- [ ] Mixtral prefill and decode
  - [ ] validated numerically correct 
  - [ ] export
  - [ ] compiled
  - [ ] benchmarked
- [ ] Grok prefill and decode
  - [x] validated numerically correct 
  - [x] export
  - [x] compiled
  - [ ] benchmarked


# Artifacts

## Guideline:
1) small files and MLIR files check into [llm-dev](https://github.com/nod-ai/llm-dev)
2) large files upload to [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) -> "halo-models" container on Azure and put link to that in the table(s) below
3) Very large files, store on GPU server and note the name/location of/on the machine in table(s) below 

Note: If a link to Azure sharkblob below gives you an error, either use az cli to download (see section Accessing sharkblobs on Azure) or click on [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) , then click on "Blob containers" and then navigate to the file manually and download it. 

## TP1
Models           |     FP16        |   FP8           |     Q4_1         |    Q4_K       |    Attention IRs
:--------------: | :-------------: |:----------------:|:---------------:|:-------------:|:------------------:
llama2-7b | | [irpa](https://sharkblobs.blob.core.windows.net/dan/qdq_full_transpose.irpa) [mlir](https://sharkblobs.blob.core.windows.net/dan/batch_llama_v1.mlir) | | | [Attention IRs](https://github.com/nod-ai/llm-dev/tree/main/models/llama_attention_irs)
llama3-8b | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_f16.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_f16.gguf) | | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_q4_1.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_q4_1.gguf) | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_q4_k.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_8b/llama8b_Q4_K.gguf) |
llama3-70b | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_f16.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_f16.gguf) | | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_q4_1.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_q4_1.gguf) | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_q4_k.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_70b/llama70b_q4_k.gguf) |
llama3-405b | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_fp16.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_fp16.gguf) | | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_q4_1.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_q4_1.gguf) | [mlir](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_q4_k.mlir) [gguf](https://sharkblobs.blob.core.windows.net/halo-models/llm-dev/llama3_405b/llama405b_q4_k.gguf) |
grok-1 | [mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/dan/grok.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/llm-dev/grok_1/grok-1-f16.gguf) |NA | [mlir](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1-irpa.mlir) [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_1.gguf) | [gguf](https://sharkpublic.blob.core.windows.net/sharkpublic/halo-models/grok-1/grok-1-q4_k.gguf) |

## TP2
Models           |     FP16        |   FP8           |     Q4_1     |  Q4_K
:--------------: | :-------------: |:----------------:|:----------------: | :----------------:
llama3-8b | | |
llama3-70b | | |
llama3-405b |NA |NA |
grok-1 |NA | |


## TP4
Models           |     FP16        |   FP8           |     Q4_1   |  Q4_K
:--------------: | :-------------: |:----------------:|:----------------:| :----------------:
llama3-8b | | |
llama3-70b | | |
llama3-405b |NA | |
grok-1 | | |

## TP8
Models           |     FP16        |   FP8           |     Q4_1 |  Q4_K
:--------------: | :-------------: |:----------------:|:----------------:| :----------------:
llama3-8b | | | 
llama3-70b | | |
llama3-405b | | |
grok-1 | | |

## MLIR generation and Compilation
[Quantization](https://github.com/nod-ai/llm-dev/blob/main/Quantization.md)
```
iree-compile --iree-hal-target-backends=rocm --iree-hip-target=gfx942 <mlir file> -o <vmfb file>
```

## Accessing sharkblobs on Azure:
In browser, click on [sharkblobs](https://portal.azure.com/#@amdcloud.onmicrosoft.com/resource/subscriptions/8c190d1b-eb91-48d5-bec5-3e7cb7412e6c/resourceGroups/pdue-nod-ai-rg/providers/Microsoft.Storage/storageAccounts/sharkblobs/storagebrowser) , then click on "Blob-containers" and the click on "halo-models"

Or, use command line by first installing az cli as:
```
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```
And then, get the account key for the storage account by clicking on "Storage Accounts" in Azure Services or searching "sharkblobs" in the top search bar. Then, click on sharkblobs. Then, on the left side bar, under Security + networking, click on "Access keys". Copy the account key from here and use in the following command
To upload:
```
az storage blob upload --account-name sharkblobs --container-name halo-models --name <azure path, example: llm-dev/llama3_8b/llama8b_f16.mlir> --file <local_path_on_computer> --account-key <key_retrieved_from_directions_above>
```

To download:
```
az storage blob download --account-name sharkblobs --container-name halo-models --name <azure path, example: llm-dev/llama3_8b/llama8b_f16.mlir> --file <local_path_on_computer> --account-key <key_retrieved_from_directions_above>
```

if you are downloading from "sharkpublic" then replace instructions above by sharkpublic and get your account access key for sharkpublic.
Example:
```
az storage blob download --account-name sharkpublic --container-name sharkpublic --name ian/llama8b_f16.gguf --file llama8b_f16.gguf --account-key <key string>
```

# AMD GPU Machines
[MI250-300](https://github.com/nod-ai/playbook/blob/main/HOWTO/access-mi250-mi300.md)
