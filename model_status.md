### Model status

|Model|IR generation|compilation|runtime|shortfin serving|Kubernetes cluster|
|---|---|---|---|---|---|       
|8B-FP16(unsharded)|PASS|PASS|PASS|PASS|NTD
|70B-FP16(unsharded)|PASS|PASS|PASS|NTD|NTD
|405B-FP16(sharded)|Prefill-PASS|Prefill-PASS|Prefill-PASS|NTD|NTD
|8B-Instruct-FP16(unsharded)|NTD|NTD|NTD|NTD|NTD
|70B-Instruct-FP16unsharded)|NTD|NTD|NTD|NTD|NTD
|405B-Instruct-FP16(sharded)|NTD|NTD|NTD|NTD|NTD


### issue with models

### Unsharded

|Model| IR generation |Compilation|runtime|comment|
|---|---|---|---|---|                                         
|405B-FP16-Prefill|PASS|PASS|FAIL|RESOURCE_EXHAUSTED; HIP driver error 'hipErrorOutOfMemory' (2): out of memory|
|405B-FP16-Decode|PASS|PASS|FAIL|RESOURCE_EXHAUSTED; HIP driver error 'hipErrorOutOfMemory' (2): out of memory|



### Sharded


|Model|IR generation|Compilation|runtime|Comment|
|---|---|---|---|---|                                         
|8B-Prefill|PASS|FAIL|FAIL|Memory access fault by GPU node-4 (Agent handle: 0x58470a300960) on address 0x7182ec58b000. Reason: Unknown|
|8B-Decode|PASS|FAIL|FAIL|:0:rocdevice.cpp            :2984: 2787027630305 us: [pid:688936 tid:0x7dfc4e600640] Callback: Queue 0x7dfbe0300000 aborting with error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address. code: 0x29
|405B-Decode|PASS|PASS|FAIL| Seems input is not correct. INVALID_ARGUMENT; function expected fewer input values; parsing input `@/data/llama3.1/weights/405b/decode_args_bs4_128_stride_32/cs_f16_shard_7.npy


