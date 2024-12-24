### Model status

|Model|IR generation|compilation|runtime|shortfin serving|
|---|---|---|---|---|       
|8B-FP16(unsharded)|PASS|PASS|PASS|NTD
|70B-FP16(unsharded)|PASS|PASS|PASS|NTD
|405B-FP16(sharded, prefill)|PASS|PASS|PASS|NTD




### issue with models

### Unsharded

|Model| IR generation |Compilation|runtime|comment|
|---|---|---|---|---|                                         
|405B-FP16-Prefill|PASS|PASS|FAIL|RESOURCE_EXHAUSTED; HIP driver error 'hipErrorOutOfMemory' (2): out of memory|
|405B-FP16-Decode|PASS|PASS|FAIL|RESOURCE_EXHAUSTED; HIP driver error 'hipErrorOutOfMemory' (2): out of memory|






### Sharded


|Model|IR generation|Compilation|runtime|Comment|
|---|---|---|---|---|                                         
|8B-Prefill|PASS|FAIL|Memory access fault by GPU node-4 (Agent handle: 0x58470a300960) on address 0x7182ec58b000. Reason: Unknown|
|8B-Decode|PASS|FAIL|:0:rocdevice.cpp            :2984: 2787027630305 us: [pid:688936 tid:0x7dfc4e600640] Callback: Queue 0x7dfbe0300000 aborting with error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address. code: 0x29
|70B-Prefill|PASS|PASS|
|70B-Decode|PASS|PASS|
|405B-Prefill|PASS|PASS|FAIL|
|405B-Decode|PASS|FAIL|


