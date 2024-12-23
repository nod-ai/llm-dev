### Model status

### Unsharded

|Model|Compilation|runtime|comment|
|---|---|---|---|                                         
|8B-Prefill|PASS|PASS|
|8B-Decode|PASS|PASS|
|70B-Prefill|PASS|PASS|
|70B-Decode|PASS|PASS|
|405B-Prefill|PASS|FAIL|RESOURCE_EXHAUSTED; HIP driver error 'hipErrorOutOfMemory' (2): out of memory|
|405B-Decode|PASS|FAIL|RESOURCE_EXHAUSTED; HIP driver error 'hipErrorOutOfMemory' (2): out of memory|






### Sharded


|Model|Compilation|runtime|Comment|
|---|---|---|---|                                         
|8B-Prefill|PASS|FAIL|Memory access fault by GPU node-4 (Agent handle: 0x58470a300960) on address 0x7182ec58b000. Reason: Unknown|
|8B-Decode|PASS|FAIL|:0:rocdevice.cpp            :2984: 2787027630305 us: [pid:688936 tid:0x7dfc4e600640] Callback: Queue 0x7dfbe0300000 aborting with error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address. code: 0x29
|70B-Prefill|PASS|PASS|
|70B-Decode|PASS|PASS|
|405B-Prefill|PASS|FAIL|
|405B-Decode|PASS|FAIL|


