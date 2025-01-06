### Model status

#### Token length: 2048

|Model|sharktank|iree-compile|iree-run-module|iree-benchmark-module|shortfin-sglang|kubernetes|
|---|---|---|---|---|---|---|       
|8B-FP16-TP1|PASS|PASS|PASS|PASS|PASS|NTD|NTD
|70B-FP16-TP1|PASS|PASS|PASS|FAIL(prefill) [19569](https://github.com/iree-org/iree/issues/19569)|NTD|NTD
|70B-FP16-TP8-decode|PASS|PASS|PASS|FAIL [19574](https://github.com/iree-org/iree/issues/19574)|NTD|NTD
|70B-FP16-TP8-prefill|PASS|PASS|PASS|PASS|NTD|NTD
|405B-FP16-TP8-Prefill|PASS|PASS|PASS|PASS|NTD|NTD
|405B-FP16-TP8-decode|PASS|PASS|FAIL [19574](https://github.com/iree-org/iree/issues/19574)|FAIL[19574](https://github.com/iree-org/iree/issues/19574)|NTD|NTD|
|8B-Instruct-FP16-TP1|PASS|PASS|PASS|PASS|PASS|NTD
|70B-Instruct-FP16-TP1|PASS|PASS|PASS|FAIL(prefill) [19569](https://github.com/iree-org/iree/issues/19569)|NTD|NTD
|405B-Instruct-FP16-TP8-prefill|PASS|PASS|PASS|PASS|NTD|NTD
|405B-Instruct-FP16-TP8-decode|

### Token length: 128

|Model|sharktank|iree-compile|iree-run-module|iree-benchmark-module|shortfin-sglang|kubernetes|
|---|---|---|---|---|---|---|       
|8B-FP16-TP1|PASS|PASS|PASS|PASS|PASS|NTD|NTD
|70B-FP16-TP1|PASS|PASS|PASS|PASS|NTD|NTD
|70B-FP16-TP8-decode|PASS|PASS|PASS|PASS|NTD|NTD
|70B-FP16-TP8-prefill|PASS|PASS|PASS|PASS|NTD|NTD
|405B-FP16-TP8-Prefill|PASS|PASS|PASS|PASS|NTD|NTD
|405B-FP16-TP8-decode|PASS|PASS|PASS|FAIL [19574](https://github.com/iree-org/iree/issues/19574)|NTD|NTD|
|8B-Instruct-FP16-TP1|PASS|PASS|PASS|PASS|PASS|NTD
|70B-Instruct-FP16-TP1|PASS|PASS|PASS|PASS|NTD|NTD
|405B-Instruct-FP16-TP8-prefill|PASS|PASS|PASS|PASS|NTD|NTD
|405B-Instruct-FP16-TP8-decode|




N.B. The weight file for 70B-Instruct was generated using `llama.cpp/convert_hf_to_gguf.py` through the following command:
```sh
 python3 convert_hf_to_gguf.py <path_to_hf_safetensor_files> --outtype f16 --outfile llama_70b_3.1_instruct.gguf
```

### Performance 

|Model|Export time(sec)| iree-compile time(sec)
|---|---|---|
|8B-FP16-prefill-unsharded|145|18|
|8B-FP16-prefill-sharded|954|170|
|8B-FP16-decode-unsharded|203|32|
|8B-FP16-decode-sharded|1684|370|
|8B-inst-FP16-prefill-unsharded|140|19|
|8B-inst-FP16-prefill-sharded|940|171|
|8B-inst-FP16-decode-unsharded|193|31|
|8B-inst-FP16-decode-sharded|1674|393|
|70B-FP16-prefill-unsharded|269|40|
|70B-FP16-prefill-sharded|2475|722|
|70B-FP16-decode-unsharded|413|66|
|70B-FP16-decode-sharded|4797|1642|
|70B-inst-FP16-prefill-unsharded|161|40|
|70B-inst-FP16-prefill-sharded|2505|706|
|70B-inst-FP16-decode-unsharded|416|66|
|70B-inst-FP16-decode-sharded|4784|1670|
|405B-FP16-prefill-sharded|4711|1691|
|405B-FP16-decode-sharded|5648|3600|
|405B-inst-FP16-prefill-sharded|4745|1594|
|405B-inst-FP16-decode-sharded|5798|4003|



