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



