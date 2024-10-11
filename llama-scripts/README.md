# Llama Performance Benchmarks
## Prerequisites
1. Llama irpa file
2. Exported Llama IR from SHARK-Platform `export_paged_llm_v1.py` using `--irpa-file`
3. Llama prefill and decode numpy files with actual inputs
4. export PATH=/path/to/iree/build/tools:$PATH

## Compiling Llama
```
./compile-llama.sh gfx942 /path/to/llama.mlir
```

## Benchmarking Llama Prefill
```
./benchmark-llama-prefill.sh 0 /path/to/llama.irpa /path/to/llama_prefill_args
```

## Benchmarking Llama Decode
```
./benchmark-llama-decode.sh 0 /path/to/llama.irpa /path/to/llama_decode_args
```
