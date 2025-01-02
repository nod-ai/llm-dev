If there is a Segmentation fault, OOM, or core dumped error from the IREE runtime, these are some tools that
can be helpful in narrowing down the issue.

# GDB
Run your command prepended with `gdb --args`:
```
gdb --args iree-benchmark-module ...
```

# Build IREE with ASAN (Address Sanitizer)
Build IREE with `-DIREE_ASAN_BUILD=ON`.

# Use VM execution tracing
The `--trace_execution` flag to runtime tools like `iree-run-module` will print each VM instruction as it is
executed. This can help with associating other logs and system behavior with the compiled VM program.
