If there is a Segmentation fault, OOM, or core dumped error from the IREE runtime, these are some tools that
can be helpful in narrowing down the issue.

# GDB
Build IREE in `RelWithDebInfo` or `Debug` mode.

Run your command prepended with `gdb --args`:
```
gdb --args ../iree-build-trace/tools/iree-run-module ...
```

When GDB starts, enter the commands below to load symbol table from iree-run-module, set a breakpoint, and run the debugger:
```
Reading symbols from ../iree-build-trace/tools/iree-run-module...
(gdb) file ../iree-build-trace/tools/iree-run-module
Load new symbol table from "../iree-build-trace/tools/iree-run-module"? (y or n) y
Reading symbols from ../iree-build-trace/tools/iree-run-module...
(gdb) break runtime/src/iree/hal/drivers/hip/event_semaphore.c:673
(gdb) r
```


# Build IREE with ASAN (Address Sanitizer)
Build IREE with `-DIREE_ASAN_BUILD=ON`.

Compile and run module to see logs.

# Use VM execution tracing
The `--trace_execution` flag to runtime tools like `iree-run-module` will print each VM instruction as it is
executed. This can help with associating other logs and system behavior with the compiled VM program.
