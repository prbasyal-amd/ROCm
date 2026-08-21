# ROCm consolidated changelog

This page is a historical overview of changes made to ROCm components. This
consolidated changelog documents key modifications and improvements across
different versions of the ROCm software stack and its components.

## ROCm 10.0.0

See the [ROCm 10.0.0 release notes](https://rocm-stg.amd.com/en/latest/about/release-notes.html#rocm-core-sdk-10-0-0-release-notes)
for a complete overview of this release.


#### **AMD SMI** (27.0.0)

##### Changed

- Bumped the library major version to 27.0.0 (breaking).
  - The shared library SONAME is now `libamd_smi.so.27`. Consumers linked against `libamd_smi.so.26` must relink; no source changes are required beyond the API changes listed elsewhere in this release.

- Restructured AMD SMI C++ tests into unit and functional suites.
  - The `amdsmitst` source tree now separates unit tests from hardware-backed functional tests under `tests/amd_smi_test/unit/` and `tests/amd_smi_test/functional/`.
  - GTest suite names now follow a `<Component><Type>[<Operation>]` scheme: functional tests are `<Component>FunctionalReadOnly`/`<Component>FunctionalReadWrite` (e.g. `GpuFunctionalReadOnly`) and unit tests are `<Component>Unit` (e.g. `GpuUnit`). This replaces the old `amdsmitstReadOnly`/`amdsmitstReadWrite` and `AmdSmiDynamicMetricTest` names.
  - Consumers that pass explicit `--gtest_filter` values should update those filters to the new suite names.
  - See the [AMD SMI test design](https://rocm.docs.amd.com/projects/amdsmi/en/docs-10.0.0/conceptual/test-design.html#naming-conventions) for the suite naming convention and `--gtest_filter` usage.

##### Optimized

- Optimized `amdsmi_get_gpu_process_list()` to skip redundant KFD topology discovery.
  - The per-process KFD lookup rebuilt the entire KFD node topology (an expensive sysfs walk) on every call just to translate the device BDF into its KFD GPU id.
  - The caller already knows this value, so it is now passed through to `gpuvsmi_get_pid_info()`, eliminating one full topology discovery per process per refresh. Falls back to the original discovery path when the id is unavailable.

##### Resolved issues

- Fixed `amd-smi ras --cper --json` emitting nothing when there are no CPER entries.
  - The common no-entries case printed empty output, so consumers feeding stdout to `json.loads` failed with `Expecting value: line 1 column 1 (char 0)`. The command now always emits exactly one valid JSON document: `[]` when there are no entries, or a single aggregated array across all GPUs when there are. `--follow` mode stays silent until entries appear. The human-readable primary-partition warning is also suppressed in JSON mode so it no longer corrupts the output.

- Fixed `amd-smi set --ptl-status` silently failing to change PTL state.
  - The set path wrote `"1"`/`"0"` to the `ptl/ptl_enable` sysfs node, which only accepts `"enabled"`/`"disabled"`; the driver ignored the numeric write while the API still reported success. The state now changes as expected, and a rejected write returns a real error instead of a generic success.

- Fixed `amd-smi process` hiding compute processes owned by other users.
  - A caller without permission to read another process's `/proc/<pid>/fd` was misdetected as running in a separate PID namespace, which caused the whole compute-process list to come back empty. Such processes are now listed with a redacted (`N/A`) name instead of being dropped.

- Fixed CU%/SDMA column alignment in the `amd-smi` process table.
  - The `SDMA` header no longer sits a column left of its values, and valid `CU %`/`SDMA` values are no longer truncated.

- Fixed compute processes being reported on every GPU.
  - A process was attributed to a GPU whenever it had a KFD context on that GPU, so a job with queues on a single GPU appeared under every GPU. Attribution now uses the process's active KFD queues plus any GPU where it holds a non-zero VRAM allocation, so a process is listed only against the GPUs it actually uses.

- Fixed `amd-smi` hanging in `amdsmi_init()` on UALink systems when the IFoE driver is unresponsive.
  - `amdsmi_init()` (and every CLI command) opened a per-GPU IFoE/UALoE fabric session up front, so it blocked indefinitely when the Broadcom IFoE driver was unresponsive, even for queries that never use fabric data.
  - The fabric session is now opened only on the first fabric query, so initialization and non-fabric queries no longer touch the IFoE driver.

- Fixed ctypes `DeprecationWarning` from `amdsmi_wrapper.py` on Python 3.14.
  - Python 3.14 deprecates the implicit ctypes structure layout when `_pack_` is set (slated to become an error in 3.19). Each packed structure/union in the generated wrapper now sets `_layout_ = 'ms'`, preserving the existing MSVC-compatible layout (no ABI change) while silencing the warning.

#### **HIP** (10.0.0)

##### Added
* New HIP APIs
    - Stream Ordered Memory Allocator: Support for the following APIs for parity with corresponding CUDA APIs.
      * `hipMemGetDefaultMemPool` returns the default memory pool for the specified location and allocation type.

##### Optimized

* Improved `hipMemcpy2D()` and `hipMemcpy2DAsync()` performance for copy operations with very small row widths and large row counts.
Previously, non-4-byte-aligned row or slice pitches could cause the runtime to issue a separate copy for each row, resulting in significant
performance degradation for workloads such as 1-byte-wide transfers with millions of rows.
These transfers are now handled using a single shader-based copy operation instead of a separate copy per row, significantly reducing transfer times.
Copy operations at or below the 256-row threshold are unchanged.

##### Resolved issues

* Resolved library loading error messages thrown by `rocminfo` during driver initialization in WSL (Windows Subsystem for Linux) environment due to failure in loading the HSA runtime library `libhsa-runtime64.so`,
since it is not available in the dynamic linker search path. Since `rocminfo` already links against `libhsa-runtime64.so`, the runtime now correctly locates and loads the HSA runtime library using the `RTLD_NOLOAD` option,
enabling successful ROCm initialization, HSA agent discovery, and subsequent ROCm operations.
* Fixed a segmentation fault in HIP queue idle detection caused by referencing a recycled completion signal. Idle state is now derived from a queue-owned signal with a safe lifetime.
* Resolved incorrect NaN handling in the ordered not-equal comparison intrinsics `__hne` (for `__half`) and `__hne` (for `__hip_bfloat16`), along with their vector forms. Being *ordered* predicates, they now correctly return `false` when either operand is NaN.

#### **hipBLAS** (3.6.0)

##### Added

* Per-batch `alpha`/`beta` support for Level 2 batched and strided-batched forms of `symv`, `hemv`, `sbmv` and `spmv` via `hipblasSetBatchAlphaStride` and/or `hipblasSetBatchBetaStride` (device pointer mode).
* Per-batch `alpha` support for Level 2 batched and strided-batched forms of `syr` via `hipblasSetBatchAlphaStride` (device pointer mode).
* Per-batch `alpha` (scalar vector) API support for Level 1 batched and strided-batched forms of `scal` and the `_ex` forms through `hipblasSetBatchAlphaStride` when `hipblasHandle_t` is in mode `HIPBLAS_POINTER_MODE_DEVICE`.

##### Resolved issues

* PyTorch users can avoid user-constraint-based memory allocation failures (`HIPBLAS_STATUS_ALLOC_FAILED`) by exporting `HIPBLAS_WORKSPACE_CONFIG=:0:0` to allow rocBLAS managed memory to grow automatically.

#### **hipCUB** (5.0.0)

##### Added

* Feature parity with CCCL/CUB 3.0.0.
* `::hip::std` support.

##### Changed

* Changed `CCCL_MINIMUM_VERSION` to `3.0.0` to align with CUB.
* Add support for large num_items `DeviceMerge` and `DeviceSegmentedSort`.
* Replace `#pragma unroll` by `_CCCL_PRAGMA_UNROLL_FULL()` and `_CCCL_PRAGMA_NOUNROLL()` by `_CCCL_PRAGMA_NOUNROLL()`.
* Add `_CCCL_SORT_MAYBE_UNROLL()` in block merge sort and thread sort.
* Update `WarpExchange` template parameters for CUB compatibility.

##### Removed

* hipCUB compatibility with PyTorch v2.9 and v2.10 has been removed in this release. Use PyTorch v2.11 or later.
* Removed `hipcub::BaseTraits::CATEGORY`, `hipcub::BaseTraits::nullptr_TYPE` and `hipcub::BaseTraits::PRIMITIVE`.
* Removed `ConstantInputIterator`, `CountingInputIterator`, `DiscardOutputIterator` and `TransformInputIterator`, which were deprecated in hipCUB-4.1.0.
* Removed `DeviceSpmv`, which was removed from CUB after CCCL's 2.8.0 release. Use `hipSPARSE` or `rocSPARSE` libraries instead.
* Removed `GridBarrier`.
* Removed `HIPCUB_MIN`, `HIPCUB_MAX`, `HIPCUB_QUOTIENT_FLOOR`, `HIPCUB_QUOTIENT_CEILING`, `HIPCUB_ROUND_UP_NEAREST` and `HIPCUB_ROUND_DOWN_NEAREST` which were deprecated in hipCUB-4.1.0.
* Removed `LEGACY_PTX_ARCH`.
* Removed `hipcub:max` and `hipcub:min`, which were deprecated. Use `hip::std::max` and `hip::std::min` instead.
* Deprecated `hipcub::Swap`, use `rocprim::swap` instead.
* Deprecated `HIPCUB_IS_INT128_ENABLED`, use `_CCCL_HAS_INT128()` instead.
* Deprecated `hipcub::Equality`, `hipcub::Inequality`, `hipcub::InequalityWrapper`, `hipcub::Sum`, `hipcub::Difference`, `hipcub::Division`, `hipcub::Max` and `hipcub::Min` operators. Use `hip::std::equal_to`, `hip::std::not_equal_to`, `hip::std::plus`, `hip::std::minus`, `hip::std::divides`, `hip::maximum`, and `hip::minimum` operators instead.

#### **hipFFT** (1.0.25)

##### Changed

* Minor internal changes.

#### **hipFile** (0.4.0)

##### Added

* A KFD-based alternative check for P2P DMA support was added to `ais-check`. This inspects the `capability` property under `/sys/class/kfd/kfd/topology/nodes/*/properties`.
* Guides for setting up storage targets to the documentation.

##### Changed

* `ais-check` now lists the AIS-capable file system mounts detected on the system and fails if none are found.
* Fastpath-only tests are now automatically skipped on systems that do not support the AIS fastpath instead of failing. Running ctest in verbose mode (`ctest -V`) will provide the reason the test was skipped.
* Updated INSTALL.md to point to official install docs.

#### **hipSOLVER** (3.6.0)

##### Changed

* Minor internal changes.

#### **hipSPARSE** (4.7.0)

##### Added
* Blocked ELL format support to the `hipsparseDenseToSparse` routine, along with the new `hipsparseBlockedEllSetPointers` function.
* The `HIPSPARSE_SPMV_CSR_ALG3` algorithm to `hipsparseSpMV`, which exposes the rocSPARSE CSR nnz split algorithm (`rocsparse_spmv_alg_csr_nnzsplit`).
* CSC format support to `hipsparseSpSV` and `hipsparseSpSM`.

##### Resolved issues
* Fixed an issue with `hipsparseSpMM`, which produced incorrect results for the Blocked ELL sparse format.

#### **RCCL** (2.30.7)

##### Added
* Compatibility with NCCL 2.30.7.
* Scalable AllGatherV pattern: grouped `ncclBroadcast` calls with distinct roots are fused into a single ring kernel, improving performance at large scale. Gated by `NCCL_ALLGATHERV_ENABLE` (default off).
* GPU-only multi-segment registration for symmetric memory windows, enabling contiguous VA ranges backed by multiple physical segments (single-node validated).
* Elastic Buffer support for symmetric windows spanning device and host/`HOST_NUMA` memory segments (`NCCL_ELASTIC_BUFFER_REGISTER`, `NCCL_SYM_REUSE_SYSMEM_HANDLES`). Single-node path validated; multi-node registration remains limited pending HIP/HSA multi-segment DMA-BUF export support.

##### Changed
* Adapted the device-initiated GIN backends (Anvil SDMA and rocSHMEM GDA) to the NCCL 2.30.7 GIN API v14:
  * Added the new `getGinProperties` host op.
  * Dropped the data-path ops (`iput`/`iputSignal`/`iget`/`iflush`/`test`) that moved out of GIN under the GIN/RMA split.
  * Switched `createContext` to `ncclGinConfig_v14_t`.
  * Updated the device dispatch signatures, and matched the GIN type renumbering (`ROCSHMEM_GDA` and `ANVIL_SDMA` shifted after the new `GIN_GPI` type).
  * The plugins now use the generic (unversioned) `ncclGin_t` / `ncclGinConfig_t` / `ncclGinProperties_t` typedefs so future ABI bumps do not require touching call sites.
* Updated the ROCSHMEM GIN plugin registration to the v14 layout (corrected struct field names and the conditional that previously only compiled without ROCSHMEM GIN).
* Adapted the InfiniBand transports (`net_ib` and `net_ib_cast`) to the v14 GIN/RMA split: the host/proxy backend is now registered as an `ncclRma_t` vtable (`RMA_IB_PROXY`) that owns the `iput`/`iputSignal`/`iget`/`iflush`/`test` data-path ops, with GIN layered on top through the generic `ncclGinProxy`.

##### Known issues
* The improved AllGatherV support breaks the NCCL profiler support for ncclBroadcast operations, limiting visibility to API events. `NCCL_ALLGATHERV_ENABLE=0` can be used as a workaround until it is fixed in a future release.
* Multi-node multi-segment and Elastic Buffer symmetric-window registration is not yet enabled; NET and LSA+GIN multi-segment paths depend on runtime support for exporting contiguous DMA-BUF handles across all physical segments.

#### **rocBLAS** (5.6.0)

##### Added

* Per-batch `alpha`/`beta` support for Level 2 batched and strided-batched `symv`, `hemv`, `sbmv`, and `spmv` via `rocblas_set_batch_alpha_stride` and `rocblas_set_batch_beta_stride` (device pointer mode).
* Per-batch `alpha` support for Level 2 batched and strided-batched `syr` via `rocblas_set_batch_alpha_stride` (device pointer mode).
* Per-batch `alpha` (scalar vector) API support for Level 1 `scal_batched`, `scal_strided_batched`, and their `_ex` forms through `rocblas_set_batch_alpha_stride` when `rocblas_handle` is in `rocblas_pointer_mode_device`.
* Support custom build with CMake arguments `BUILD_WITH_HIPBLASLT_ONLY=ON` that bypasses legacy Tensile.

##### Upcoming changes

* Deprecated the `ROCBLAS_USE_HIPBLASLT_BATCHED` environment variable. Recent optimizations mean hipBLASLt no longer needs to be disabled for batched operations only. This environment variable is planned for removal in a future release.

#### **rocDecode** (1.9.0)

##### Added

* Invalid video size handling for AVC/HEVC.

##### Resolved issues

* Fixed decode errors of some AVC interlaced container streams by adding support for the picture data packet from the demuxer which contains multiple pictures.
* Corrected fake CTest passes.

#### **rocFFT** (1.0.39)

##### Added

* Optional ROCm Communication Collectives Library (RCCL) backend for single-node multi-GPU communication, enabled via `-DROCFFT_RCCL_ENABLE=ON`.

##### Changed

* Relaxed the usage requirements for `rocfft_setup` and `rocfft_cleanup`.
* Removed the ROCFFT_RTC_PROCESS_HELPER debug environment variable.

##### Optimized

* Improved performance of unit-strided, interleaved, real-to-complex FFTs on gfx1201, gfx90a, gfx942, and gfx950 for the following lengths:
  * (100,100,100)
  * (192,96,96)
  * (200,96,96)
  * (128,128,256)
  * (160,168,168)
  * (160,168,192)
  * (168,168,192)
  * (168,192,192)
  * (192,192,192)
  * (192,192,200)
  * (192,200,200)
  * (200,200,200)
  * (216,216,216)
  * (216,104,100)
  * (216,104,104)
  * (224,104,104)
  * (224,108,104)
  * (224,108,108)

##### Resolved issues

* Addressed internal issues causing multi-device plans to fall back to the least-performant code path for certain 3D real transforms (e.g., multi-device single-precision real out-of-place 3D of size 320x320x320 using slab decomposition).
* Fixed a thread-safety issue that could cause `rocfft_plan_create` to crash when called concurrently from many threads.

#### **rocJPEG** (1.7.0)

##### Added

* `rocJpegDecodeAsync` and `rocJpegDecodeSync` APIs to support asynchronous single-image JPEG decoding, allowing decode submission and completion to be separated across threads for improved pipeline throughput.

#### **ROCm Compute Profiler** (3.8.0)

##### Added

* ``--pc-sampling-rows`` analyze option to cap the PC sampling table at the top N rows (default 10); set ``0`` to show all. Must be non-negative.

* ``--overwrite`` profile mode option to explicitly allow replacing existing workload output.

* Experimental Triton support to ML API tracing. Profile with `--experimental --triton-trace` to emit a ROCTX marker per Triton/Inductor kernel launch attributed to the user call site, and analyze with `--experimental --list-triton-operators` or `--experimental --triton-operator <pattern>` to list or filter Triton operators independently of Torch.

* Support for GPU metrics on gfx1153 hardware.

##### Changed

* Split Python version requirements by mode. Profile mode now runs on Python 3.8+ (standard library only). Analyze mode requires Python 3.9+ and exits with a clear message on older interpreters instead of failing with an import error.

* `--pc-sampling-sorting-type` now defaults to `count` (was `offset`), so the PC sampling table shows the most-sampled instructions first.

* Renamed the `Pct of Peak` / `PoP` analysis column to `Percent of Peak` in analysis output.

* `--torch-trace` now wraps the tensor methods `to`, `cpu`, `cuda`, and `contiguous` by default. Previously these wraps were enabled by setting `ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS=1`. Set `ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS=0` (or `false`, `no`, `off`) to disable them.

* Renamed the torch-trace output files and directory from `torch_trace_*` to `ml_api_trace_*`.

* Profile mode now errors when the target workload directory is non-empty unless `--overwrite` is passed. `--bench-only` likewise requires `--overwrite` before replacing an existing `roofline.csv`.

* Renamed `num_hbm_channels` to `num_memory_channels` in machine specifications to unify memory channel reporting across GPU families.

##### Removed

* Removed the multi-node analysis options ``--nodes``, ``--list-nodes`` (analyze mode) and the experimental ``--spatial-multiplexing`` option (profile and analyze modes). These features did not work as expected and will be redesigned in a future release.

##### Optimized

* Improved GPU Benchmarking and Roofline profiling/analysis support for gfx1150/gfx1151/gfx1152 architectures.
  * gfx11xx supports Wave Matrix Multiply Accumulate (WMMA), replacing MFMA operations.

##### Resolved issues

* The Dual VALU (VOPD) instruction mix metric is now reported for gfx115x in the WGP panel.

* Fixed multi-user roofline benchmarking on shared systems: the per-GPU lock file under `/tmp/rocprof-compute-benchmark/` is now created world-readable/writable (0666) so any user can acquire it, regardless of which user created it first or the active umask. Stale unreadable lock files left by older versions in a sticky `/tmp` cannot be repaired automatically and must be removed manually by their owner or an administrator.

* Fixed CDNA memory chart CLI output to show the numbered `3. Memory Chart` header without repeating the default per-kernel normalization label.

##### Known issues

* Workloads profiled with earlier versions must be re-profiled before analysis. The sysinfo schema changed and older workload directories are not compatible.

* CLI mode block 4 Roofline plot's legend will not appear if there are too many kernels to list, in relation to the user's terminal size. Same per-kernel roofline rate metrics and AI plot point details can be read in block 4's preceding tables.

#### **ROCm Debugger (ROCgdb)** (16.3)

##### Added

- The "catch hiperr" feature is now exposed to MI too, with a new
  `-catch-hiperr` command and related fields in `*stopped` records.
  See the "HIP Runtime Error" subsection of the "GDB/MI Catchpoint
  Commands" section in the ROCgdb manual.

#### **ROCm Systems Profiler** (1.8.0)

##### Added

- hipFile (GPU-direct storage) API tracing. Add `hipfile_api` to
  `ROCPROFSYS_ROCM_DOMAINS` (shorthand: `hipfile`) to capture hipFile API traces. Requires ROCprofiler-SDK version 1.3.5 or later.

- `--exe-only` flag for `rocprof-sys-instrument`: shorthand for excluding every shared
  library from instrumentation, leaving only the main executable.

- `--exclude-internal-lib-paths` flag for `rocprof-sys-instrument`: by default, each
  internal library is excluded only at the path linked at startup; when enabled, every
  on-disk path matching an internal library's filename is excluded.

- `--max-library-functions` option for `rocprof-sys-instrument`: skips shared libraries
  whose procedure count exceeds the given threshold, keeping instrumentation overhead
  manageable. The target executable is never gated by this, and the check is bypassed by
  the module include/restrict (`--module-include`/`-MI`, `--module-restrict`/`-MR`) and
  function include/restrict (`--function-include`/`-I`, `--function-restrict`/`-R`)
  regexes.

- rocSHMEM host-stream API tracing via `ROCPROFSYS_ROCM_DOMAINS=rocshmem_api`.
  ROCm Systems Profiler now captures the nine host-stream rocSHMEM API calls
  (`putmem_on_stream`, `getmem_on_stream`, `putmem_signal_on_stream`,
  `signal_wait_until_on_stream`, `broadcastmem_on_stream`, `alltoallmem_on_stream`,
  `barrier_all_on_stream`, `sync_all_on_stream`, `quiet_on_stream`) as
  `rocm_rocshmem_api` spans in Perfetto traces and rocpd databases. Requires
  rocprofiler-sdk >= 1.3.4 and rocSHMEM >= 3.6.0 (included in ROCm 10.0).
  As of rocSHMEM 3.6.0, `USE_ROCPROFILER_REGISTER` defaults to `ON`, so
  package installations automatically include this support. A `rocshmem` example
  demonstrating two-PE usage of all nine APIs is included under `examples/rocshmem`.

##### Changed

- `ROCPROFSYS_BUILD_TESTING` no longer implies `ROCPROFSYS_BUILD_EXAMPLES`.

- Introduced the new `profiler-hub` writer backend for trace persistence, as a
  replacement for the existing SQLite3/rocpd backend.

##### Removed

- Removed the `-p` / `--pid` option from `rocprof-sys-instrument` for attaching to
  an already running process. Use the `rocprof-sys-attach` executable instead, which
  attaches to and profiles running processes via the ROCprofiler-SDK `rocattach` API.

- Removed `--parse-all-modules` from `rocprof-sys-instrument`. The tool iterates through objects and modules to extract the functions by default.

#### **rocPRIM** (4.5.0)

##### Added

* A parallel `device_topk`, which finds the largest/smallest K elements from an input array of keys.
* A parallel `device_segmented_topk`, which finds the largest/smallest K elements from segmented groups.
* `device_topk` and `device_segmented_topk` are now controlled by the CMake flag `ROCPRIM_ENABLE_TOPK`. Set `-DROCPRIM_ENABLE_TOPK=ON` to enable these features.
* C++17 style `type_traits` utilities:
 * `is_floating_point_v`
 * `is_integral_v`
 * `is_arithmetic_v`
 * `is_fundamental_v`
 * `is_unsigned_v`
 * `is_signed_v`
 * `is_scalar_v`
 * `is_compound_v`

##### Changed

* Combined and simplified separate assertion templates using `std::is_floating_point`, `rocprim::half`, and `rocprim::bfloat16` to use `rocprim::is_floating_point`.

#### **ROCprofiler-SDK** (1.3.5)

##### Added

**API:**

  - rocSHMEM host-stream API interception for the rocSHMEM tracing domain introduced in 1.3.0:
    - `rocshmem_putmem_on_stream`, `rocshmem_getmem_on_stream`, and `rocshmem_alltoallmem_on_stream` are intercepted and emitted as per-call trace records.
    - Records are interleaved with HIP, HSA, RCCL, and other runtime traces so rocSHMEM communication activity can be viewed on the same timeline as GPU compute.
  - hipFile API tracing as a first-class tracing domain:
    - hipFile API calls are intercepted through dispatch-table wrapping and emitted as per-call trace records alongside HIP, HSA, and other runtime activity.
    - Enables file I/O operations to be correlated with GPU kernels and memory copies in a single profiling timeline.
  - Streaming Performance Monitor (SPM) counter data in the rocpd output format:
    - SPM records are stored as `rocpd_track` rows labelled `SPM`, with counter values grouped by timestamp into `rocpd_sample` rows and per-dimension data in `rocpd_pmc_event` rows.
    - The rocpd schema gains the `sample_id`, `xcc`, `shader_engine`, and `instance` columns.
    - SPM data is consumable by any tool that reads the rocpd database and is convertible to CSV via `rocpd convert`. Conversion to the other output formats, such as Perfetto and OTF2, is not yet supported.

**rocprofv3 (CLI):**

  - OpenMP (OMPT) tracing via the new `--ompt-trace` flag:
    - Accepts a bare boolean or a space-separated category list (`all` `thread` `parallel` `task` `sync` `mutex` `target` `device` `error`), following the same style as `--pmc` and `--output-format`; for example, `--ompt-trace parallel task target sync`. Categories must be space-separated; comma-separated tokens are rejected. Also folded into `--sys-trace`/`--runtime-trace`.
    - rocpd-only trace: records go to the rocpd database (the default output format) and are exported via `rocpd convert`.
    - The OMPT callback layer is already supported by ROCprofiler-SDK; this flag makes it accessible without writing a custom tool.
  - hipFile API tracing via the new `--hipfile-trace` flag (or the `ROCPROF_HIPFILE_API_TRACE` environment variable):
    - Automatically included in `--runtime-trace` and `--sys-trace`.
    - Records are emitted across all supported output backends: CSV, JSON, Perfetto, OTF2, and rocpd.
  - Container-aware `rocattach` symbol resolution: attach entry points are resolved directly from the target process mapped ELF, and tool paths are validated from the target's perspective before injection. This allows attaching from a host to a containerized process without manually copying `.so` files.

##### Changed

- Previously, `rocattach` calculated symbol offsets from the host's `librocprofiler-register.so` and applied them to the target's mapping, which failed when the host and container libraries differ in ELF layout or path. Offsets are now resolved from the target process itself.
- Idle inline queues with no active profiling consumers now bypass queue interposition entirely, reducing overhead for applications that create queues but do not immediately dispatch work.
- DWARF information is now parsed lazily, reducing startup overhead for attach and tracing sessions on large binaries.
- Disabled autoflush in thread trace to prevent premature buffer flushes that produced incomplete or corrupted traces.
- Bump rocpd schema to version 3.0.1 which supports NIC agent types.
- Bump rocpd schema to version 3.0.2 for HIP graph per-node attribution (`graph_exec_id`/`graph_node_id` columns on `rocpd_kernel_dispatch`/`rocpd_memory_copy` and the new `rocpd_graph_launch` table). The pre-graph-attribution 3.0.1 schema is now frozen under `versions/3.0.1/` per the rocpd schema versioning scheme.
- Bump rocpd schema to version 3.0.3 for SPM support. The pre-spm-support 3.0.2 schema is now frozen under `versions/3.0.2/` per the rocpd schema versioning scheme.

##### Removed

- Dependency on `libatomic`. The library was previously linked unconditionally through the `rocprofiler-sdk-atomic` interface target, which caused link failures on toolchains and container images where `libatomic1` is not installed. The single `std::atomic` use that required it has been replaced with explicit memory-ordering synchronization; behavior is unchanged.

##### Resolved issues

- A GPU stall in device thread trace that occurred when thread trace was started before `hsa_init()`.
- A counter-collection stall caused by an `InterceptQueue` ordering bug, and fixed an out-of-bounds write in `Submit()`.
- `roctxMark` calls propagating as kernel rename labels, which caused spurious kernel name changes in traces containing ROCTx markers.
- SQ aliasing on harvested WGPs and multi-counter desync on gfx11xx targets in AQLprofile, and corrected the `GcEaSeCounterBlockMaxEvent` value.
- A guard to prevent double-initialization of the PC sampling service.
- `rocprofv3` attach sessions exiting before all buffered output was flushed; attach sessions now block until the flush completes.
- The ordering of code object callbacks during attach, which could race with tools that depend on ordered delivery.
- The `fmt/format.h` include path, the `fpic` flag for samples, OMP lookup in CI, and clang-tidy quickscan enablement.

##### Known issues

- SPM sessions can remain in a stale state after abrupt termination. See [GitHub issue #6489](https://github.com/ROCm/rocm-systems/issues/6489) for details.

#### **rocRAND** (4.5.0)

##### Removed

* Removed `h_scrambled_sobol(32|64)_constants`, `rocrand_h_scrambled_sobol(32|64)_direction_vectors`, `rocrand_h_sobol(32|64)_direction_vectors` from public namespace.

#### **rocSHMEM** (3.6.0)

##### Added

* New APIs:
   * `rocshmem_broadcast_wave`
   * `rocshmem_fcollect_wave`
   * `rocshmem_alltoall_wave`
   * `rocshmem_reduce_wave`
   * `rocshmem_reducescatter_wave`
* Support for some tile-granular collectives for the IPC backend:
   * `rocshmem_tile_broadcast`
   * `rocshmem_tile_broadcast_wave`
   * `rocshmem_tile_broadcast_wg`
   * `rocshmem_ctx_tile_broadcast`
   * `rocshmem_ctx_tile_broadcast_wave`
   * `rocshmem_ctx_tile_broadcast_wg`
   * `rocshmem_tile_allgather`
   * `rocshmem_tile_allgather_wave`
   * `rocshmem_tile_allgather_wg`
   * `rocshmem_ctx_tile_allgather`
   * `rocshmem_ctx_tile_allgather_wave`
   * `rocshmem_ctx_tile_allgather_wg`
* Single node support for gfx1250 / MI455X.
* Support for HIP Fabric Handles.

##### Changed

* Dropped LLC dependency when compiling HSCO objects.

#### **rocSOLVER** (3.36.0)

##### Added

* 64-bit APIs for the symmetric/Hermitian eigensolvers:
    * SYEV_64 and HEEV_64 (with batched and strided_batched versions)
    * SYEVD_64 and HEEVD_64 (with batched and strided_batched versions)
* Support added for the gfx1250 architecture.

##### Changed

* Clarified the `geblttrf_npvt` API documentation to accurately describe the in-place LU block-factorization storage.

##### Known issues

* The 64-bit eigensolver APIs (SYEV_64, HEEV_64, SYEVD_64, HEEVD_64) require the matrix
  dimensions `n` and `lda` to fit within a 32-bit integer, because their internal tridiagonal
  reduction and back-transformation steps remain 32-bit.

#### **rocSPARSE** (5.0.0)

##### Added
* Blocked ELL format support to the `rocsparse_dense_to_sparse` routine, including the new `rocsparse_bell_set_pointers` function to set the Blocked ELL array pointers.
* CSC format support to `rocsparse_spsv` and `rocsparse_sptrsv`.
* CSC format support to `rocsparse_spsm` and `rocsparse_sptrsm`.

##### Changed
* `rocsparse_spmm` with CSR/CSC and the default algorithm (`rocsparse_spmm_alg_default` or `rocsparse_spmm_alg_csr`) now automatically selects a load-balanced (nnz-split) kernel for strongly skewed matrices (those containing a single very long row for CSR, or column for transposed CSC). Behavior is unchanged for non-skewed matrices and for explicit algorithm choices (`rocsparse_spmm_alg_csr_row_split`, `rocsparse_spmm_alg_csr_nnz_split`, `rocsparse_spmm_alg_csr_merge_path`).
* Deprecated the `rocsparse_spildlt0_input_diag` enum value. It was used to dump the diagonal `D` of the ILDLT(0) factorization, which is now stored in-place on the diagonal entries of the `L` factor.

##### Removed
* The deprecated `rocsparse_indextype_u16` enum.

##### Resolved issues
* Fixed an issue with `rocsparse_spmm`, which produced incorrect results for the Blocked ELL sparse format.

#### **rocThrust** (5.0.0)

##### Added

* Largely in feature parity with CCCL/thrust v3.0.3.
  - `thrust::tuple`, `thrust::pair` and `thrust::zip_iterator` fall back to rocThrust 4.4.0 implementations when a libhipcxx counterpart corresponding to CCCL/libcudacxx >= v3.0.3 is unavailable.
    * `thrust::tuple` and `thrust::pair`: some features may differ from CCCL/thrust v3.0.3.
    * `thrust::zip_iterator`: some iterator concepts present in CCCL/thrust v3.0.3 are missing.

##### Removed

* rocThrust compatibility with PyTorch v2.9 and v2.10 has been removed in this release. Use PyTorch v2.11 or later.

## ROCm 7.14.0

See the [ROCm 7.14.0 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html#rocm-core-sdk-7-14-0-release-notes)
for a complete overview of this release.

#### **AMD SMI** (26.5.0)

##### Added

- NIC processor discovery and information API surface.
  - New C APIs: `amdsmi_get_nic_processor_handles()`, `amdsmi_get_nic_device_bdf()`, `amdsmi_get_nic_fw_info()`, `amdsmi_get_nic_port_statistics()`, and `amdsmi_get_nic_vendor_statistics()`.
  - `amdsmi_get_nic_processor_handles()` enumerates NIC processors by socket; the BDF, firmware, and port/vendor statistics getters are reserved and currently return `AMDSMI_STATUS_NOT_YET_IMPLEMENTED`.

- Exposed APU metrics through the CLI and Python interface.
  - `amd-smi metric` now surfaces APU-specific data under `--usage`, `--power`, `--clock`, `--temperature`, `--fan`, `--voltage`, and `--throttle` when APU metrics are available.
  - `amd-smi monitor` provides APU temperature and clock fallbacks when standard dGPU sensors report N/A.
  - On APU systems, the `--pcie`, `--ecc-blocks`, `--voltage-curve`, `--overdrive`, `--xgmi-err`, and `--energy` sections are not applicable and are omitted.

- The `--partition` flag to `amd-smi metric` for partition-scoped metrics.
  - The `-X`/`--partition` flag switches the temperature, clock, and usage categories to partition-level data sources; throttle metrics are already partition-aware.
  - Reuses the existing temperature/clock/usage section schema and adds partition-only AID/XCP/MID entries within it; socket-only fields with no partition equivalent report `N/A`.
  - When `--partition` is set with `--temperature`: adds MID and per-XCP/XCD temperatures.
  - When `--partition` is set with `--clock`: sources GFX/VCLK/DCLK/SOCCLK from partition metrics and adds per-AID and per-XCP clock entries with their limits.
  - When `--partition` is set with `--usage`: reports per-XCP GFX/JPEG/VCN activity.

- `--folder` support to `amd-smi ras --afid`.
  - `amd-smi ras --afid --folder <DIR>` decodes every `*.cper` in a directory and prints a `file_name | list of afids` table (or a JSON array under `--json`).
  - Records with no AFIDs show `-`; files that cannot be parsed show `decode failed`.

- Wrapped ESMI functions in `amdsmi_go_shim`.
  - Go callers can now access ESMI CPU functionality through the existing `amdsmi_go_shim` interface.

- GPU partitioning conceptual guide and usage examples.
  - New guide at `docs/conceptual/partition.md` covering accelerator partition modes (SPX/DPX/TPX/QPX/CPX), memory partition modes (NPS1/NPS2/NPS4/NPS8), API generations, device enumeration after partition, and BDF encoding.
  - New C++ example: `example/amd_smi_partition_example.cc`.
  - New Python example: `example/amd_smi_partition_example.py`.

- An alias for `amd-smi set -C/--compute-partition` as `amd-smi set --accelerator-partition`.
  - Compute and accelerator partitions are fundamentally the same, so users can now use `--accelerator-partition` to set the compute/accelerator partition.

- Input validation for CPU `set` commands.
  - Out-of-range values are now rejected with a clear error showing the valid range:
    - `--cpu-xgmi-link-width` (0-1)
    - `--cpu-gmi3-link-width` (0-2)
    - `--cpu-lclk-dpm-level` (0-3)
    - `--cpu-disable-apb` (0-3)
  - `--cpu-pwr-limit` values above the socket maximum are now reduced to the maximum and applied, with a warning.

- A compute partition memory allocation mode API.
  - New `amd-smi static --partition` output includes `COMPUTE_PARTITION_MEM_ALLOC_MODE` field.
  - New `amd-smi set --compute-partition-mem-alloc-mode [CAPPING|ALL]` to control memory allocation mode (requires sudo).
  - New APIs: `amdsmi_get_gpu_compute_partition_mem_alloc_mode()`, `amdsmi_set_gpu_compute_partition_mem_alloc_mode()`.
  - New enum: `amdsmi_compute_partition_mem_alloc_mode_t` (`AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_CAPPING`, `AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_ALL`).
  - Reads/writes sysfs: `/sys/class/drm/cardN/device/compute_partition_mem_alloc_mode`.

- `AMDSMI_LINK_TYPE_NUMA` and `AMDSMI_LINK_TYPE_XNUMA` to `amdsmi_link_type_t`.
  - Represent NIC-to-GPU links that cross different PCIe switches on the same CPU (NUMA) or across CPUs (XNUMA).

- PID-grouped process listing across GPUs.
  - `amd-smi process --sort-by-pid` and `amd-smi monitor --sort-by-pid` group output by PID, merging each PID's per-GPU usage into one row.
  - New C and Python API `amdsmi_get_gpu_process_list_by_pid()`.

##### Changed

- Normalized JSON/CSV key casing in `amd-smi metric` clock and temperature sections.
  - The `uclk_aid`, `socclks_mid`, and temperature `xcd` keys are now lowercase (`aid_<N>`, `mid_<N>`, `xcp_<N>`) in JSON and CSV output, matching the existing `xcp_<N>` usage keys; they were previously uppercase (`AID_<N>`, `MID_<N>`, `XCP_<N>`).
  - Human-readable output is unchanged, since it uppercases all keys.

- Normalized JSON/CSV key casing in the `amd-smi topology` NIC-GPU access table.
  - The per-GPU columns are now lowercase (`gpu_<N>` for the BDF header row, `gpu_<N>_topo` for each NIC's status row) in JSON and CSV output, matching the existing `gpu_<N>` keys in the GPU-to-GPU access matrix; they were previously uppercase (`GPU<N>`, `GPU<N>_Topo`).
  - Human-readable output is unchanged, since it uppercases all keys.

- Renamed "AINIC version" to "ionic version" in `amd-smi version` output.
  - The label now correctly reflects that it shows the ionic kernel driver version.

##### Removed

- Removed the non-functional `--decode` flag from `amd-smi ras`. Out-of-band CPER decoding is available via `amd-smi ras --afid --cper-file <path>` or `--afid --folder <DIR>`.

- Removed the unused `amdsmi_nic_link_type_t` enum from the public header. No API or struct referenced it; NIC link types are reported through `amdsmi_link_type_t`, which gains `AMDSMI_LINK_TYPE_NUMA` and `AMDSMI_LINK_TYPE_XNUMA` in this release.

##### Optimized

- Improved Python test runner behavior:
  - Added `-l`/`--list` flag to list all available tests and exit without running them.
  - Added shadow detection: if `amdsmi` loads from a path other than the resolved expected path (`AMDSMI_PATH`, `ROCM_HOME`, `ROCM_PATH`, or `/opt/rocm` default), tests exit early with a clear error message and remediation steps.
  - Non-root invocations now exit with code 1 immediately with a clear message instead of failing mid-test.

##### Resolved issues

- `amd-smi set --power-cap` rejecting the minimum allowed value.
  - The lower bound is now inclusive, so setting the power cap to the exact minimum of the reported range (for example, `210` when the range is 210-300W) succeeds instead of failing validation, matching the inclusive range shown in the error message.

- Corrected invalid AMD SMI status-code names in exception messages and documentation.
  - Some `AmdSmiLibraryException` messages and API documentation entries were misspelled; they now use the correct `AMDSMI_STATUS_*` names.

- A crash in `amdsmi_get_gpu_vram_vendor()` and made `amdsmi_get_gpu_vram_info()` resilient to DRM failures.
  - `amdsmi_get_gpu_vram_vendor()` now validates the output buffer and only writes it on success, fixing a null-pointer dereference on the not-supported path.
  - `amdsmi_get_gpu_vram_info()` now reads the VRAM vendor from sysfs first and treats the DRM ioctl (VRAM type/bit width/bandwidth) as best effort, so the vendor is still returned when the DRM path is unavailable.

- AMD GPU manufacturer name display in `amd-smi static --board`.
  - The CLI now displays the canonical vendor name `Advanced Micro Devices, Inc. [AMD/ATI]` when the board manufacturer name is reported as the raw AMD PCI vendor ID (`0x1002`) because the host `pci.ids` lookup is unavailable. The C and Python APIs continue to return the raw value unchanged.
  - Standardized the hardcoded AMD vendor string on the canonical `pci.ids` spelling (with the comma) so `VENDOR_NAME` and `MANUFACTURER_NAME` are consistent with `lspci`.

- `amd-smi ras --cper` / `amdsmi_get_gpu_cper_entries()` crash (`free(): invalid pointer` / `SIGABRT`) when `libamd_smi.so` is `LD_PRELOAD`-ed under a host with a different libstdc++ (for example, device-metrics-exporter / `gpuagent`).

- `amd-smi ras --cper` failing with `AMDSMI_STATUS_FILE_ERROR` on an empty CPER ring. An empty ring (no RAS records) now reports no CPER records; `amdsmi_get_gpu_cper_entries()` returns `AMDSMI_STATUS_SUCCESS` with `entry_count == 0`.

- `amdsmi_init()` aborting entirely when CPU/ESMI initialization fails.
  - `populate_amd_cpus()` treated an `esmi_init()` failure (non-AMD CPU, missing/unsupported energy or HSMP driver, or a CPU/SMU in a bad state) as fatal, causing all of `amdsmi_init()` to fail so GPU and NIC functionality became unusable. ESMI/CPU discovery is now non-fatal and is skipped on failure, mirroring the NIC discovery paths.
  - Removed an incorrect `static_cast<amdsmi_status_t>(esmi_init())` that conflated the unrelated `esmi_status_t` and `amdsmi_status_t` enums.
  - Added checks for the previously ignored return values of `get_nr_cpu_sockets()`, `get_nr_cpu_cores()`, and `get_nr_threads_per_core()`, plus a guard against a divide-by-zero when a misbehaving driver reports zero sockets or threads.

- `amd-smi static` hanging indefinitely on gfx1153 and gfx950.
  - Added a 60-second timeout to `amdsmi_init()` in the CLI so the process exits with a clear error message instead of hanging when the GPU driver is unresponsive.
  - Added `O_NONBLOCK` to DRM device open during initialization so `open()` returns immediately if the device is wedged.

- `amd-smi ras --afid --cper-file <file>` not showing AFIDs for correctable errors.
  - `aca_decode_corrected_error` was receiving the count of `uint32_t` elements where `decode_afid` expected the count of `uint64_t` elements, causing `decode_error_info` to return `NULL` for all non-standard section types.

- `amd-smi ras --cper --json` producing invalid JSON.
  - Multi-GPU runs emitted a separate JSON array per GPU instead of a single unified array, and `--follow` mode printed an empty `[]` every iteration when no new entries existed. Both are now consolidated into a single JSON document.

- Exposed `amdsmi_get_afids_from_cper` in the Python package.
  - The CPER AFID API was implemented but missing from `py-interface/__init__.py`, making it unavailable to Python callers using `from amdsmi import ...`.

- Python unittest scripts now append a GTest-style summary after test output.
  - All `*_test.py` and `unit_tests.py` scripts print a colored `[PASSED]`/`[SKIPPED]`/`[FAILED]` block after the standard unittest output. Colors are automatically suppressed when output is not a TTY (for example, file redirection, CI log capture).

- Corrected the documented unit of `amdsmi_frequencies_t::frequency`.
  - The struct comment claimed frequencies were in MHz, but `amdsmi_get_clk_freq()` returns them in Hz. The comment now reads "List of frequencies in Hz".
  - Also removed the incorrect "in MHz" note from the `current` field, which is a frequency index, not a frequency value.
  - Updated the Python API reference to state the unit is Hz.

- Fabric telemetry APIs returning the wrong status on non-IFoE systems.
  - `amdsmi_alloc_fabric_telemetry()`, `amdsmi_get_fabric_telemetry_data()`, and `amdsmi_free_fabric_telemetry()` now return `AMDSMI_STATUS_NOT_SUPPORTED` on systems without fabric hardware, consistent with `amdsmi_get_gpu_fabric_info()`.

- `amd-smi static --clock` CSV and human-readable formatting to output frequency levels as strings instead of dictionary objects.

##### Upcoming changes

- `amdsmi_get_gpu_vram_vendor()` is deprecated in favor of `amdsmi_get_gpu_vram_info()` and will be removed in a future ROCm release. It now emits a `DeprecationWarning` from the Python interface and functions as a wrapper of `amdsmi_get_gpu_vram_info()`.

- See {ref}`AMD SMI deprecations <amd-smi-deprecations>`.

#### **HIP** (7.14)

##### Added

- New HIP APIs:
  - Execution Context Management: Support for the following APIs for parity with corresponding CUDA APIs:
    - `hipDeviceGetDevResource` returns the device resource of a given type for a device.
    - `hipDevSmResourceSplitByCount` splits SM resources into groups with at least a minimum SM count.
    - `hipDevSmResourceSplit` splits SM resources into groups with configurable per-group parameters.
    - `hipDevResourceGenerateDesc` generates a resource descriptor from one or more device resources.
    - `hipGreenCtxCreate` creates a green context from a resource descriptor.
    - `hipExecutionCtxDestroy` destroys an execution context.
    - `hipDeviceGetExecutionCtx` returns the default execution context for a device.
    - `hipExecutionCtxStreamCreate` creates a stream on an execution context with specified flags and priority.
    - `hipExecutionCtxGetDevResource` returns the device resource of a given type for an execution context.
    - `hipExecutionCtxGetDevice` returns the device associated with an execution context.
    - `hipExecutionCtxGetId` returns a unique identifier for an execution context.
    - `hipStreamGetDevResource` returns the device resource of a given type for a stream.
    - `hipExecutionCtxRecordEvent` records an event on an execution context.
    - `hipExecutionCtxSynchronize` blocks until all work on an execution context has completed.
    - `hipExecutionCtxWaitEvent` makes an execution context wait on an event.
  - Module Management: Support for the following APIs for parity with corresponding CUDA APIs:
    - `hipLibraryGetGlobal` returns the device pointer and size of a `__device__` global defined in a `hipLibrary_t`. Mirrors `cudaLibraryGetGlobal` / `cuLibraryGetGlobal`.
    - `hipLibraryGetManaged` returns the host pointer and size of a `__managed__` variable defined in a `hipLibrary_t`. Mirrors `cudaLibraryGetManaged` / `cuLibraryGetManaged`.
  - Memory Management: Support for the following APIs for parity with corresponding CUDA APIs:
    - `hipMemDiscardBatchAsync` discards a batch of memory ranges asynchronously, allowing the runtime to reclaim resources. Mirrors `cudaMemDiscardBatchAsync`.
    - `hipDrvMemDiscardBatchAsync` driver API variant of `hipMemDiscardBatchAsync`, using `hipDeviceptr_t` pointers. Mirrors `cuMemDiscardBatchAsync`.
    - `hipMemDiscardAndPrefetchBatchAsync` combines discard and prefetch in a single call, enabling the runtime to optimize data movement. Mirrors `cudaMemDiscardAndPrefetchBatchAsync`.
    - `hipDrvMemDiscardAndPrefetchBatchAsync` driver API variant of `hipMemDiscardAndPrefetchBatchAsync`, using `hipDeviceptr_t` pointers. Mirrors `cuMemDiscardAndPrefetchBatchAsync`.

- Support for non-Host Transparent (nHT) fabric handles in HIP Virtual Memory Management (VMM) APIs, enabling efficient cross-device memory sharing over Infinity Fabric over Ethernet (IFoE). This allows peer devices to directly access shared memory without routing data through the host, reducing data movement overhead and improving performance for multi-GPU and distributed workloads.
- Introduced an exported no-op function `__hipOnError(void *err_info)`, invoked from `HIP_UPDATE_ERROR_STATE` when an API returns a non-success status, enabling debuggers to set breakpoints on a stable symbol. The symbol is exported on ELF (Executable and Linkable Format) platforms via a version script and on Windows via `amdhip.def`. The `err_info` parameter is a pointer to a struct containing the error code, name, and descriptive string.

##### Optimized

- Enhanced HIP graph replay performance for asynchronous memory allocations. HIP graph replay now reduces overhead for graphs that interleave asynchronous memory allocations with compute. Allocation nodes no longer block during replay — physical memory is reused across nodes instead of being mapped and unmapped on each launch, eliminating the gaps between kernels this pattern previously caused.
- Enhanced debug information for illegal memory access errors. In multi-node and multi-GPU environments, it can be difficult to identify the source of a fault. The HIP runtime now includes the hostname, GPU index, and kernel name in GPU fault error messages, improving issue identification and debugging.

##### Resolved issues

- Resolved an issue where graph allocations that escape their originating graph (that is, allocation nodes without a corresponding free node) failed to remain valid after the graph and its executable
  instance were destroyed. Allocations created via stream capture were not properly tracked and were incorrectly classified as reusable, leading to premature unmapping during `hipGraphExecDestroy` and resulting in memory faults on subsequent access.
- Resolved an issue where an error propagated from the `hipModuleGetFunction` API, causing behavior inconsistent with the corresponding CUDA API. The HIP runtime now suppresses this propagated error to align with expected behavior.
- Resolved an issue where a stream entering an invalid state during capture could not recover, even after calling `hipStreamEndCapture`. The stream failed to return to a clean (None) state,
  and subsequent calls to `hipStreamIsCapturing` continued to report an invalidated state, preventing reuse. This behavior is now aligned with CUDA semantics.
- Resolved a race condition in HIP graph nodes. The HIP runtime now correctly manages graph node IDs within each `GraphNode` constructor to ensure thread safety.
  This prevents duplicate ID assignment when multiple threads concurrently construct graph nodes (for example, during XLA command-buffer fusion).
  As a result, nodes are no longer silently dropped from dispatched packets, eliminating uninitialized output buffers and preventing out-of-bounds or corrupted values.
- Segmentation fault in the `hipMemRetainAllocationHandle` API when a pointer allocated with `hipMalloc` was passed. The HIP runtime now validates non-VMM allocations and returns an appropriate error instead.
- Resolved an issue where `__managed__` global variables were misclassified by the `hipPointerGetAttributes` API both before and after kernel access. This behavior has been corrected to align with CUDA semantics.
- Resolved an issue in the classic graph execution path (RunOneNode and RunNodes) where missing synchronization for child graph nodes caused data races and incorrect results when executing graphs with child nodes under multi-stream parallelism.
  The HIP runtime now properly synchronizes child graph nodes within the execution path.
- Issue in `hipGraphMemsetNode` that caused incorrect validation for flat allocations. For 2D `memsets`, the `userData` `width/height/depth` extents are only initialized by `hipMallocPitch` and `hipMalloc3D`;
  allocations from `hipMalloc` leave these fields unset, leading to spurious validation failures. The HIP runtime now skips `userData`-based checks when extents are zero and relies on `ihipMemset3D_validate`
  for accurate size validation. Additionally, the exec flag is propagated through `ihipGraphNodeSetParams` to ensure executable graph updates use the correct validation path.
- Deadlock caused by `hipMemMap` and `hipMemUnmap` operations on the null stream that could lead to hangs. The HIP runtime now implements proper synchronization to all devices with access to a mapped pointer before unmapping it.
- Resolved an issue where streams created within an execution context remained usable after the context was destroyed, which did not align with CUDA behavior. The HIP runtime now flags such streams as detached when their execution context is destroyed and returns `hipErrorStreamDetached` if they are subsequently used.

##### Known issues

- Kernels using `cooperative_groups::reduce()` with block dimensions whose .y or .z component is different from 1 may produce incorrect results or fail to launch.

#### **hipBLAS** (3.5.0)

##### Added

- The following APIs have been added:
  - `hipblasSetBatchAlphaStride()`
  - `hipblasGetBatchAlphaStride()`
  - `hipblasSetBatchBetaStride()`
  - `hipblasGetBatchBetaStride()`
  - `hipblasGetVersion()`
  - `hipblasGetProperty()`

##### Resolved issues

- Guarded x86-specific code and compiler options.

#### **hipBLASLt** (1.4.1)

##### Added

- Introduced a new API: `hipBLASLt_ext::isSolutionSupported()`. This API is used by the new hipBLASLt integration from rocBLAS to check if a given solution is supported for a specific GPU and problem type.

#### **hipCUB** (4.5.0)

##### Added

- Support for the gfx1250 architecture.

##### Upcoming changes

- CCCL 2.8.x compatibility is deprecated. hipCUB and rocThrust will be brought forward to CCCL 3.0.x compatibility in an upcoming release.

#### **hipFFT** (1.0.24)

##### Added

- Support for the gfx1250 architecture.

#### **hipRAND** (3.4.0)

##### Added

- Support for the gfx1250 architecture.

#### **hipSOLVER** (3.5.0)

##### Changed

- Minor internal changes.

#### **hipSPARSE** (4.6.0)

##### Added

- `hipsparseCreateBsr` and `hipsparseCreateConstBsr` to enable BSR format support in generic routines.
- BSR format support to `hipsparseSpMV` and `hipsparseSpMM`.

##### Resolved issues

- Issue where calling `hipsparseSpMV` multiple times with different `hipsparseOperation_t`, `hipsparseSpMVAlg_t`, or compute-datatypes using the same sparse matrix descriptor resulted in errors.

##### Upcoming changes

- The routines `hipsparseXcsrgeamNnz`, `hipsparseScsrgeam`, `hipsparseDcsrgeam`, `hipsparseCcsrgeam`, and `hipsparseZcsrgeam` have been deprecated and will be removed in a future release. Use `hipsparseScsrgeam2_bufferSizeExt`, `hipsparseDcsrgeam2_bufferSizeExt`, `hipsparseCcsrgeam2_bufferSizeExt`, `hipsparseZcsrgeam2_bufferSizeExt`, `hipsparseXcsrgeam2Nnz`, `hipsparseScsrgeam2`, `hipsparseDcsrgeam2`, `hipsparseCcsrgeam2`, and `hipsparseZcsrgeam2` instead.

#### **hipSPARSELt** (0.2.9)

##### Added

- Support for the following data type combinations for the LLVM target gfx942:
  - FP8_FNUZ(E4M3_FNUZ) inputs, F32 output, and F32 Matrix Core accumulation.
  - BF8_FNUZ(E5M2_FNUZ) inputs, F32 output, and F32 Matrix Core accumulation.

#### **MIOpen** (3.5.2)

##### Changed

- [Conv] Naive convolution solvers are now skipped by default during find when any non-naive solver succeeds across any algorithm. Set `MIOPEN_NAIVE_DISABLE_IF_ALT=0` to restore the previous behavior.

##### Resolved issues

- [RNN] RNN workspace tensor descriptor integer overflow.
- [Conv] Enabled grouped Composable Kernel (CK) xdlops fwd, bwd, and wrw convolution (2D and 3D) for tensors whose strides exceed the int32 range.
- [Conv] `miopenStatusInternalError` thrown by Find on depthwise NHWC grouped convolutions under `MIOPEN_FIND_MODE=NORMAL`.

#### **RCCL** (2.30.4)

##### Added

- Compatibility with NCCL 2.30.4, NCCL 2.29.7, and NCCL 2.28.9
- Proxytrace profiler plugin and core proxy-diagnostics hooks (`RCCL_PROXYTRACE`).
- `ncclBarrierSession` LSA validation for barrier sessions.
- Symmetric-memory ReduceScatter kernel (`RailA2A_LsaLD`) on gfx942/gfx950.
- Bias (accumulation) `AllReduce` on gfx1250.
- Optimized scale-up `ReduceScatter`, `AllGather`, and `AllToAll` kernels.
- ROCprofiler-SDK coverage for `ncclCommGrow` and `ncclCommGetUniqueId`.
- Auto-enabled P2P batching for gfx950 in combination with non-AINIC NICs.
- Display HIP/ROCm runtime versions in `NCCL_DEBUG` output.
- Detect ROCm version via core symlink for multi-architecture installs.
- Skip DDA IPC initialization for directMode and MNNVL topologies.
- Load versioned `libamd_smi` SONAME instead of an unversioned symlink.
- Pythonic API bindings under `bindings/nccl4py/` (RCCL fork of NVIDIA `nccl4py` v0.2.0). Provides Python access to RCCL collectives via Cython bindings, an on-disk `cuda.core` HIP shim for ROCm hosts without `cuda-bindings` / `cuda-core`, and RCCL-only collective wrappers (`ncclAllReduceWithBias`, `ncclAllToAllv`).
- RCCL examples to the repository.
- `RCCL host API` pull-in from NCCL 2.30.

##### Changed

- Enabled WarpSpeed auto mode for grow communicators.
- Refactored AllGather algorithm selection; hierarchical AllGather now enabled by default for multi-node.
- Swapped legacy `net_ib` with the `net_ib` implementation from NCCL 2.29.
- Skip per-warp channel LDS copy when `warpComm` is disabled.
- Hardened proxy RPC setup against malformed peer input.
- Changed the bootstrap AllGather to use the bidirectional ring (N/2 steps) by default on the socket OOB path. `NCCL_BOOTSTRAP_BIDIR_ALLGATHER` now defaults to `1`; set it to `0` to fall back to the unidirectional ring. The net OOB path (`NCCL_OOB_NET_ENABLE`) and its bidirectional variant (`NCCL_BOOTSTRAP_BIDIR_NET`) remain off by default.
- `NCCL_PXN_C2C` is kept default-off (`0`); upstream NCCL defaults it to `1` since 2.28. The C2C PXN routing path is currently not applicable on AMD hardware.

##### Removed

- Removed NPKit profiling support (build option ``ENABLE_NPKIT``, headers, device and proxy instrumentation, install script flag ``--npkit-enable``, and related documentation and tooling). Use the profiler plugin API for profiling instead.
- Removed Kernel COLLTRACE support, including the `COLLTRACE` build option, device-side collective trace buffers, debug kernel variants, and related install/CI wiring. The host latency profiler is unchanged.
- Removed legacy `ENABLE_PROFILING` device profiling support and the `PROFILE` build option. Use the profiler plugin API instead.

##### Optimized

- Tuned symmetric memory kernels.
- Parallelized communicator destruction across child processes to reduce teardown latency.

##### Resolved issues

- `ncclCommGrow` channel-count divergence causing incorrect collective routing.
- A `ncclCommGrow` hang when growing to an 8-rank single-node communicator.
- Symmetric LDS under-reservation in legacy (non-device-linker) builds.
- LL128 protocol correctness for gfx1250.
- XGMI topology mapping for multi-system (NPS) nodes.
- gfx950 collective hang caused by a tuner race condition.
- `net_ib_cast`: gate CTS offload path on per-connection state.
- `net_ib`: avoid flagging a non-fatal Isend CTS no-match as a fatal error.
- Acquire-tail polling for gfx950 P2P host staging.
- LDS overflow in device-linker builds.
- Symmetric memory correctness issues.
- `ncclCommFree` to free symmetric window objects automatically (NCCL 2.29.7 defect).
- DDA IPC initialization skip on architectures that do not run DDA.
- Static build (`BUILD_SHARED_LIBS=OFF`) failing with `install(EXPORT "rccl-targets" ...)` error when `fmt` is fetched via `FetchContent`. The `fmt-header-only` target is now scoped to the build interface and excluded from RCCL's exported usage requirements.
- Proxy channel staging buffers ignoring the new GDR mode selection on HIP < 7.12 builds. The legacy `#else` branch in `sendProxyConnect` / `recvProxyConnect` now honors `resources->useDmaBuf`, so peermem-equipped hosts on older HIP no longer fall through to `hsa_amd_portable_export_dmabuf` when peermem was selected in `*ProxySetup`. Workaround for affected RCCL builds: `NCCL_DMABUF_ENABLE=0`.
- RCCL initialization failing (`Failed to find ROCm runtime library`) on runtime-only ROCm trees that ship no unversioned `libhsa-runtime64.so` developer symlink (e.g. TheRock multi-arch pip-wheel `/opt/rocm-less` deployments). RCCL no longer `dlopen`s the HSA runtime by name; instead it directly links `hsa-runtime64::hsa-runtime64` (already a hard transitive dependency via the HIP runtime) and binds `hsa_init`, `hsa_system_get_info`, `hsa_status_string`, and `hsa_amd_portable_export_dmabuf` to those symbols. The linker records `DT_NEEDED libhsa-runtime64.so.1` and resolves it through librccl's existing RPATH, removing the SONAME version-string fragility and load-scope (`RTLD_LOCAL`) issues. The `RCCL_ROCR_PATH` override is no longer needed and has been removed.

##### Known issues

- Elastic-buffer support for GIN (multi-segment symmetric memory windows backed by a mix of device and CPU/`HOST_NUMA` memory, exposed through `NCCL_ELASTIC_BUFFER_REGISTER` and `NCCL_SYM_REUSE_SYSMEM_HANDLES`) was newly synced from upstream and compiles on ROCm, but is unverified on AMD hardware.

#### **RDC** (1.3.1)

##### Added

- 59 new telemetry fields to close the gap with Device Metrics Exporter (DME).
  - Energy: `RDC_FI_GPU_ENERGY` — total energy consumed via `amdsmi_get_energy_count()`.
  - Temperature: `RDC_FI_GPU_JUNCTION_TEMP` — dedicated junction/hotspot temperature field.
  - Clock ranges: `RDC_FI_GPU_CLOCK_MIN`, `RDC_FI_GPU_CLOCK_MAX` — min/max GPU clock frequencies. Additional clock types: `RDC_FI_SOC_CLOCK`, `RDC_FI_VCLK0`, `RDC_FI_DCLK0`.
  - Memory: `RDC_FI_GPU_MEMORY_FREE` (free VRAM), visible VRAM (`RDC_FI_GPU_VIS_VRAM_TOTAL/USED/FREE`), GTT memory (`RDC_FI_GPU_GTT_TOTAL/USED/FREE`).
  - PCIe: `RDC_FI_PCIE_SPEED`, `RDC_FI_PCIE_MAX_SPEED`, `RDC_FI_PCIE_REPLAY_ROLLOVER`, `RDC_FI_PCIE_BANDWIDTH_BIDIR` with sentinel value handling for APU platforms.
  - Instantaneous activity: `RDC_FI_GPU_GFX_BUSY_INST`, `RDC_FI_GPU_VCN_BUSY_INST`, `RDC_FI_GPU_JPEG_BUSY_INST` from `gpu_metrics.xcp_stats`.
  - ECC deferred errors: 19 per-block deferred error fields (`RDC_FI_ECC_*_DE`) plus `RDC_FI_ECC_DEFERRED_TOTAL`, reading `deferred_count` from `amdsmi_error_count_t`.
  - Violation/throttle metrics: 19 new `RDC_HEALTH_*` fields covering accumulated counts and percentages for processor hot, PPT power, socket/VR/HBM thermal, gfx clock host limits, and low utilization violations via `amdsmi_get_violation_status()`. Driver 1.8 XCP/XCC fields return NOT_SUPPORTED on older platforms.

- An automated DME-RDC metric sync check.
  - New script `tools/dme_rdc_metric_sync_check.py` parses DME's protobuf metric definitions and compares against RDC field enums via a curated mapping file (`tools/dme_rdc_metric_mapping.json`).
  - New GitHub Action (`.github/workflows/rdc-dme-sync-check.yml`) runs weekly and on PRs touching metric definitions. Automatically creates GitHub issues when DME adds metrics not yet tracked in RDC.

##### Changed

- Bumped gRPC from 1.67.1 to 1.78.1. See [ROCm/TheRock#4172](https://github.com/ROCm/TheRock/pull/4172).

##### Removed

- Removed RVS integration. [RVS](https://github.com/ROCm/ROCmValidationSuite) is built independently of RDC and TheRock, so its integration has been disabled.
  - `BUILD_RVS` now defaults to `OFF` (#7116).

##### Resolved issues

- The `Failed to insert module: N3amd3rdc10RdcRVSLibE` error.

#### **rocBLAS** (5.5.0)

##### Added

- Per-batch `alpha`/`beta` support for Level 2 batched and strided-batched `gemv` via `rocblas_set_batch_alpha_stride` and `rocblas_set_batch_beta_stride` (device pointer mode).
- Per-batch `alpha` support for Level 2 batched and strided-batched `ger`, `geru`, and `gerc` via `rocblas_set_batch_alpha_stride` (device pointer mode).
- Per-batch `alpha` (scalar vector) API support for `axpy_batched`, `axpy_strided_batched`, and their `_ex` forms through `rocblas_set_batch_alpha_stride` when `rocblas_handle` is in `rocblas_pointer_mode_device`.
- Support custom build with CMake arguments `GPU_TARGET=amdgcnspirv` when using `BUILD_WITH_TENSILE=OFF`.

##### Resolved issues

- Incorrect results on gfx12 in `trsv`, `asum`, and `nrm2` with large `batch_count` exceeding 65536.
- `gemm` with very large `K` or inner product leading dimension for which element byte offset overflowed `int32`.
- `install.sh/rmake.py` builds when `CMAKE_GENERATOR=Ninja` is set.

#### **rocFFT** (1.0.38)

##### Added

- Generalized multi-device computations for transforms such that each length dimension is fully covered either in all the input field's bricks or in all the output field's bricks, regardless of the type and placement of the transform. Specifically for real transforms, the innermost length dimension must be fully covered in all the input (respectively, output) field's bricks for real forward (respectively, inverse) transforms.
- Support for the gfx1250 architecture.

##### Optimized

- Improved performance of even-length real transforms with real lengths between 512 and 8192, extending to larger lengths (up to around 32768) on devices with more LDS.

##### Changed

- Modified the `rocfft_plan_get_work_buffer_size` and `rocfft_execution_info_set_work_buffer` functions to get and set work memory for the current HIP device.
  - Multi-device transforms can require work memory on any of the devices used for input or output bricks, and the current device set at plan creation. Users should loop over the set of devices used by the input/output of the transform and check the work memory requirements for each device.

##### Resolved issues

- Possible incorrect results for multi-dimensional real transforms with small lengths (for example, smaller than 128) along the two fastest-varying dimensions.

#### **ROCgdb** (16.3)

##### Added

- Improved core dumping speed for AMD GPU programs with the `gcore` command,
  particularly for kernels that use small amounts of VRAM.
- A new `catch hiperr` command that stops the inferior when a HIP API
  call returns an error. The convenience variable `$_hiperr` holds
  the error code at the catchpoint.

#### **rocJPEG** (1.6.0)

##### Added

- A logging mechanism for core APIs that can be controlled by setting the `ROCJPEG_LOG_LEVEL` environment variable.

#### **ROCm Compute Profiler** (3.7.0)

##### Added

- `--bench-only` profile mode option to run the roofline microbenchmark standalone (without profiling an application or collecting performance counters). No application run is required. Useful for regenerating `roofline.csv` in an existing workload directory or running the microbenchmark on systems where only HIP is available but ROCprofiler-SDK is not.

- LDS arithmetic intensity as a roofline plot point and analysis database field.

- Backward compatibility for live attach mode to work with older ROCm 7.x.x releases.

- Support for GPU metrics on gfx1150 and gfx1152 hardware.

- Roofline benchmarking support for gfx1150 and gfx1152 hardware.

- Operator statistics and per-operator summary table in the analysis output of torch operator profiling, including the following statistics for every torch operator and its children:
  - Number of invocations.
  - Number of kernel dispatches.
  - Min/Max/Mean and Total duration of kernel dispatches.

##### Changed

- Moved `--gui` and `--tui` analyze options to experimental status. These features now require the `--experimental` flag to be enabled (for example, `rocprof-compute analyze --experimental --gui`).

- `--output-format csv` in analyze mode now uses the database analysis workflow and produces one CSV per analysis view. Requires `--format-rocprof-output rocpd` and no longer prints the report to the terminal (matching `db` format).

- Changed ratio metric aggregation from `AVG(A/B)` (arithmetic mean of per-dispatch ratios) to `SUM(A)/SUM(B)` (ratio of totals) across all analysis YAML configurations and all GPU architectures. `SUM(A)/SUM(B)` is a weighted average where each dispatch contributes proportionally to its denominator magnitude (duration, access count, cycle count). Single-dispatch workloads are unaffected (mathematically identical). Multi-dispatch workloads with different kernels or varying durations will see corrected values.

- `--torch-trace` now captures backward-pass and nested operators that were previously missed or misattributed. The first run builds and caches a helper under `~/.cache/rocprofiler-compute/`, so it takes longer than later runs.

- Profile workload output folder name for Strix Halo series (gfx1151) changed from `strix_halo` to `rdna35_halo`.

- Unified accumulator handling across profile and analyze so each `_ACCUM`-suffixed counter is preserved instead of collapsing to `SQ_ACCUM_PREV_HIRES`.

- Reworded the N/A metric-evaluation warning to "divide-by-zero or empty counter data" (the prior "missing counter data" message could only fire for non-missing causes).

- PC sampling in profile mode now opts in via the `--experimental --pc-sampling` option. Explicit `-b 21` / `--block 21` is no longer accepted on its own.

- PC sampling profiling now emits only `ps_file_results.json`. The per-sample, kernel-trace, and agent-info CSV artifacts are no longer produced or consumed by analysis.

- PC sampling analysis without `-k` now shows the full per-instruction table across all kernels (with a `Kernel_Name` column), identical in schema to the single-kernel view, instead of a collapsed source-line summary.

- `--pc-sampling-interval` now defaults to a method-appropriate value (512 microseconds for `host_trap`, 1048576 cycles for `stochastic`). Stochastic intervals are validated to be a power of 2 and at least 65536; previously invalid values were passed through silently.

##### Removed

- `--path` and `--subpath` options have been removed from profile mode. Use `--output-directory` instead.

- Removed redundant `if (X != 0) else None` divide-by-zero guards from metric equations across all analysis YAML configurations. Division by zero is already handled by the metric evaluation engine, which returns `"N/A"` for `inf` and `NaN` results.

##### Optimized

- Flattened the analyze-mode PMC dataframe to a single-index frame.

- Eliminated "missing counter" warnings during analyze when profile-mode `-b` was used. Analyze now skips metrics outside the selected blocks.

##### Resolved issues

- Roofline panel L1/L2 bandwidth and arithmetic intensity on gfx942 and gfx950 now use the correct 128B cache line, matching the values reported in the Speed-of-Light and vL1D/L2 cache panels for the same run. Bandwidth values on these architectures are 2x and AI values are 0.5x compared to prior releases.

- "ROCPROF_OUTPUT_PATH environment variable must be set" crash that aborted profiling when `ROCPROF_OUTPUT_PATH` was unset or empty (observed when profiling shell-script targets such as `rocprof-compute profile -o /tmp/out -- bash run.sh`). The collector now silently falls back to a documented default instead of aborting.

- `inf` display for metrics with zero-denominator counters (e.g., L2-Fabric Write Latency when no write requests are issued). The metric evaluation path now catches `inf` scalar results and returns `"N/A"`, consistent with existing `NaN` handling.

- Kernels with missing counter data after iteration multiplexing imputation are now excluded from metrics calculations. A warning at analysis time lists the affected kernels. Their execution times remain visible in Top Stats.

- Empirical roofline benchmark to correctly produce double the Matrix BF16 Gflop/s on gfx90a (AMD Instinct MI200 Series) GPUs.

- PC sampling collection now runs when requested via the `pc_sampling` block alias (`--block pc_sampling`), instead of being silently skipped.

##### Upcoming changes

- Roofline support for gfx1153 devices.

##### Known issues

- On gfx1151, `TCP_REQ_sum` is zero in single-pass counter collection, so the related `GL0` metrics always report zero. This issue will be fixed in a future release.

- On gfx1151, `$max_mclk` is not automatically populated in sysinfo, so the related bandwidth metrics may be incorrect. Use `amd-smi` to obtain the maximum memory clock and provide it via `--specs-correction`.

- In analyze mode, `--nodes` is not suitable for multi-rank analysis. Use `--path` with the rank-specific path (for example, `--path workload/1`) instead of `--path workload --nodes 1`.

#### **ROCm Systems Profiler** (1.7.0)

##### Added

- `--output-format` flag for `rocprof-sys-run` and `rocprof-sys-sample` to select
  output format(s) in a single, intuitive option: `proto` (Perfetto), `rocpd`
  (RocPD database), and `json` / `text` (Timemory profile; `txt` aliases `text`).
  Tokens are space- or comma-separated and authoritative — only the listed
  formats are produced. The existing `--trace`, `--profile`, `--flat-profile`,
  and `--profile-format` flags and their environment variables remain available,
  but cannot be combined with `--output-format` on the same command line.

- Unified-memory profiling reports (`unified_memory.txt` and
  `unified_memory.json`) summarizing KFD page fault and page migration events,
  including per-GPU counts, trigger breakdown (`gpu_page_fault`,
  `cpu_page_fault`, `prefetch`), and Host-to-Device/Device-to-Host migration
  bandwidth. Enable with `ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON`; requires
  `HSA_XNACK=1` on an XNACK-capable AMD GPU and ROCm 7.13 or later. For standalone ROCprofiler-SDK installations, ROCprofiler-SDK 1.2.2 or later. The required KFD tracing domains are enabled automatically.

- Dedicated `ROCPROFSYS_UNIFIED_MEMORY_OUTPUT_PATH` setting for routing
  unified-memory profiling reports to an explicit output directory.

- MPI-rank-based console output filtering features controlled with CLI arguments:
  `--rank-filter-logs` and `--rank-filter-id`.

- GPU Hardware Performance Counter (PMC) sampling via the ROCprofiler-SDK device
  counting service. Periodic per-GPU hardware counters are collected alongside
  existing PMC sources and exposed in both Perfetto and RocPD outputs. Specify
  counters with `ROCPROFSYS_GPU_PERF_COUNTERS` (comma-separated; suffix
  `:device=N` to target a specific GPU). Requires ROCprofiler-SDK 0.6.0 or
  later (ROCm 6.4.0+).

- GPU graphics and memory clock frequency metrics (`gfx_clock`, `mem_clock`) via
  AMD SMI, exposing `current_gfxclk` and `current_uclk` in MHz as PMC samples.
  Configure via `ROCPROFSYS_AMD_SMI_METRICS=gfx_clock,mem_clock`.
- Progress bars during trace cache post-processing: perfetto generation
  (`sequential dispatch`) shows one bar per buffered_storage file in turn; rocpd
  generation (`multithreaded dispatch`) shows a single aggregate bar accumulating
  updates from all worker threads.
- Per-stream Perfetto tracks (`HIP Activity Stream {N}`) for kernel dispatch,
  scratch memory, and memory copy events in the trace-cache path, matching the
  buffered tracing behavior. Controlled via `ROCPROFSYS_ROCM_GROUP_BY_QUEUE`
  (default: `false` — group by HIP stream).
- `--list-domains` and `--list-operations <domain>` to `rocprof-sys-avail`.
  These new options allow the user to query more information about available
  ROCm domains (used in `ROCPROFSYS_ROCM_DOMAINS`) and their operations.
- `rocprofsys_push_trace_with_args`, a public API for pushing a user trace region
  with a pre-serialized argument string attached. The arguments are recorded in cached
  tracing mode (the default); in legacy tracing (`ROCPROFSYS_TRACE_CACHED=OFF`) they are
  ignored.

##### Changed

- Split PMC AMD SMI, ROCprofiler-SDK, and procfs wrappers into standalone
  internal backend targets under `source/lib/backends`, replacing the old
  PMC `drivers` layout.
- Removed Boost as a Dyninst dependency by replacing Boost usage with in-tree
  `dyncompat` shims and C++17 standard library equivalents; Bundled Dyninst now
  requires GCC ≥ 10.
- The `trace-openmp` configuration preset no longer includes `HSA_API`,
  by default.
- `rocprof-sys-sample` — Aligned flags with `rocprof-sys-run`. Renamed `--freq`,
  `--cputime` and `--realtime` to `--sampling-freq`, `--sampling-cputime` and
  `--sampling-realtime`, respectively. Old flags are still handled as a part of
  backward compatibility.
- Allowed presets to use `--gpus`/`--cpu`/`--ai-nics` flags without
  `--device`/`--host` flags.
- Minimum required C++ standard raised from C++17 to C++20. timemory now builds
  against the `rocprofiler-systems-cppstd20` branch and spdlog was bumped to
  v1.17.0 (bundled fmt v12).
- Supported environment variables for rank detection: removed `MPI_RANK` and
  `MPI_LOCALRANKID`, added `PMI_RANK` and `SLURM_PROCID`.

##### Resolved issues

- An issue affecting the ElfUtils build on GCC 15.
- Output directory of `rocpd` files was not unique when re-attaching to the same process
  with `rocprof-sys-attach`. Now, each session will have a unique output folder.
- CPU-related counters (like CPU frequency) were missing from `rocpd` output.
- The "group-by-queue" option was not handled correctly in the Perfetto generator.
- The visualization of GPU counters made it look like there was activity
  between kernel dispatches.
- A hang due to mismatched versions of `binutils` between system and bundled
  versions. Ensure that the vendored version of `binutils`'s symbols are hidden.
- The ASAN build on TheRock.
- An issue that could cause certain events to appear in trace, when they should
  have been excluded due to roctx region filtering.
- A CMake issue that caused the wrong version of `elfutils` to be linked when
  building for TheRock. The system version of `elfutils` was used, rather than
  the vendored version causing package install failures.
- Documentation and internal config handling that referenced the non-existent
  `ROCPROFSYS_USE_TRACE`. The Perfetto tracing backend is controlled by
  `ROCPROFSYS_TRACE`; setting `ROCPROFSYS_USE_TRACE` had no effect.
- A pre-main `rocprof-sys-run` `SIGSEGV` in `rocprofiler_configure()` when
  profiling OpenMPI GPU-aware MPI workloads.

##### Known issues

- A push/pop trace count imbalance can occur for workloads that instrument runtime
  internals such as OMPT. When pushes exceed pops, rocprof-sys completes
  finalization, emits a warning, and omits any still-open trace regions from the
  generated trace output.

#### **rocPRIM** (4.5.0)

##### Added

- `generate_resource_spec.cpp` to the test directory, built as a new target by CMake. It generates the resource spec file required by CTest when running tests in parallel.
- Support for the gfx1250 architecture.
- A parallel `device_topk`, which finds the largest/smallest K elements from an input array of keys.

##### Changed

- Updated the documentation on how to run rocPrim tests on multiple GPUs in parallel.

##### Removed

- Removed the `GenerateResourceSpec.cmake` script - it is replaced by the added `generate_resource_spec.cpp` code above.

#### **ROCprofiler-SDK** (1.3.2)

##### Added

**API:**

- Streaming Performance Monitor (SPM) counter collection support (beta):
  - New experimental API in `rocprofiler-sdk/experimental/spm.h`.
  - GPU-timestamped counter values alongside kernel dispatch information.
- `spm_support` along with reserved padding to `rocprofiler_counter_info_v1_t`.

**rocprofv3 (CLI):**

- SPM counter collection support in `rocprofv3` (beta):
  - `--spm <counter>` flag to specify counters for SPM collection.
  - `--spm-sample-interval` and `--spm-sample-interval-unit` parameters to configure sampling rate.
  - `--spm-beta-enabled` flag or the `ROCPROFILER_SPM_BETA_ENABLED` environment variable to opt in to the beta SPM feature via `rocprofv3`. For API-based usage, set `ROCPROFILER_SPM_BETA_ENABLED`.
  - `--spm-config` option in `rocprofv3-avail` to list available SPM configurations.
- JSON and rocpd output format support for SPM.

**Documentation:**

- [SPM API reference guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/api-reference/spm.html).
- [SPM usage guide for rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-spm.html).
- `--spm-config` documentation to [`rocprofv3-avail` usage guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3-avail.html).

##### Changed

- Bumped rocpd schema to version 3.0.1 which supports NIC agent types.

#### **rocRAND** (4.5.0)

##### Added

- Support for the gfx1250 architecture.

#### **rocSHMEM** (3.5.0)

##### Added

- New APIs:
  - `rocshmem_align`
  - `rocshmem_calloc`
  - `rocshmem_buffer_unregister_all`
  - `rocshmem_buffer_register/unregister` for GDA backend
  - `rocshmem_reduce_on_stream`
  - `rocshmem_team_split_2D`
- Tile-granular RMA operations for the IPC backend.
- Host-initiated RMA operations in the IPC backend for the non-MPI bootstrapping path.
- Team creation using non-contiguous parent teams in the IPC backend.
- Python bindings for memory-management APIs.
- Python bindings coverage for team APIs.
- Support for GPU-initiated operations using the SDMA engines.
- ASAN build support.

##### Changed

- Changed default `ROCSHMEM_DEBUG_LEVEL` from `WARN` to `ERROR`.
- Performance optimizations:
  - Separated put/get memcpy primitives to apply correct cache coherence semantics and fences.
  - Use constmem for backend variables and provider muxing.
  - Updated O(1) IPC availability check using pattern detection.

#### **rocSOLVER** (3.35.0)

##### Added

- Support for the gfx1250 architecture.

##### Optimized

- Refined `potf2_run_small` dispatch by `BS2` to avoid over-generating specialized kernels while preserving runtime bounds checks on `nb`.

##### Resolved issues

- An out-of-bounds read in `bdsqr_lower2upper`.
- An invalid kernel launch in the small-matrix LU factorization (GETF2/GETRF) for large batch counts.
- A synchronization issue in GETRI and TRTRI on wave 32 architectures.

#### **rocSPARSE** (4.7.0)

##### Added

- `rocsparse_spildlt0` routine for incomplete LDL' factorization with zero fill-in (ILDLT(0)) for symmetric (real) or Hermitian (complex) sparse matrices in CSR format, with strided batched computations enabled.

##### Known issues

- The HIP graph capture/launch path for the factorization routines `bsric0`, `bsrilu0`, `csric0` and `csrilu0` can fail with `hipErrorOutOfMemory` at `hipGraphLaunch` on memory-constrained GPUs such as the gfx110X family. The corresponding `graph_test` cases are marked `known_bug` and excluded from the standard test suites until the fix lands.

##### Upcoming changes

- Deprecated the `rocsparse_indextype_u16` index type and will be removed in a future release. Use `rocsparse_indextype_i32` or `rocsparse_indextype_i64` instead.

#### **rocThrust** (4.5.0)

##### Added

- Support for the gfx1250 architecture.
- One-time runtime warning for hipstdpar algorithms running on GPUs that support xnack when `__HIPSTDPAR_INTERPOSE_ALLOC__` or `__HIPSTDPAR_INTERPOSE_ALLOC_V1__` are not enabled and xnack is off.

##### Upcoming changes

- CCCL 2.8.x compatibility is deprecated. hipCUB and rocThrust will be brought forward to CCCL 3.0.x compatibility in an upcoming version.

#### **rocWMMA** (2.2.1)

##### Added

- Support for the gfx1250 architecture.
