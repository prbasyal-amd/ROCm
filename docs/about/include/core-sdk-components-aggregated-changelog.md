## Detailed component changes

The following sections describe key changes to ROCm components.

```{note}
For a historical overview of ROCm component updates, see the {doc}`ROCm consolidated changelog </release/changelog>`.
```

### **AMD SMI** (27.0.0)

#### Added

- **Added accelerator partition memory allocation mode API**.
  - New APIs: `amdsmi_get_gpu_accelerator_partition_mem_alloc_mode()`, `amdsmi_set_gpu_accelerator_partition_mem_alloc_mode()`.
  - New enum: `amdsmi_accelerator_partition_mem_alloc_mode_t` (`AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING`, `AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_ALL`).
  - Supersedes the equivalent `compute_partition` memory allocation mode APIs, which are now deprecated.

#### Changed

- **Bumped the library major version to 27.0.0** (breaking).
  - The shared library SONAME is now `libamd_smi.so.27`. Consumers linked against `libamd_smi.so.26` must relink; no source changes are required beyond the API changes listed elsewhere in this release.

- **Widened five `amdsmi_gpu_metrics_t` accumulator counters from 32-bit to 64-bit** (breaking).
  - `gfx_activity_acc`, `mem_activity_acc`, `pcie_nak_sent_count_acc`, `pcie_nak_rcvd_count_acc`, and `pcie_lc_perf_other_end_recovery` are now `uint64_t` to match the amdgpu pmfw metrics header. Recompile callers that read these fields; the struct layout and field offsets have changed.

- **Aligned the fabric telemetry and NIC firmware API surface with the unified ABI** (breaking).
  - `amdsmi_fabric_telem_id_to_string()` now returns `amdsmi_status_t` and writes the name through a `const char**` out-parameter, instead of returning a `const char*` directly.
  - Renamed `amdsmi_nic_fw_t` to `amdsmi_nic_fw_entry_t`.
  - Added the `amdsmi_fabric_type_t` enumerator `AMDSMI_FABRIC_TYPE_UALINK`; the previous spelling `AMDSMI_FABRIC_TYPE_UALLINK` is retained as a deprecated alias with the same value and is slated for removal in a future ROCm release.

- **Flattened the `amdsmi_fabric_info_t` structure and removed `amdsmi_fabric_info_ver_t`**.
  - The intermediate `amdsmi_fabric_info_ver_t` type was removed from the public header. Its payload union is now the `fabric_info` member of `amdsmi_fabric_info_t`, and its version field is exposed directly as the top-level `fabric_version` field.
  - Field access simplifies from `fabric_info.fabric_version.v1.<field>` to `fabric_info.v1.<field>`, and `fabric_info.version` becomes `fabric_version`.
  - The change is ABI-preserving: field offsets and the overall structure size are unchanged. The Python `amdsmi_get_gpu_fabric_info()` dictionary keys are also unchanged.

- **Prefixed public preprocessor macros with `AMDSMI_` in `amdsmi.h`** (breaking).
  - `MAX_SVI3_RAIL_INDEX`, `MAX_SVI3_RAIL_SELECTION`, `POWER_EFFICIENCY_MODE_4`, `POWER_EFFICIENCY_MODE_5`, and `MAX_NUMBER_OF_AFIDS_PER_RECORD` are now `AMDSMI_MAX_SVI3_RAIL_INDEX`, `AMDSMI_MAX_SVI3_RAIL_SELECTION`, `AMDSMI_POWER_EFFICIENCY_MODE_4`, `AMDSMI_POWER_EFFICIENCY_MODE_5`, and `AMDSMI_MAX_NUMBER_OF_AFIDS_PER_RECORD`. The unused `CENTRIGRADE_TO_MILLI_CENTIGRADE` macro was removed. Update references to the new names.

- **Deprecated the `compute_partition` APIs in favor of `accelerator_partition` equivalents**.
  - `amdsmi_get_gpu_compute_partition()`, `amdsmi_set_gpu_compute_partition()`, `amdsmi_get_gpu_compute_partition_mem_alloc_mode()`, and `amdsmi_set_gpu_compute_partition_mem_alloc_mode()` are slated for removal in a future ROCm release. They now emit a `DeprecationWarning` from the Python interface and delegate to their `accelerator_partition` counterparts.
  - The `amdsmi_compute_partition_type_t` and `amdsmi_compute_partition_mem_alloc_mode_t` enums are deprecated in favor of `amdsmi_accelerator_partition_type_t` and `amdsmi_accelerator_partition_mem_alloc_mode_t`.

- **Deprecated `amdsmi_set_gpu_memory_partition()` in favor of `amdsmi_set_gpu_memory_partition_mode()`**.
  - `amdsmi_set_gpu_memory_partition` is slated for removal in a future ROCm release. It now emits a `DeprecationWarning` from the Python interface and functions as a wrapper of `amdsmi_set_gpu_memory_partition_mode()`.

- **Namespaced `amdsmi_clk_limit_type_t` and `amdsmi_io_bw_encoding_t` enumerators with an `AMDSMI_` prefix**.
  - Added `AMDSMI_CLK_LIMIT_MIN`/`AMDSMI_CLK_LIMIT_MAX` and `AMDSMI_AGG_BW0`/`AMDSMI_RD_BW0`/`AMDSMI_WR_BW0`.
  - The unprefixed names (`CLK_LIMIT_MIN`, `CLK_LIMIT_MAX`, `AGG_BW0`, `RD_BW0`, `WR_BW0`) are retained as deprecated aliases with unchanged values and are slated for removal in a future ROCm release.

- **Removed the `amdsmi_gpu_driver_reload()` API and its Python binding** (breaking).
  - Reload the amdgpu driver out of band with `sudo modprobe -r amdgpu && sudo modprobe amdgpu` to apply memory partition changes instead.

- **Removed unused public macros to match the unified ABI** (breaking).
  - Dropped the `AMDSMI_MAX_VF_COUNT`, `AMDSMI_MAX_DRIVER_NUM`, `AMDSMI_DFC_FW_NUMBER_OF_ENTRIES`, `AMDSMI_MAX_WHITE_LIST_ELEMENTS`, `AMDSMI_MAX_BLACK_LIST_ELEMENTS`, `AMDSMI_MAX_TA_WHITE_LIST_ELEMENTS`, `AMDSMI_MAX_ERR_RECORDS`, `AMDSMI_MAX_PROFILE_COUNT`, and `AMDSMI_PF_INDEX` defines, which were unreferenced by any API or struct.

- **Deduplicated the `amdsmi_fabric_telemetry_category_t` enumerators**.
  - `AMDSMI_FABRIC_TELEMETRY_CATEGORY_INVALID` is now the canonical `0xFFFFFFFF` value; the previous `AMDSMI_FABRIC_TELEMETRY_CATEGORY_UNKNOWN` name (same value) is retained as a deprecated alias and is slated for removal in a future ROCm release.

- **Deprecated `amdsmi_get_gpu_vram_vendor()`**. Use `amdsmi_get_gpu_vram_info()` and read the `vram_vendor` field instead. The function is retained (it now delegates to `amdsmi_get_gpu_vram_info()`, and the Python binding emits a `DeprecationWarning`) and is slated for removal in a future ROCm release.

- **Removed `amdsmi_get_cpusocket_handles()` from the Python interface** (breaking). Use `amdsmi_get_cpu_handles()` instead.

- **Removed the `plpds` key from `amdsmi_get_xgmi_plpd()` Python output** (breaking). Use the `policies` key instead.

- **Removed `amdsmi_set_gpu_clk_range()`** (breaking). Use `amdsmi_set_gpu_clk_limit()` instead.

- **Restructured AMD SMI C++ tests into unit and functional suites**.
  - The `amdsmitst` source tree now separates unit tests from hardware-backed functional tests under `tests/amd_smi_test/unit/` and `tests/amd_smi_test/functional/`.
  - GTest suite names now follow a `<Component><Type>[<Operation>]` scheme: functional tests are `<Component>FunctionalReadOnly`/`<Component>FunctionalReadWrite` (e.g. `GpuFunctionalReadOnly`) and unit tests are `<Component>Unit` (e.g. `GpuUnit`). This replaces the old `amdsmitstReadOnly`/`amdsmitstReadWrite` and `AmdSmiDynamicMetricTest` names.
  - Consumers that pass explicit `--gtest_filter` values should update those filters to the new suite names.
  - See the [AMD SMI test design](docs/conceptual/test-design.md#naming-conventions) for the suite naming convention and `--gtest_filter` usage.

#### Fixed

- **Fixed `amd-smi ras --cper --json` emitting nothing when there are no CPER entries**.
  - The common no-entries case printed empty output, so consumers feeding stdout to `json.loads` failed with `Expecting value: line 1 column 1 (char 0)`. The command now always emits exactly one valid JSON document: `[]` when there are no entries, or a single aggregated array across all GPUs when there are. `--follow` mode stays silent until entries appear. The human-readable primary-partition warning is also suppressed in JSON mode so it no longer corrupts the output.

#### Optimized

- **Optimized `amdsmi_get_gpu_process_list()` to skip redundant KFD topology discovery**.
  - The per-process KFD lookup rebuilt the entire KFD node topology (an expensive sysfs walk) on every call just to translate the device BDF into its KFD GPU id.
  - The caller already knows this value, so it is now passed through to `gpuvsmi_get_pid_info()`, eliminating one full topology discovery per process per refresh. Falls back to the original discovery path when the id is unavailable.

#### Resolved Issues

- **Fixed `amd-smi set --ptl-status` silently failing to change PTL state**.
  - The set path wrote `"1"`/`"0"` to the `ptl/ptl_enable` sysfs node, which only accepts `"enabled"`/`"disabled"`; the driver ignored the numeric write while the API still reported success. The state now changes as expected, and a rejected write returns a real error instead of a generic success.

- **Fixed `amd-smi process` hiding compute processes owned by other users**.
  - A caller without permission to read another process's `/proc/<pid>/fd` was misdetected as running in a separate PID namespace, which caused the whole compute-process list to come back empty. Such processes are now listed with a redacted (`N/A`) name instead of being dropped.

- **Fixed CU%/SDMA column alignment in the `amd-smi` process table**.
  - The `SDMA` header no longer sits a column left of its values, and valid `CU %`/`SDMA` values are no longer truncated.

- **Fixed compute processes being reported on every GPU**.
  - A process was attributed to a GPU whenever it had a KFD context on that GPU, so a job with queues on a single GPU appeared under every GPU. Attribution now uses the process's active KFD queues plus any GPU where it holds a non-zero VRAM allocation, so a process is listed only against the GPUs it actually uses.

- **Fixed `amd-smi` hanging in `amdsmi_init()` on UALink systems when the IFoE driver is unresponsive**.
  - `amdsmi_init()` (and every CLI command) opened a per-GPU IFoE/UALoE fabric session up front, so it blocked indefinitely when the Broadcom IFoE driver was unresponsive, even for queries that never use fabric data.
  - The fabric session is now opened only on the first fabric query, so initialization and non-fabric queries no longer touch the IFoE driver.

- **Fixed ctypes `DeprecationWarning` from `amdsmi_wrapper.py` on Python 3.14**.
  - Python 3.14 deprecates the implicit ctypes structure layout when `_pack_` is set (slated to become an error in 3.19). Each packed structure/union in the generated wrapper now sets `_layout_ = 'ms'`, preserving the existing MSVC-compatible layout (no ABI change) while silencing the warning.

#### Upcoming Changes

- **UUIDs will be replaced by CUIDs in an upcoming version**.
  - UUIDs will soon be replaced with Component Unified IDs (CUIDs). These CUIDs will be consistent across various AMD tools and products so users will be able to definitively identify their devices regardless of what tool they're using.
  - `amdsmi_get_gpu_device_cuid` has been added as an API for this upcoming change but will remain disabled until full support from the amdgpu driver is available.
  - The CLI `list` output and GPU selection now report the CUID in place of the UUID when a CUID is available, and fall back to the UUID otherwise.

### **HIP** (10.0.0)

#### Added
* New HIP APIs
    - Stream Ordered Memory Allocator: Support for the following APIs for parity with corresponding CUDA APIs.
      * `hipMemGetDefaultMemPool` returns the default memory pool for the specified location and allocation type

#### Resolved issues

* Resolved library loading error messages thrown by `rocminfo` during driver initialization in WSL (Windows Subsystem for Linux) environment due to failure in loading the HSA runtime library `libhsa-runtime64.so`
since it is not available in the dynamic linker search path. Since `rocminfo` already links against `libhsa-runtime64.so`, the runtime now correctly locates and loads the HSA runtime library using `RTLD_NOLOAD` option,
enabling successful ROCm initialization, HSA agent discovery, and subsequent ROCm operations.
* Fixed a segmentation fault in HIP queue idle detection caused by referencing a recycled completion signal. Idle state is now derived from a queue-owned signal with a safe lifetime.
* Resolved incorrect NaN handling in the ordered not-equal comparison intrinsics `__hne` (for `__half`) and `__hne` (for `__hip_bfloat16`), along with their vector forms. Being *ordered* predicates, they now correctly return `false` when either operand is NaN.

#### Optimized

* Improved `hipMemcpy2D()` and `hipMemcpy2DAsync()` performance for copy operations with very small row widths and large row counts.
Previously, non-4-byte-aligned row or slice pitches could cause the runtime to issue a separate copy for each row, resulting in significant
performance degradation for workloads such as 1-byte-wide transfers with millions of rows.
These transfers are now handled using a single shader-based copy operation, dramatically reducing transfer times.
Copy operations at or below the 256-row threshold are unchanged.

### **hipBLAS** (3.6.0)

#### Added

* Per-batch `alpha`/`beta` support for Level 2 batched and strided-batched forms of `symv`, `hemv`, `sbmv` and `spmv` via `hipblasSetBatchAlphaStride` and/or `hipblasSetBatchBetaStride` (device pointer mode).
* Per-batch `alpha` support for Level 2 batched and strided-batched forms of `syr` via `hipblasSetBatchAlphaStride` (device pointer mode).
* Per-batch `alpha` (scalar vector) API support for Level 1 batched and strided-batched forms of `scal` and the `_ex` forms through `hipblasSetBatchAlphaStride` when `hipblasHandle_t` is in mode `HIPBLAS_POINTER_MODE_DEVICE`.

#### Resolved issues

* PyTorch users can avoid user constraint based memory allocation failures (`HIPBLAS_STATUS_ALLOC_FAILED`) by exporting `HIPBLAS_WORKSPACE_CONFIG=:0:0` to allow rocBLAS managed memory to grow automatically.

### **hipCUB** (5.0.0)

#### Added

* Feature parity with CCCL/CUB 3.0.0.
* Added `::hip::std` support.

#### Changed

* Changed `CCCL_MINIMUM_VERSION` to `3.0.0` to align with CUB.
* Add support for large num_items `DeviceMerge` and `DeviceSegmentedSort`.
* Replace `#pragma unroll` by `_CCCL_PRAGMA_UNROLL_FULL()` and `_CCCL_PRAGMA_NOUNROLL()` by `_CCCL_PRAGMA_NOUNROLL()`.
* Add `_CCCL_SORT_MAYBE_UNROLL()` in block merge sort and thread sort.
* Update `WarpExchange` template parameters for CUB compatibility.

#### Removed

* hipCUB compatibility with PyTorch v2.9 and v2.10 has been removed in this release.  Please use PyTorch v2.11 or later.
* Removed `hipcub::BaseTraits::CATEGORY`, `hipcub::BaseTraits::nullptr_TYPE` and `hipcub::BaseTraits::PRIMITIVE`.
* Removed  `ConstantInputIterator`, `CountingInputIterator`, `DiscardOutputIterator` and `TransformInputIterator` which were deprecated in hipCUB-4.1.0.
* Removed `DeviceSpmv`, which was removed from CUB after CCCL's 2.8.0 release. Use `hipSPARSE` or `rocSPARSE` libraries instead.
* Removed `GridBarrier`.
* Removed `HIPCUB_MIN`, `HIPCUB_MAX`, `HIPCUB_QUOTIENT_FLOOR`, `HIPCUB_QUOTIENT_CEILING`, `HIPCUB_ROUND_UP_NEAREST` and `HIPCUB_ROUND_DOWN_NEAREST` which were deprecated in hipCUB-4.1.0.
* Removed `LEGACY_PTX_ARCH`.
* Removed `hipcub:max` and `hipcub:min`, which were deprecated. Use `hip::std::max` and `hip::std::min` instead.
* Deprecated `hipcub::Swap`, use `rocprim::swap` instead.
* Deprecated `HIPCUB_IS_INT128_ENABLED`, use `_CCCL_HAS_INT128()` instead.
* Deprecated `hipcub::Equality`, `hipcub::Inequality`, `hipcub::InequalityWrapper`, `hipcub::Sum`, `hipcub::Difference`, `hipcub::Division`, `hipcub::Max` and `hipcub::Min` operators. Use `hip::std::equal_to`, `hip::std::not_equal_to`, `hip::std::plus`, `hip::std::minus`, `hip::std::divides`, `hip::maximum` and `hip:minimum` operators instead.

### **hipFile** (0.4.0)

#### Added

* A KFD-based alternative check for P2P DMA support was added to `ais-check`. This inspects the `capability` property under `/sys/class/kfd/kfd/topology/nodes/*/properties`.
* Added guides for setting up storage targets to the documentation

#### Changed

* `ais-check` now lists the AIS-capable file system mounts detected on the system and fails if none are found.
* Fastpath-only tests are now automatically skipped on systems that do not support the AIS fastpath instead of failing. Running ctest in verbose mode (`ctest -V`) will provide the reason why the test was skipped.
* Updated INSTALL.md to point to official install docs

### **hipSPARSE** (4.7.0)

#### Added
* Added Blocked ELL format support to the `hipsparseDenseToSparse` routine, along with the new `hipsparseBlockedEllSetPointers` function.
* Added the `HIPSPARSE_SPMV_CSR_ALG3` algorithm to `hipsparseSpMV`, which exposes the rocSPARSE CSR nnz split algorithm (`rocsparse_spmv_alg_csr_nnzsplit`).
* Added CSC format support to `hipsparseSpSV`.
* Added CSC format support to `hipsparseSpSM`.

#### Resolved issues
* Fixed an issue with `hipsparseSpMM`, which produced incorrect results for the Blocked ELL sparse format.

### **rocBLAS** (5.6.0)

#### Added

* Per-batch `alpha`/`beta` support for Level 2 batched and strided-batched `symv`, `hemv`, `sbmv`, and `spmv` via `rocblas_set_batch_alpha_stride` and `rocblas_set_batch_beta_stride` (device pointer mode).
* Per-batch `alpha` support for Level 2 batched and strided-batched `syr` via `rocblas_set_batch_alpha_stride` (device pointer mode).
* Per-batch `alpha` (scalar vector) API support for Level 1 `scal_batched`, `scal_strided_batched`, and their `_ex` forms through `rocblas_set_batch_alpha_stride` when `rocblas_handle` is in `rocblas_pointer_mode_device`.
* Support custom build with CMake arguments `BUILD_WITH_HIPBLASLT_ONLY=ON` that bypasses legacy Tensile.

#### Resolved issues
* Fix for issue in `rocblas_gemm_batched_ex_get_solutions`. Starting in `rocBLAS 5.5.0` when using `hipBLASLt` backend it could provide sub-optimal solutions.

#### Upcoming changes

* Deprecated the `ROCBLAS_USE_HIPBLASLT_BATCHED` environment variable. It is no longer required to disable only batched use of hipBLASLt due to optimizations. This env control is planned for removal in a future release.

### **rocDecode** (1.9.0)

#### Added

* Invalid video size handling for AVC/HEVC.

#### Resolved issues

* Fixed decode errors of some AVC interlaced container streams by adding support for the picture data packet from the demuxer which contains multiple pictures.
* Corrected fake CTest passes.
* Resolved vendored libva link issue in samples without extra env vars.

### **rocFFT** (1.0.39)

#### Optimized

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

#### Added

* Added optional RCCL (ROCm Collective Communications Library) backend for single-node multi-GPU communication, enabled via `-DROCFFT_RCCL_ENABLE=ON`.

#### Changed

* Relaxed the usage requirements for `rocfft_setup` and `rocfft_cleanup`.
* Removed the ROCFFT_RTC_PROCESS_HELPER debug environment variable.

#### Resolved issues

* Addressed internal issues causing multi-device plans to fall back to the least-performant code path for certain 3D real transforms (e.g., multi-device single-precision real out-of-place 3D of size 320x320x320 using slab decomposition).
* Fixed a thread-safety issue that could cause `rocfft_plan_create` to crash when called concurrently from many threads.

### **rocJPEG** (1.7.0)

#### Added

* Added rocJpegDecodeAsync and rocJpegDecodeSync APIs to support asynchronous single-image JPEG decoding, allowing decode submission and completion to be separated across threads for improved pipeline throughput.

### **ROCm Compute Profiler** (3.8.0)

#### Added

* Added ``--pc-sampling-rows`` analyze option to cap the PC sampling table at the top N rows (default 10); set ``0`` to show all. Must be non-negative.

* Added ``--overwrite`` profile mode option to explicitly allow replacing existing workload output.

* Improved GPU Benchmarking and Roofline profiling/analysis support for gfx1150/gfx1151/gfx1152 architectures.
  * gfx11 supports Wave Matrix Multiply Accumulate (WMMA), replacing MFMA operations.

* Added experimental Triton support to ML API tracing. Profile with `--experimental --triton-trace` to emit a ROCTX marker per Triton/Inductor kernel launch attributed to the user call site, and analyze with `--experimental --list-triton-operators` or `--experimental --triton-operator <pattern>` to list or filter Triton operators independently of Torch.

* Added support for GPU metrics on gfx1153 hardware.

#### Changed

* Split Python version requirements by mode. Profile mode now runs on Python 3.8+ (standard library only). Analyze mode requires Python 3.9+ and exits with a clear message on older interpreters instead of failing with an import error.

* `--pc-sampling-sorting-type` now defaults to `count` (was `offset`), so the PC sampling table shows the most-sampled instructions first.

* Renamed the `Pct of Peak` / `PoP` analysis column to `Percent of Peak` in analysis output.

* `--torch-trace` now wraps the tensor methods `to`, `cpu`, `cuda`, and `contiguous` by default. Previously these wraps were enabled by setting `ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS=1`. Set `ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS=0` (or `false`, `no`, `off`) to disable them.

* Renamed the torch-trace output files and directory from `torch_trace_*` to `ml_api_trace_*`.

* Profile mode now errors when the target workload directory is non-empty unless `--overwrite` is passed. `--bench-only` likewise requires `--overwrite` before replacing an existing `roofline.csv`.

* Renamed `num_hbm_channels` to `num_memory_channels` in machine specifications to unify memory channel reporting across GPU families.

#### Removed

* Removed the multi-node analysis options ``--nodes``, ``--list-nodes`` (analyze mode) and the experimental ``--spatial-multiplexing`` option (profile and analyze modes). These features did not work as expected and will be redesigned in a future release.

#### Resolved issues

* The Dual VALU (VOPD) instruction mix metric is now reported for gfx115x in the WGP panel.

* Fixed multi-user roofline benchmarking on shared systems: the per-GPU lock file under `/tmp/rocprof-compute-benchmark/` is now created world-readable/writable (0666) so any user can acquire it, regardless of which user created it first or the active umask. Stale unreadable lock files left by older versions in a sticky `/tmp` cannot be repaired automatically and must be removed manually by their owner or an administrator.

* Fixed CDNA memory chart CLI output to show the numbered `3. Memory Chart` header without repeating the default per-kernel normalization label.

#### Known issues

* Workloads profiled with earlier versions must be re-profiled before analysis. The sysinfo schema changed and older workload directories are not compatible.

* CLI mode block 4 Roofline plot's legend will not appear if there are too many kernels to list, in relation to the user's terminal size. Same per-kernel roofline rate metrics and AI plot point details can be read in block 4's preceding tables.

### **ROCm Debugger (ROCgdb)** (16.3)

#### Added

- Dumping core of AMD GPU programs with the "gcore" command is now
  significantly faster, particularly for kernels that use small
  amounts of VRAM.
- New "catch hiperr" command that stops the inferior when a HIP API
  call returns an error.  The convenience variable `$_hiperr` holds
  the error code at the catchpoint.

### **ROCm Systems Profiler** (1.8.0)

#### Added

- hipFILE (GPU-direct storage) API tracing. Add `hipfile_api` to
  `ROCPROFSYS_ROCM_DOMAINS` (shorthand: `hipfile`) to capture hipFILE API traces. Requires ROCProfiler-SDK version 1.3.5 or later.

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

#### Changed

- `ROCPROFSYS_BUILD_TESTING` no longer implies `ROCPROFSYS_BUILD_EXAMPLES`.

- Introduced the new `profiler-hub` writer backend for trace persistence, as a
  replacement for the existing SQLite3/rocpd backend.

#### Removed

- Removed the `-p` / `--pid` option from `rocprof-sys-instrument` for attaching to
  an already running process. Use the `rocprof-sys-attach` executable instead, which
  attaches to and profiles running processes via the rocprofiler-sdk rocattach API.

- Removed `--parse-all-modules` from `rocprof-sys-instrument`. The tool iterates through objects and modules to extract the functions by default.

### **rocPRIM** (4.5.0)

#### Added

* Added a parallel `device_topk`, which finds the largest/smallest K elements from an input array of keys.
* Added a parallel `device_segmented_topk`, which finds the largest/smallest K elements from segmented groups.
* `device_topk` and `device_segmented_topk` are controlled by cmake flag `ROCPRIM_ENABLE_TOPK`. Passing `-DROCPRIM_ENABLE_TOPK=ON` to enable these features
* Added C++ 17 style type_traits utilities
 * is_floating_point_v
 * is_integral_v
 * is_arithmetic_v
 * is_fundamental_v
 * is_unsigned_v
 * is_signed_v
 * is_scalar_v
 * is_compound_v

#### Changed

* Combined and simplified separate assertion templates using `std::is_floating_point`, `rocprim::half`, and `rocprim::bfloat16` to use `rocprim::is_floating_point`

### **ROCprofiler-SDK** (1.3.5)

#### Added

**API:**

  - Streaming Performance Monitor (SPM) counter collection support (beta):
    - New experimental API in `rocprofiler-sdk/experimental/spm.h`:
    - GPU-timestamped counter values alongside kernel dispatch information.
  - Added `spm_support` along with reserved padding to `rocprofiler_counter_info_v1_t`

**rocprofv3 (CLI):**

  - SPM counter collection support in `rocprofv3` (beta):
    - `--spm <counter>` flag to specify counters for SPM collection.
    - `--spm-sample-interval` and `--spm-sample-interval-unit` parameters to configure sampling rate.
    - `--spm-beta-enabled` flag to opt in to the beta SPM feature.
    - `--spm-config` option in `rocprofv3-avail` to list available SPM configurations.
  - JSON and rocpd output format support for SPM.
  - OpenMP (OMPT) tracing via the new `--ompt-trace` flag:
    - Accepts a bare boolean or category list (`all, thread, parallel, task, sync, mutex, target, device, error`); also folded into `--sys-trace`/`--runtime-trace`.
    - rocpd-only trace: records go to the rocpd database (auto-added when another format is requested) and export via `rocpd convert`.

**Documentation:**

  - SPM API reference guide (`api-reference/spm.rst`).
  - SPM usage guide for `rocprofv3` (`how-to/using-spm.rst`).
  - `--spm-config` documentation to `rocprofv3-avail` usage guide.

#### Changed
- Bump rocpd schema to version 3.0.1 which supports NIC agent types.
- Bump rocpd schema to version 3.0.2 for HIP graph per-node attribution (`graph_exec_id`/`graph_node_id` columns on `rocpd_kernel_dispatch`/`rocpd_memory_copy` and the new `rocpd_graph_launch` table). The pre-graph-attribution 3.0.1 schema is now frozen under `versions/3.0.1/` per the rocpd schema versioning scheme.
- Bump rocpd schema to version 3.0.3 for SPM support. The pre-spm-support 3.0.2 schema is now frozen under `versions/3.0.2/` per the rocpd schema versioning scheme.

### **rocRAND** (4.5.0)

#### Removed

* Removed `h_scrambled_sobol(32|64)_constants`, `rocrand_h_scrambled_sobol(32|64)_direction_vectors`, `rocrand_h_sobol(32|64)_direction_vectors` from public namespace.

### **rocSOLVER** (3.36.0)

#### Added

* 64-bit APIs for the symmetric/Hermitian eigensolvers
    * SYEV_64 and HEEV_64 (with batched and strided\_batched versions)
    * SYEVD_64 and HEEVD_64 (with batched and strided\_batched versions)
* Support added for the gfx1250 architecture.

#### Changed

* Clarified the `geblttrf_npvt` API documentation to accurately describe the in-place LU block-factorization storage.

#### Known issues

* The 64-bit eigensolver APIs (SYEV_64, HEEV_64, SYEVD_64, HEEVD_64) require the matrix
  dimensions `n` and `lda` to fit within a 32-bit integer, because their internal tridiagonal
  reduction and back-transformation steps remain 32-bit.

### **rocSPARSE** (5.0.0)

#### Added
* Added Blocked ELL format support to the `rocsparse_dense_to_sparse` routine, including the new `rocsparse_bell_set_pointers` function to set the Blocked ELL array pointers.
* Added CSC format support to `rocsparse_spsv` and `rocsparse_sptrsv`.
* Added CSC format support to `rocsparse_spsm` and `rocsparse_sptrsm`.

#### Changed
* `rocsparse_spmm` with CSR/CSC and the default algorithm (`rocsparse_spmm_alg_default` or `rocsparse_spmm_alg_csr`) now automatically selects a load-balanced (nnz-split) kernel for strongly skewed matrices (those containing a single very long row for CSR, or column for transposed CSC). Behavior is unchanged for non-skewed matrices and for explicit algorithm choices (`rocsparse_spmm_alg_csr_row_split`, `rocsparse_spmm_alg_csr_nnz_split`, `rocsparse_spmm_alg_csr_merge_path`).

#### Resolved issues
* Fixed an issue with `rocsparse_spmm`, which produced incorrect results for the Blocked ELL sparse format.

#### Removed
* The deprecated `rocsparse_indextype_u16` enum.

#### Upcoming changes
* Deprecated the `rocsparse_spildlt0_input_diag` enum value. It was used to dump the diagonal `D` of the ILDLT(0) factorization, which is now stored in-place on the diagonal entries of the `L` factor. It will be removed in a future release.

### **rocThrust** (5.0.0)

#### Added

* Largely in feature parity with CCCL/thrust v3.0.3.
  - `thrust::tuple`, `thrust::pair` and `thrust::zip_iterator` fall back to rocThrust 4.4.0 implementations when a libhipcxx counterpart corresponding to CCCL/libcudacxx >= v3.0.3 is unavailable, ie
    * `thrust::tuple` and `thrust::pair`: some features may differ from CCCL/thrust v3.0.3.
    * `thrust::zip_iterator`: some iterator concepts present in CCCL/thrust v3.0.3 are missing.

#### Removed

* rocThrust compatibility with PyTorch v2.9 and v2.10 has been removed in this release.  Please use PyTorch v2.11 or later.
