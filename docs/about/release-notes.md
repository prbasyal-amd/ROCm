# ROCm Core SDK {{ ROCM_VERSION }} release notes

These release notes summarize notable changes since the previous ROCm release.

- [Release highlights](#release-highlights)
- [AMD hardware support](#amd-hardware-support)
- [Operating system support](#operating-system-support)
- [Installation updates](#installation-updates)
- [Kernel driver and firmware bundle support](#kernel-driver-and-firmware-bundle-support)
- [GPU virtualization support](#gpu-virtualization-support)
- [GPU partitioning support](#gpu-partitioning-support)
- [AI ecosystem support](#ai-ecosystem-support)
- [ROCm Core SDK components](#rocm-core-sdk-components)
- [ROCm known issues](#rocm-known-issues)
- [ROCm upcoming changes](#rocm-upcoming-changes)

```{note}
Since ROCm 7.14, ROCm uses [TheRock](https://github.com/ROCm/TheRock) as its build and release system. For more information, see the [transition guide](/about/transition-guide-TheRock).
```

## Release highlights

This release focuses on AI inference, distributed workloads, and profiling across AMD Instinct™, Radeon™, and Ryzen™ AI platforms. Highlights include inference-ready vLLM images and packages, ROCprofiler-SDK adoption across AI profiling workflows, expanded system telemetry and validation coverage, and updates to math, sparse, and communication libraries.

### Platform and hardware support

This release expands GPU, operating system, virtualization, and partitioning support.

#### Expanded AMD GPU support

ROCm 10.0.0 adds support for the following AMD GPUs:

* [AMD Radeon RX 9050 (gfx1151)](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9050.html)

For the complete list of supported AMD hardware, see [AMD hardware support](#amd-hardware-support).

#### Expanded operating system support

ROCm 10.0.0 adds support for Ubuntu 26.04.1 and Ubuntu 24.04.5 on AMD Instinct and Radeon GPUs, replacing support for Ubuntu 26.04 and Ubuntu 24.04.4, respectively.

For the full list of supported Linux distributions, see [Operating system support](#operating-system-support).

#### Expanded GPU virtualization support for Instinct and Radeon GPUs

ROCm 10.0.0 adds support for the following virtualization configurations on AMD Instinct GPUs:

* On MI350XP: VMware ESXi 9.1 with Ubuntu 24.04 guest OS.

Supported Single Root I/O Virtualization (SR-IOV) configurations require the [AMD GPU Virtualization Driver (GIM) 9.1.0.K](https://github.com/amd/MxGPU-Virtualization/releases/tag/9.1.0.K). For details, see [GPU virtualization support](#gpu-virtualization-support).

#### Expanded Instinct GPU partitioning support

ROCm 10.0.0 has no changes in GPU partitioning support from previous release.

For details, see [GPU partitioning support](#gpu-partitioning-support).

### AI inference and frameworks

This release enables support for the following frameworks:

* PyTorch 2.13.0
* JAX 0.11.0
* JAX 0.10.2
* vLLM 0.26.0
* SGLang 0.5.15

The updated framework support replaces the previous PyTorch 2.10.0, JAX 0.9.1, vLLM 0.23.0, and SGLang 0.5.13 support.

For details, see [AI ecosystem support](#ai-ecosystem-support).

### Developer tools, profiling, and validation

This release improves ROCm developer workflows with new HIP APIs, expanded profiling and tracing capabilities, and broader telemetry coverage.

#### HIP feature highlights

The following are notable enhancements to HIP:

##### HIP Record and Replay (HRR) support

HIP Record and Replay (HRR) captures HIP API calls made by an application and stores them in a binary archive (`.hrr`). The recorded workload can then be replayed on a GPU, reproducing application behavior, including multi-threaded execution, graph launches, and GPU memory transfers. This capability enables efficient bug reproduction, performance regression testing, and kernel benchmarking without requiring access to the original application. For more details, see [HIP Record & Replay](https://rocm.docs.amd.com/projects/HIP/en/develop/how-to/debugging.html#hip-record-replay).

##### Improved HIP performance

Improved `hipEventRecord` performance by using the `hipEventDisableTiming` flag to avoid
unnecessary profiling when timing information is not required. Event operations are
now coalesced to eliminate redundant barrier submissions, reducing runtime overhead
and improving execution efficiency.

##### HIP cooperative groups exclusive and inclusive scan support

HIP `cooperative_groups` library adds `cooperative_groups::inclusive_scan` and `cooperative_groups::exclusive_scan` scan APIs in parity with CUDA. Both accept any cooperative group type and an optional custom binary operator, defaulting to summation when none is given.

##### HIP API addition for CUDA parity

HIP adds `hipMemGetDefaultMemPool`, which returns the default memory pool for a given memory location and allocation type.

For more information, see the [HIP section](#hip-10-0-0) in the ROCm component changelogs.

#### ROCprofiler-SDK feature highlights

The following are notable enhancements to ROCprofiler-SDK:

##### rocSHMEM API tracing

ROCprofiler-SDK and `rocprofv3` add rocSHMEM as a first-class tracing domain. Host-stream APIs, including `rocshmem_putmem_on_stream`, `rocshmem_getmem_on_stream`, and `rocshmem_alltoallmem_on_stream` are now intercepted and emitted as per-call trace records. These records appear inline with HIP, HSA, RCCL, and other runtime traces, enabling you to see rocSHMEM communication activity in the same timeline as GPU compute and understand its contribution to overall application performance.

In `rocprofv3`, rocSHMEM tracing is enabled with the `--rocshmem-trace` flag (or the `ROCPROF_ROCSHMEM_API_TRACE` environment variable). It is also automatically included in `--runtime-trace` and `--sys-trace`. Records are emitted across all supported output backends: CSV, JSON, Perfetto, OTF2, and rocpd.

##### OpenMP (OMPT) tracing for rocprofv3

`rocprofv3` exposes OpenMP Tools (OMPT) tracing as a first-class command-line flag. The `--ompt-trace` option accepts a bare Boolean or a space-separated category list, (for example `--ompt-trace parallel task target sync`), following the same style as `--pmc` and `--output-format`. OMPT records are written to the rocpd database on the default output path. The flag is also folded into `--sys-trace` and `--runtime-trace` if you want full-coverage system traces. ROCprofiler-SDK has supported the OMPT callback layer since an earlier release; this change makes it accessible without writing a custom tool.

##### HIP Graph per-node attribution

ROCprofiler-SDK and `rocprofv3` now add full per-graph-node attribution for HIP graph kernels and memory copies. Each dispatch record produced by a graph launch is tagged with the identity of the graph and the specific node within it that produced it. This allows profiling tools to group dispatches by source node across many launches, compute per-node timing and counter aggregates, and correlate graph-level summary records with their individual dispatch records.

In `rocprofv3`, graph attribution is enabled with the `--hip-graph-trace` flag, which is automatically included when using `--hip-trace` or `--hip-runtime-trace`. Attribution data is available in JSON and rocpd output, and can be converted to other formats such as Perfetto, OTF2, and CSV using `rocpd convert`.

##### SPM ROCpd output support

ROCprofiler-SDK extends the rocpd output format to include Streaming Performance Monitor (SPM) counter data. SPM records are stored as `rocpd_track` rows with a `"SPM"` label, with counter values grouped by timestamp into `rocpd_sample` rows and per-dimension data in `rocpd_pmc_event` rows. The rocpd schema is updated to include `sample_id`, `xcc`, `shader_engine`, and `instance` columns. SPM data can now be consumed by any tool that reads the rocpd database, or converted to other output formats such as Perfetto.

Known Issue: SPM sessions can remain in a stale state after abrupt termination. See [GitHub issue #6489](https://github.com/ROCm/ROCm/issues/6489) for details.

##### hipFile tracing support

ROCprofiler-SDK and `rocprofv3` add hipFile as a first-class tracing domain. hipFile API calls are intercepted via dispatch-table wrapping and emitted as per-call trace records alongside HIP, HSA, and other runtime activity. This allows you to see file I/O operations in the same profiling timeline as GPU kernels and memory copies, making it straightforward to quantify storage overhead and its impact on end-to-end application performance.

In `rocprofv3`, hipFile tracing is enabled with the `--hipfile-trace` flag (or the `ROCPROF_HIPFILE_API_TRACE` environment variable). It is also automatically included in `--runtime-trace` and `--sys-trace`. Records are emitted across all supported output backends: CSV, JSON, Perfetto, OTF2, and rocpd.

##### Live Attach with Advanced Thread Trace (ATT) support

ROCprofiler-SDK extends the live attach workflow to include Advanced Thread Trace (ATT). When `rocprofv3` attaches to a running process, it now registers for code-object iteration and creation callbacks so that thread trace can operate correctly on code objects that were loaded before the attach occurred. This makes ATT available for already-running production workloads without requiring an application restart.

##### Container-aware rocattach symbol resolution

`rocprofv3` improves attach support when the target process is running inside a container. ROCprofiler-SDK now resolves attach entry points directly from the target process mapped ELF, and validates tool paths from the target's perspective before injection. This allows attaching from a host to a containerized process without manually copying .so files. Previously, `rocattach` calculated symbol offsets from the host's `librocprofiler-register.so` and applied them to the target's mapping, which failed when the host and container libraries differ in ELF layout or path.

##### Python API for rocprof-trace-decoder

`rocprof-trace-decoder` now ships a Python API, that allows you to decode Advanced Thread Trace (ATT) / SQTT data directly from Python without writing a C++ consumer. The API wraps the decoder library and exposes thread trace decoding as a first-class Python interface, with samples included to demonstrate common workflows. Integration tests for the decoder have been migrated to Python, simplifying test authoring and making it easier for downstream tools to validate their trace-decoding pipelines. This is particularly useful for analysis scripts, Jupyter notebooks, and custom profiling tools that need to process ATT output programmatically.

##### SQTT quick scan support for thread trace path (Experimental)

ROCprofiler-SDK introduces an experimental SQTT quick scan mode for thread trace, accessible through a new CMake flag. The quick scan path collects thread trace data without packet insertion or HSA signal manipulation, removing the queue interception overhead that the standard ATT path requires. Individual kernels can be traced without serialization, and the approach is independent of the ROCm runtime version. This is an experimental feature intended to validate the new collection path and pave the way for out-of-process thread trace and long-kernel tracing in future releases.

##### Removed libatomic dependency

ROCprofiler-SDK no longer depends on `libatomic`. The library was previously linked unconditionally through the `rocprofiler-sdk-atomic` interface target, causing link failures on toolchains and container images where `libatomic1` is not installed. The single `std::atomic` use that required the library has been replaced with explicit memory-ordering synchronization, removing the dependency without changing behavior.

##### Quality and stability improvements

This release includes a range of quality and stability improvements across ROCprofiler-SDK and `rocprofv3`:

* **Thread trace stall issue fixed:** Resolved a GPU stall that occurred when device thread trace was started before `hsa_init()`.
* **Counter collection stall issue fixed:** Corrected an `InterceptQueue` ordering bug that caused counter-collection sessions to stall, and fixed an out-of-bounds write in `Submit()`.
* **Thread trace autoflush disabled:** Disabled autoflush in thread trace to prevent premature buffer flushes that caused incomplete or corrupted traces.
* **roctxMark kernel rename issue fixed:** `roctxMark` calls no longer propagate as kernel rename labels, fixing spurious kernel name changes in traces that contained ROCTx markers.
* **Queue interposition bypass:** Idle inline queues with no active profiling consumers now bypass interposition entirely, reducing overhead for applications that create queues but do not immediately dispatch work.
* **AQLprofile gfx11xx counter issue fixed:** Corrected SQ aliasing on harvested WGPs and multi-counter desync on gfx11xx targets. Also fixed the `GcEaSeCounterBlockMaxEvent` value in AQLprofile.
* **PC sampling service check:** Added a guard to prevent double-initialization of the PC sampling service.
* **Attach output flush:** `rocprofv3` attach sessions now correctly block until all buffered output is flushed before exiting.
* **Code object callback ordering:** Corrected the ordering of code object callbacks during attach to prevent race conditions with tools that depend on ordered delivery.
* **DWARF parsing:** DWARF information is now parsed lazily, reducing startup overhead for attach and tracing sessions on large binaries.
* **Build and CI improvements:** Fixed `fmt/format.h` include path, `fpic` flag for samples, OMP lookup in CI, and clang-tidy quickscan enablement.

#### ROCm Compute Profiler feature highlights

The following are notable enhancements to the ROCm Compute Profiler (rocprofiler-compute):

##### gfx1153 support

Profiling, GPU metrics, and analysis now cover gfx1153. The Dual VALU (VOPD) instruction mix metric is now also reported for gfx115x GPUs in the WGP panel. For the supported hardware list, see [Compatible GPUs/APUs](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/reference/compatible-accelerators.html).

##### Triton operator tracing (experimental)

Operator tracing now covers Triton and `torch.compile` kernels in addition to PyTorch, and a single option traces every supported machine learning framework in one run. For details, see [Triton trace](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/profile/mode.html#triton-trace), [ML API trace](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/profile/mode.html#ml-api-trace), and [Operator filtering](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/analyze/cli.html#operator-filtering).

##### Improved roofline support on gfx1150, gfx1151, and gfx1152

Roofline benchmarking and analysis on these GPUs now report the correct set of supported precisions, so `--roofline-data-type` no longer offers precisions that cannot be measured. Machine specification reporting for APUs is corrected as well. Roofline benchmarking on gfx1153 is not yet supported. For details, see [Standalone roofline](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/profile/mode.html#standalone-roofline) and [Roofline HTML generation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/analyze/cli.html#roofline-html-generation).

For more information, see the [ROCm Compute Profiler section](#rocm-compute-profiler-3-8-0) in the ROCm component changelogs.

#### ROCm Systems Profiler feature highlights

The following are notable enhancements to ROCm Systems Profiler:

##### hipFILE (GPU-direct storage) API tracing

ROCm Systems Profiler can now trace hipFile GPU-direct storage API calls, giving you visibility into storage I/O paths that move data directly between storage and GPU memory. Enable it by adding `hipfile_api` (shorthand: `hipfile`) to `ROCPROFSYS_ROCM_DOMAINS`. This capability requires ROCprofiler-SDK 1.3.5 or later. For details, see the ROCm domains section in [Configuring runtime options](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-systems/docs/how-to/configuring-runtime-options.rst).

##### rocSHMEM host-stream API tracing

ROCm Systems Profiler now captures the nine host-stream rocSHMEM API calls (`putmem_on_stream`, `getmem_on_stream`, `putmem_signal_on_stream`, `signal_wait_until_on_stream`, `broadcastmem_on_stream`, `alltoallmem_on_stream`, `barrier_all_on_stream`, `sync_all_on_stream`, and `quiet_on_stream`) as `rocm_rocshmem_api` spans in both Perfetto traces and rocpd databases. Enable it with `ROCPROFSYS_ROCM_DOMAINS=rocshmem_api`. This capability requires ROCprofiler-SDK 1.3.5 or later and rocSHMEM 3.6.0 or later (included in ROCm 10.0.0). Since rocSHMEM 3.6.0 enables USE_ROCPROFILER_REGISTER by default, package installations include this support automatically. A rocshmem example demonstrating two-PE usage of all nine APIs is included under examples/rocshmem. For details, see the ROCm domains section in Configuring runtime options.

For more information, see the [ROCm Systems Profiler section](#rocm-systems-profiler-1-8-0) in the ROCm component changelogs.

### Libraries

This release updates ROCm math, sparse compute, and communication libraries with additional routines, expanded datatype support, and performance improvements. It also adds the hipFile storage library.

#### hipBLASLt adds GEMM Kernel Optimizer

The GEMM Kernel Optimizer (GEKO) is now available as a command-line tool within hipBLASLt. It lets you tune GEMM kernels for your workloads locally, without sharing confidential model or workload data with AMD. GEKO automates the full tuning workflow, from workload analysis to a final optimized library. It uses Ductile, which replaces exhaustive search for faster tuning. It targets AMD Instinct MI350 Series GPUs (CDNA 4 architecture).

For more information, see the [hipBLASLt documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html).

#### rocPRIM adds parallel top-K algorithms

rocPRIM adds `rocprim::device_topk` and `rocprim::device_segmented_topk`, parallel device-level algorithms that find the largest or smallest K elements from an input array or from segmented groups, respectively. To enable this feature, add the `-DROCPRIM_ENABLE_TOPK=ON` CMake build option. The default variant is hipGraph-compatible; a stable-ordering variant is also available for callers that need guaranteed ordering.

#### rocFFT supports multi-GPU RCCL backend

rocFFT adds an optional RCCL backend for single-node, multi-GPU FFT communication within a single process, enabled via the `-DROCFFT_RCCL_ENABLE=ON` CMake build option. RCCL's GPU topology-awareness targets help improve communication performance over rocFFT's existing memory-copy-based transport in this configuration.

#### hipSPARSE and rocSPARSE feature highlights

The following are notable enhancements to hipSPARSE and rocSPARSE:

##### rocSPARSE and hipSPARSE add Blocked ELL format support

rocSPARSE and hipSPARSE now support Blocked ELL format in their dense-to-sparse conversion routines,`rocsparse_dense_to_sparse` and `hipsparseDenseToSparse`. Each library adds a companion pointer-setter function, `rocsparse_bell_set_pointers` and `hipsparseBlockedEllSetPointers` respectively, to configure the Blocked ELL array pointers.

##### CSC format support for sparse triangular solves in rocSPARSE and hipSPARSE

rocSPARSE and hipSPARSE sparse triangular solve routines now accept matrices in Compressed Sparse Column (CSC) format directly, removing the need to convert to Compressed Sparse Row (CSR) first. CSC support extends to `rocsparse_spsv/rocsparse_sptrsv` and `rocsparse_spsm/rocsparse_sptrsm` in rocSPARSE, and to `hipsparseSpSV` and `hipsparseSpS` in hipSPARSE.

##### rocSPARSE removes rocsparse_indextype_u16 index type

The `rocsparse_indextype_u16` field of the `rocsparse_indextype` enumerator is now removed; and only `rocsparse_indextype_i32` and `rocsparse_indextype_i64` remain. `rocsparse_indextype_u16` was deprecated in ROCm 7.14.0; code that still references it will now fail to compile.

##### rocSPARSE improves default SpMM algorithm selection

rocSPARSE's default `rocsparse_spmm` algorithm now switches to a nnz-split kernel for strongly skewed CSR/CSC matrices  (a single very long row or column). This avoids the throughput loss the previous row-split default caused on such matrices. Non-skewed matrices and explicitly chosen algorithms are unaffected.

##### hipSPARSE adds the SpMV nnz-split algorithm

hipSPARSE adds the `HIPSPARSE_SPMV_CSR_ALG3` algorithm to `hipsparseSpMV`, exposing the rocSPARSE's analysis-free `nnz-split` CSR algorithm (`rocsparse_spmv_alg_csr_nnzsplit`) for sparse matrix-vector multiplication. The algorithm distributes work across threads based on the number of non-zero entries per row and requires no preliminary analysis step before execution.


(release-supported-hw)=

## AMD hardware support

The following table lists supported AMD Instinct GPUs, Radeon GPUs, and Ryzen APUs. Each supported device is listed with its corresponding GPU microarchitecture and LLVM target.

:::{note}

If your GPU is not listed, it might be community-enabled through TheRock nightly builds. For more information, see [TheRock supported GPUs](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md). For installation guidance, see [TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).
:::

```{include} ./include/hardware-support-table.md
:parser: myst
```

(release-supported-os)=

## Operating system support

ROCm supports the following Linux distributions and Microsoft Windows versions. If you're running ROCm on Linux, ensure your system is using a supported kernel version.

:::{important}
The following table is a general overview of supported operating systems. Actual support might vary by AMD GPU or APU. Use the {doc}`Compatibility matrix </compatibility/compatibility-matrix>` to verify support for your specific setup before installation.
:::

```{include} ./include/os-support-table.md
:parser: myst
```

## Installation updates

ROCm 10.0.0 introduces several improvements to the Runfile Installer:

(release-supported-fw)=

## Kernel driver and firmware bundle support

ROCm requires a coordinated stack of compatible firmware, driver, and user-space components. Maintaining version alignment between these layers ensures correct GPU operation and performance, especially for AMD data center products. While AMD publishes the AMD GPU driver and ROCm user space components, your server OEM (original equipment manufacturer) or infrastructure provider distributes the firmware packages. AMD supplies those firmware images (platform level data model (PLDM) bundles), which the OEM integrates and distributes.

```{include} ./include/driver-firmware-support-table.md
:parser: myst
```

(release-virtualization-support)=

## GPU virtualization support

AMD Instinct data center GPUs support virtualization in the following configurations. Supported SR-IOV configurations require the AMD GPU Virtualization Driver (GIM) 9.1.0.K—see the [AMD Instinct Virtualization Driver documentation](https://instinct.docs.amd.com/projects/virt-drv/en/mainline-9.1.0.k/) for more information.

```{include} ./include/virtualization-support-table.md
:parser: myst
```

(release-gpu-partitioning-support)=

## GPU partitioning support

```{include} ./include/partitioning-support-table.md
:parser: myst
```

See the [AMD GPU partitioning](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/index.html) topic in the AMD GPU Driver documentation to learn more.

(release-ai-ecosystem)=

## AI ecosystem support

ROCm 10.0.0 provides optimized support for popular deep learning frameworks and AI inference engines. The following table lists supported frameworks and libraries, their compatible operating systems, and validated versions.

:::{important}
The following table is a general overview of supported frameworks and AI inference engines. Actual support might vary by AMD GPU or APU. Use the {doc}`Compatibility matrix </compatibility/compatibility-matrix>` to verify support for your specific setup.
:::

```{include} ./include/ai-ecosystem-support-table.html
:parser: myst
```

(release-components)=

## ROCm Core SDK components

The following table lists core tools and libraries included in the ROCm 10.0.0 release.

:::{important}
The following table is a general overview of ROCm Core SDK components. Actual support for these libraries and tools can vary by GPU and OS. Use the {doc}`Compatibility matrix </compatibility/compatibility-matrix>` to verify support for your specific setup.
:::

```{include} ./include/core-sdk-components-table.html
:parser: myst
```

### ROCm component changelogs

The following sections describe key changes to ROCm Core SDK components.

```{note}
For a historical overview of ROCm component updates, see the {doc}`ROCm consolidated changelog </release/changelog>`.
```

```{include} ./include/core-sdk-components-aggregated-changelog.md
:parser: myst
```

## ROCm known issues

ROCm known issues are noted on {fab}`github` [GitHub](https://github.com/ROCm/ROCm/labels/Verified%20Issue). These issues will be fixed in a future ROCm release. For known issues related to individual components, review the [ROCm component changelogs](#rocm-component-changelogs).

## ROCm upcoming changes

Future releases will add support for:

* Additional ROCm Core SDK components.

* Domain-specific expansion toolkits (data science, life sciences, finance, simulation, and other HPC domains).

* More AMD hardware support.

(amd-smi-deprecations)=
### AMD SMI deprecations

The AMD SMI library will deprecate the following APIs. Certain APIs will be
deprecated with or without a replacement; see the following tables for details.
We suggest updating your code to use the replacement identifiers before the
targeted removal releases.

#### Planned removal in the next major release

The following APIs, defines, enums, and struct fields are deprecated and
scheduled for removal in the next major release.

##### APIs

| Deprecated | Replacement |
|---|---|
| `amdsmi_get_cpusocket_handles()` | No replacement; functionality removed |
| `amdsmi_get_gpu_vram_vendor()` | `amdsmi_get_gpu_vram_info()` |
| `amdsmi_gpu_driver_reload()` | No replacement; functionality removed |
| `amdsmi_get_xgmi_plpd()` | Python: use the `policy` attribute instead of `plpds` |
| `amdsmi_set_gpu_clk_range()` | `amdsmi_set_gpu_clk_limit()` |

##### Types

| Deprecated | Replacement |
|---|---|
| `amdsmi_fabric_info_ver_t` | Moved inside `amdsmi_fabric_info_t` |
| `amdsmi_nic_fw_t` | `amdsmi_nic_fw_entry_t` |

##### Defines and enums

| Deprecated | Replacement |
|---|---|
| `MAX_NUMBER_OF_AFIDS_PER_RECORD` | `AMDSMI_MAX_NUMBER_OF_AFIDS_PER_RECORD` |
| `MAX_SVI3_RAIL_INDEX` | `AMDSMI_MAX_SVI3_RAIL_INDEX` |
| `MAX_SVI3_RAIL_SELECTION` | `AMDSMI_MAX_SVI3_RAIL_SELECTION` |
| `POWER_EFFICIENCY_MODE_4` | `AMDSMI_POWER_EFFICIENCY_MODE_4` |
| `POWER_EFFICIENCY_MODE_5` | `AMDSMI_POWER_EFFICIENCY_MODE_5` |
| `CENTRIGRADE_TO_MILLI_CENTIGRADE` | No replacement; constant removed |
| `_AMDSMI_MAX_STRING_LENGTH` | No replacement; private symbol, do not use |
| `_AMDSMI_STRING_LENGTH` | No replacement; private symbol, do not use |

##### `amdsmi_gpu_metrics_t` field type widening

The following fields in `amdsmi_gpu_metrics_t` will change from `uint32_t` to `uint64_t` to support next generation AMD Instinct counter ranges:

* `gfx_activity_acc`
* `mem_activity_acc`
* `pcie_nak_sent_count_acc`
* `pcie_nak_rcvd_count_acc`
* `pcie_lc_perf_other_end_recovery`

Recompile any code that reads these fields. Any assignments into fixed-width 32-bit variables must be updated to use 64-bit types.

#### Future deprecation notice: planned removal after the next major release

The following APIs, types, and enums are deprecated and will be removed sometime **after** the next major release.

##### APIs

| Deprecated | Replacement |
|---|---|
| `amdsmi_get_gpu_compute_partition_mem_alloc_mode()` | `amdsmi_get_gpu_accelerator_partition_mem_alloc_mode()` |
| `amdsmi_set_gpu_compute_partition_mem_alloc_mode()` | `amdsmi_set_gpu_accelerator_partition_mem_alloc_mode()` |
| `amdsmi_get_gpu_compute_partition()` | `amdsmi_get_gpu_accelerator_partition_profile()` |
| `amdsmi_set_gpu_compute_partition()` | `amdsmi_set_gpu_accelerator_partition_profile()` |
| `amdsmi_set_gpu_memory_partition()` | `amdsmi_set_gpu_memory_partition_mode()` |

##### Types

* `amdsmi_compute_partition_type_t`
* `amdsmi_compute_partition_mem_alloc_mode_t`

##### Enums

| Deprecated | Replacement |
|---|---|
| `CLK_LIMIT_MIN` | `AMDSMI_CLK_LIMIT_MIN` |
| `CLK_LIMIT_MAX` | `AMDSMI_CLK_LIMIT_MAX` |
| `AGG_BW0` | `AMDSMI_AGG_BW0` |
| `RD_BW0` | `AMDSMI_RD_BW0` |
| `WR_BW0` | `AMDSMI_WR_BW0` |
