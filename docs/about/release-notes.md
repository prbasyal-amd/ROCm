# ROCm Core SDK {{ ROCM_VERSION }} release notes

ROCm Core SDK {{ ROCM_VERSION }} transitions ROCm to [TheRock](https://github.com/ROCm/TheRock), a build and release system that introduces a modular architecture to improve flexibility, maintainability, and alignment with community use cases:

* **Leaner core**: The Core SDK focuses on essential runtime and development components.
* **Use case-specific expansions**: Optional domain-specific SDKs for AI, data science, and HPC.
* **Modular installation**: Install only the components required for your workflow.

This approach streamlines installation, reduces footprint, and accelerates innovation through independently released packages. To learn more, see the [transition guide](/about/transition-guide-TheRock).

(preview-stream-note)=
:::{note}
ROCm {{ ROCM_VERSION }} follows the [versioning discontinuity that began with the 7.9.0 preview](https://rocm.docs.amd.com/en/7.9.0-preview/about/release-notes.html#preview-stream-note) release.
:::

## Release highlights

This release focuses on AI inference, distributed workloads, and profiling across AMD Instinct™, Radeon™, and Ryzen™ AI platforms. Highlights include inference-ready vLLM images and packages, ROCprofiler-SDK adoption across AI profiling workflows, expanded system telemetry and validation coverage, and updates to math, sparse, and communication libraries.

### Platform and hardware support

This release expands GPU, operating system, virtualization, and partitioning support.

#### Expanded AMD GPU support

ROCm 10.0.0 adds support for the following AMD GPUs:

* [AMD Radeon RX 9050 (gfx1151)](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9050.html)

For the complete list of supported AMD hardware, see [AMD hardware support](#amd-hardware-support).

#### Expanded operating system support

ROCm 10.10.0 adds support for Ubuntu 26.04.1 and Ubuntu 24.04.5 on AMD Instinct and Radeon GPUs, replacing support for Ubuntu 26.04 and Ubuntu 24.04.4, respectively.

For the full list of supported Linux distributions, see [Operating system support](#operating-system-support).

#### Expanded GPU virtualization support for Instinct and Radeon GPUs

ROCm 10.0.0 adds support for the following virtualization configurations on AMD Instinct GPUs:

* On MI350XP: VMware ESXi 9.1 with Ubuntu 24.04 guest OS.

Supported Single Root I/O Virtualization (SR-IOV) configurations require the [AMD GPU Virtualization Driver (GIM) 9.1.0.K](https://github.com/amd/MxGPU-Virtualization/releases/tag/9.1.0.K). For details, see [GPU virtualization support](#gpu-virtualization-support).

#### Expanded Instinct GPU partitioning support

ROCm 10.0.0 has enabled and optimized multi-VF partition modes for the following GPU partitioning configurations in SR-IOV deployments:

On MI355X and MI350X:

DPX compute partition mode with NPS2 memory partitioning.

CPX compute partition mode with NPS2 memory partitioning.

For details, see [GPU partitioning support](#gpu-partitioning-support).

### AI inference and frameworks

This release enables support for the following frameworks:

* PyTorch 2.12.0
* JAX 0.10.0
* vLLM 0.23.0 <a id="id4" class="footnote-reference brackets" href="#vllm-support-footnotes" role="doc-noteref"><span class="fn-bracket">[</span>*<span class="fn-bracket">]</span></a>
* SGLang 0.5.13
* TensorFlow 2.21

The updated framework support replaces the previous PyTorch 2.9.1, JAX 0.8.2, vLLM 0.19.1, and SGLang 0.5.9 support.

For details, see [AI ecosystem support](#ai-ecosystem-support).

<aside class="footnote brackets" id="vllm-support-footnotes" role="doc-footnote">
<span id="fn4" class="label"><span class="fn-bracket">[</span><a href="#id4" role="doc-backlink">*</a><span class="fn-bracket">]</span></span>
<p>You might observe significantly longer LLM warmup times on some Radeon GPUs. Refer to the <a href="#vllm-warmup-known-issue">known issue</a> for details.</p>
</aside>

### Developer tools, profiling, and validation

This release improves ROCm developer workflows with new HIP APIs, expanded profiling and tracing capabilities, and broader telemetry coverage.

#### HIP feature highlights

The following are notable enhancements to HIP:


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

Known Issue: SPM sessions can remain in a stale state after abrupt termination. See [GitHub issue #6489](https://github.com/ROCm/ROCm/issues/6489 for details.

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

#### AMD SMI feature highlights

The following are notable enhancements to AMD SMI:

* **Per-partition GPU metrics**: AMD SMI now reports temperature, clock, and usage at the partition level through the new `amd-smi metric --partition` flag, giving partition-level observability where previously only socket-level metrics were available. For CLI usage, see [AMD SMI CLI tool](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html); for partitioning concepts, see [GPU partitioning](https://rocm.docs.amd.com/projects/amdsmi/en/latest/conceptual/partition.html).

* **Compute partition memory allocation mode**: AMD SMI now controls memory allocation behavior at the compute partition level through the new `amd-smi set --compute-partition-mem-alloc-mode` command. The current mode is visible in `amd-smi static --partition` output, and new C and Python APIs expose the same controls programmatically.

* **APU CLI metrics**: AMD SMI now surfaces APU-specific data through the existing `amd-smi metric` flags when APU metrics are available. `amd-smi monitor` adds temperature and clock fallbacks when standard discrete GPU sensors report N/A.

* **APU VRAM carve-out and GTT tuning**: AMD SMI now tunes APU memory from the command line, consolidating the get and set controls previously handled by the standalone `amd-ttm` tool and adding VRAM carve-out configuration. Carve-out and GTT changes take effect after the next reboot, and AMD SMI rebuilds the initramfs automatically so the new configuration is applied at boot. For details, see the memory tuning section in [AMD SMI CLI tool](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html).

* **PID-grouped process listing**: AMD SMI now groups multi-GPU process output by PID with `amd-smi process --sort-by-pid` and `amd-smi monitor --sort-by-pid`, merging each process's per-GPU usage into a single row. A new C and Python API, `amdsmi_get_gpu_process_list_by_pid()`, exposes the same data programmatically.

* **Fabric clock (FCLK) capping on MI300A**: You can now cap the maximum fabric clock (FCLK) on AMD Instinct MI300A APUs to steer power, using the new `fclk` clock type for `amd-smi set --clk-limit`. Only a maximum limit is supported.

* **Go bindings for CPU telemetry**: AMD SMI now exposes EPYC System Management Interface (ESMI) CPU functionality through its Go bindings, so Go applications can query CPU telemetry in-process without invoking external binaries or embedding C or Python runtimes. This simplifies integrating AMD CPU observability into Go-based control planes. For details, see [AMD SMI Go interface](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-go-lib.html).

For more information, see the [AMD SMI section](#amd-smi-26-5-0) in the ROCm component changelogs.

#### RDC expands telemetry coverage for DME parity

ROCm Data Center (RDC) adds 59 telemetry fields, bringing its metric coverage near parity with the Device Metrics Exporter (DME). New fields cover energy, temperature, clocks, memory, PCIe, engine activity, error correction code (ECC), and health and throttle metrics. Some metrics require recent driver and hardware support. For the available field groups and how to monitor them, see [Using RDC features](https://rocm.docs.amd.com/projects/rdc/en/latest/how-to/using_RDC_features.html).

For more information, see the [RDC section](#rdc-1-3-1) in the ROCm component changelogs.

#### ROCm Bandwidth Test (RBT) reaches end-of-life

ROCm Bandwidth Test (RBT) is deprecated and reaches end-of-life with the TheRock-based ROCm 10.0.0 release. Active development has ceased, and no further feature enhancements or fixes are planned. For equivalent and expanded functionality, transition to [TransferBench](https://rocm.docs.amd.com/projects/TransferBench/en/latest/) and the [ROCm Validation Suite (RVS)](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/).

For more details, refer to the [ROCm Bandwidth Test](https://rocm.docs.amd.com/projects/rocm_bandwidth_test/en/latest/) documentation.

### Libraries

This release updates ROCm math, sparse compute, and communication libraries with additional routines, expanded datatype support, and performance improvements. It also adds the hipFile storage library.

#### hipFile direct storage I/O support

hipFile is now included in the ROCm Core SDK, enabling direct data transfers between storage and GPU memory as part of AMD Infinity Storage. hipFile enables storage-intensive workloads to bypass host-side copies, reducing latency and command overhead for high-throughput GPU I/O.

hipFile is supported on Linux with AMD Instinct GPUs. See the [ROCm hipFile examples](https://github.com/ROCm/rocm-examples/tree/release/therock-7.14/Systems/hipFile) and the [hipFile documentation](https://rocm.docs.amd.com/projects/hipFile/en/latest/) to get started.

#### hipBLASLt adds GEMM Kernel Optimizer

The GEMM Kernel Optimizer (GEKO) is now available as a command-line tool within hipBLASLt. It lets you tune GEMM kernels for your workloads locally, without sharing confidential model or workload data with AMD. GEKO automates the full tuning workflow, from workload analysis to a final optimized library. It uses Ductile, which replaces exhaustive search for faster tuning. It targets AMD Instinct MI350 Series GPUs (CDNA 4 architecture).

For more information, see the [hipBLASLt documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html).

#### Per-batch scalar coefficients for Level 2 batched BLAS

rocBLAS and hipBLAS now support per-batch scalar coefficients for Level 2 batched and strided-batched routines in device pointer mode. Each batch index uses its own device-resident scalar rather than a single value shared across the entire batch:

* **GEMV**: per-batch alpha and beta.
* **GER, GERU, and GERC**: per-batch alpha.

S, D, C, and Z precision variants are available for all routines.

For more information, see the [rocBLAS section](#rocblas-5-5-0) and [hipBLAS section](#hipblas-3-5-0) in the ROCm component changelogs.

#### Per-batch alpha for axpy_batched and axpy_strided_batched

rocBLAS adds per-batch alpha support for `axpy_batched`, `axpy_strided_batched`, and their `_ex` variants via `rocblas_set_batch_alpha_stride` in device pointer mode.

For more information, see the [rocBLAS section](#rocblas-5-5-0) in the ROCm component changelogs.

#### hipSPARSE feature highlights

The following are notable enhancements to hipSPARSE:

* **BSR format support in hipSPARSE generic routines**: hipSPARSE adds Block Sparse Row (BSR) format support to its generic sparse compute routines: `hipsparseSpMM` (sparse matrix-matrix multiplication) and `hipsparseSpMV` (sparse matrix-vector multiplication). Two new descriptor functions, `hipsparseCreateBsr` and `hipsparseCreateConstBsr`, let you construct BSR-format sparse matrices for use with the generic API. This brings hipSPARSE to parity with the equivalent NVIDIA cuSPARSE routines, where BSR was previously available only through the rocSPARSE API.

* **Legacy SpGEAM routines deprecated**: The legacy hipSPARSE `csrgeam` routines (`hipsparseXcsrgeamNnz`, `hipsparseScsrgeam`, `hipsparseDcsrgeam`, `hipsparseCcsrgeam`, and `hipsparseZcsrgeam`) are deprecated and will be removed in a future release. Use the `csrgeam2` routines instead: `hipsparseScsrgeam2_bufferSizeExt`, `hipsparseDcsrgeam2_bufferSizeExt`, `hipsparseCcsrgeam2_bufferSizeExt`, `hipsparseZcsrgeam2_bufferSizeExt`, `hipsparseXcsrgeam2Nnz`, `hipsparseScsrgeam2`, `hipsparseDcsrgeam2`, `hipsparseCcsrgeam2`, and `hipsparseZcsrgeam2`.

#### rocSPARSE feature highlights

The following are notable enhancements to rocSPARSE:

* **Incomplete LDLᵀ factorization**: rocSPARSE adds the `rocsparse_spildlt0` routine, which computes an incomplete LDLᵀ factorization with zero fill-in (ILDLT(0)) for symmetric real or Hermitian complex sparse matrices in CSR format. The routine supports strided batched computation for factoring multiple matrices in a single call, a common building block for preconditioning iterative sparse solvers.

* **`rocsparse_indextype_u16` index type deprecated**: The `rocsparse_indextype_u16` field of the `rocsparse_indextype` enum is deprecated in this release. Code using `rocsparse_indextype_u16` now produces deprecation warnings at compile time. Migrate to `rocsparse_indextype_i32` or `rocsparse_indextype_i64`; `rocsparse_indextype_u16` will be removed in a future release.

#### RCCL feature highlights

The following are notable enhancements to RCCL:

* **Hierarchical AllGather**: RCCL adds a hierarchical AllGather algorithm for large multi-node jobs by separating inter-node from intra-node communication, relieving the concurrency pressure that constrains the existing ring and direct algorithms across many GPUs. On AMD Instinct MI350X GPUs, hierarchical AllGather is enabled by default for multi-node configurations. To disable it, set `RCCL_HIERARCHICAL_ALLGATHER=0`.

* **Direct reduce-scatter**: RCCL adds a direct reduce-scatter algorithm for small-to-medium message sizes on AMD Instinct MI350X GPUs, as an alternative to the existing ring-based implementation. RCCL selects it automatically for multi-node reduce-scatter operations within a configurable message-size threshold.

* **Copy Engine collectives (Preview)**: RCCL now offloads collective data movement to the GPU copy engine on AMD Instinct MI355X GPUs through new Copy Engine collectives. This frees compute units during communication-bound collectives, so compute and communication can overlap. RCCL uses a batched copy path when available, falls back to multi-stream or single-stream transfers otherwise, and preserves correct behavior during HIP graph capture.

For more information, see the [RCCL section](#rccl-2-30-4) in the ROCm component changelogs.

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

### Multi-architecture GPU support

The installer provides multi-architecture support, allowing you to install ROCm components for one or more GPU architectures. This is particularly useful for heterogeneous GPU environments or when deploying across multiple systems with different GPU types.

* Install single or multiple GPU architectures in one installation.
* Autodetect GPU and install matching architecture.
* Query available and installed architectures.
* Selectively uninstall specific architectures while keeping others.

### Flexible component selection

Choose exactly which ROCm components to install, reducing installation time and disk space requirements:

* **core**: Essential runtime libraries and tools (default)
* **core-dev**: Development headers and files
* **dev-tools**: Debugging and profiling utilities
* **core-sdk**: Comprehensive SDK with libraries and development tools
* **opencl**: OpenCL runtime support

### Graphics support

Optional graphics support for Mesa and OpenGL workloads is now available. When enabled, the installer includes the `amdgpu-lib` package for graphics capabilities.

### Build and manifest information

* Display TheRock build information including commit hash, GitHub run ID, and build date
* View complete manifest of all components and their versions included in the installer
* Query components by specific GPU architecture

### Universal installer

A single installer file now supports all Linux distributions, eliminating the need to download distribution-specific builds.

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

### PyTorch might display a warning when libnuma is not installed

PyTorch might display a warning when importing on Linux if the system libnuma package is not installed on some Radeon graphics products, such as Radeon AI PRO R9600D. As a workaround, install the system libnuma package or configure the library path to use the ROCm-bundled NUMA libraries. See [GitHub issue #6485](https://github.com/ROCm/ROCm/issues/6485).

<a id="vllm-warmup-known-issue"></a>

### Significantly longer LLM warmup times on some Radeon GPUs

Significantly longer warmup times might be observed in some large language model inference workloads on AMD Radeon GPUs using vLLM versions v0.21.0 through v0.25.0. As a workaround, use a vLLM release earlier than v0.21.0 or upgrade to vLLM v0.26.0 or later, which includes a fix for this issue. See [GitHub issue #6486](https://github.com/ROCm/ROCm/issues/6486).

### SGLang default settings and some models might cause failures on Radeon GPUs

ROCm 7.14 introduces initial SGLang support for AMD Radeon GPUs. Radeon GPU users should disable AITER and unset `SGLANG_ROCM_FUSED_DECODE_MLA`, as both are enabled by default in the SGLang Docker image and might cause some workloads to fail:

```bash
export SGLANG_USE_AITER=false
export SGLANG_ROCM_FUSED_DECODE_MLA=false
```

Additionally, some models might not function correctly on Radeon GPUs, including certain Mixture-of-Experts (MoE) models (such as GPT-OSS-20B and MiniMax-M2.7) and Qwen3-ASR models. Users experiencing these issues are recommended to use the latest upstream SGLang versions, which will include the necessary fixes once they are merged. See the [SGLang environment variables reference](https://docs.sglang.io/docs/references/environment_variables#environment-variables) for more details. See [GitHub issue #6487](https://github.com/ROCm/ROCm/issues/6487).

### Lower-than-expected LLM inference performance on RDNA3 Radeon GPUs and Ryzen AI Max / Max+ Series Processors

Lower-than-expected performance might be observed in some large language model inference workloads, including vLLM FP16 decode workloads with batch sizes of 8 or greater, on AMD Radeon RX 7900 Series Graphics, AMD Radeon RX 7800 XT Graphics, and AMD Ryzen AI Max / Max+ Series Processors when using PyTorch versions earlier than 2.14. As a workaround, set the `TORCH_BLAS_PREFER_HIPBLASLT=1` environment variable to use the hipBLASLt backend. This setting becomes the default for these architectures in PyTorch 2.14. See [GitHub issue #6488](https://github.com/ROCm/ROCm/issues/6488).

### ROCprofiler-SDK SPM sessions can remain in a stale state after abrupt termination

If a Streaming Performance Monitors (SPM) session is terminated abruptly (for example, with `Ctrl+C`), KFD-side SPM resources might not be released cleanly. When this happens, the KFD-side SPM resources can remain in a stale state, which might cause subsequent SPM profiling sessions to hang or fail to start with the error `Unable to acquire KFD thread: 4096`. To recover, if the profiling process is still running, terminate it manually. If the error persists, a system reboot is currently required to restore the GPU to a usable state for SPM profiling. This issue is under active investigation for a fix. See [GitHub issue #6489](https://github.com/ROCm/ROCm/issues/6489).

### ROCm Compute Profiler might report inflated Avg values with per_kernel normalization

When using ROCm Compute Profiler with `per_kernel` normalization, the reported Avg value for certain normalized metrics might be incorrectly inflated and can exceed the corresponding Min and Max values. This issue affects analysis results only. As a workaround, use an alternative normalization unit (`-n`/`--normal-unit`) until a fix is available. See [GitHub issue #6490](https://github.com/ROCm/ROCm/issues/6490).

### rocALUTION and hipTensor have no dedicated HPC Expansion tarball

The `amdrocm-hpc` meta-package installs rocALUTION and hipTensor, but there is no dedicated HPC Expansion tarball for tarball-based installations. The standard ROCm tarballs include both libraries. See [GitHub issue #6491](https://github.com/ROCm/ROCm/issues/6491).

### HIP SPIR-V kernels might segfault on first launch

HIP kernels compiled with the SPIR-V target (`--offload-arch=amdgcnspirv`) might segfault on first kernel launch at `hipLaunchKernel`. The failure affects both library-level workloads such as rocBLAS and standalone HIP applications built against the SPIR-V offload bundle. Applications compiled for a native GPU architecture target are not affected. As a workaround, compile using a direct GPU architecture target instead of `--offload-arch=amdgcnspirv`. See [GitHub issue #6492](https://github.com/ROCm/ROCm/issues/6492).

### RCCL might show degraded performance on multi-node configurations

RCCL operations with message sizes in the 64 MB to 512 MB range might show suboptimal performance on multi-node, multi-threaded configurations. This issue affects the packaged binary distribution and might severely impact production workloads. Known affected workloads include Llama 3 405B and JAX stack; additional workloads might also be affected. Single-node (scale-up) configurations are not affected.
To work around this issue, recompile RCCL from the ROCm 7.14 source with fault injection disabled (see Building and installing RCCL). You can either set the option in the RCCL CMake file:

```cmake
option(FAULT_INJECTION         "Enable fault injection"           OFF)
```

Alternatively, add the following CMake flag during compilation:

```text
-DFAULT_INJECTION=OFF
```

See [GitHub issue #6493](https://github.com/ROCm/ROCm/issues/6493).

### AMD SMI NIC telemetry supports Pollara 400 adapters only

In ROCm 10.0.0, AMD SMI NIC telemetry only supports AMD AI NIC Pollara 400 adapters. Broadcom NIC support is planned for a future release. See [GitHub issue #6497](https://github.com/ROCm/ROCm/issues/6497).

## ROCm resolved issues

The following notable issues have been fixed in ROCm 10.0.0.

### ROCm Compute Profiler failed when profiling bash scripts or commands

Previously, running a bash script or command as a target for ROCm Compute Profiler failed because bash overwrote the required environment variables.

### LLVM-based compilers failed when compiling half-precision vector operations

Previously, LLVM-based compilers failed, returning the `Failed to find subregs!` error message in `SIInstrInfo::copyPhysReg`, when compiling half-precision vector operations with optimization enabled at levels `-O1` to `-O3`.

### hipBLAS test suites returned non-zero exit codes on Windows

Previously, when using hipBLAS on Windows, the test suites returned non-zero exit codes even when all mathematical correctness tests passed, blocking automated testing workflows.

### Illegal memory address error when using placement new with device function returns

Previously, HIP kernels that used placement new to construct objects in `hipMalloc`-allocated device memory crashed with a `hipErrorIllegalAddress` error when a `__device__` function return value was passed as the constructor argument for non-trivially copyable types.

### GPU kernels failed to launch in ASan builds with large thread counts

Previously, when building GPU libraries with ASan enabled, kernels configured with large thread counts failed to launch with an `HSA_STATUS_ERROR_INVALID_ISA` error.

### ASan prevented multi-architecture HIP binary builds from launching

Previously, HIP applications built with ASan enabled and targeting multiple GPU architectures failed to launch with `RuntimeError: .hipFatBinSegment size N is not a multiple of wrapper size (24)` and `RuntimeError: Unexpected magic 0x00000000 at wrapper i` error messages.

### ROCm Systems Profiler overwrote ROCPD output after process re-attachment

Previously, when using `rocprof-sys-attach` to re-attach to a previously profiled process, the ROCPD output database files (`.db`) were written to the initial session's output directory instead of a new timestamped directory.

### hipCUB DeviceMerge large-size stress test failed with out-of-memory error on gfx1150

Previously, on gfx1150 APUs, the hipCUB DeviceMerge large-size stress test (`MergeLargeSizeIterators`) failed with an out-of-memory error when running ROCm 7.12.0. Standard DeviceMerge test cases were not affected.

### HIP kernel launch limit caused failures for some models

Previously, with PyTorch 2.10, some models hit the HIP kernel launch limit of 2³² kernel launches within a single process, causing HIP kernel launch errors. One known affected model was `black-forest-labs/flux`.

### Non-deterministic GPU memory faults when passing large data structures on MI300X

Previously, applications running on AMD Instinct MI300X GPUs that passed large, complex data structures between device functions using scratch memory encountered non-deterministic GPU memory access faults and became unresponsive when compiler optimizations minimized the number of copy operations.

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
