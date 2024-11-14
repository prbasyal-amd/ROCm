# ROCm 6.2.4 release notes

The release notes provide a summary of notable changes since the previous ROCm release.

- [Release highlights](#release-highlights)

- [Operating system and hardware support changes](#operating-system-and-hardware-support-changes)

- [ROCm components versioning](#rocm-components)

- [Detailed component changes](#detailed-component-changes)

- [ROCm known issues](#rocm-known-issues)

- [ROCm upcoming changes](#rocm-upcoming-changes)

```{note}
If you’re using Radeon™ PRO or Radeon GPUs in a workstation setting with a
display connected, continue to use ROCm 6.2.3. See the [Use ROCm on Radeon
GPUs](https://rocm.docs.amd.com/projects/radeon/en/latest/index.html)
documentation to verify compatibility and system requirements.
```

## Release highlights

The following are notable new features and improvements in ROCm 6.2.4. For changes to individual components, see
[Detailed component changes](#detailed-component-changes).

#### ROCm documentation updates

ROCm documentation continues to be updated to provide clearer and more comprehensive guidance for
a wider variety of user needs and use cases.

* Added a new GPU cluster networking guide. See
  [Cluster network performance validation for AMD Instinct accelerators](https://rocm.docs.amd.com/projects/gpu-cluster-networking/en/docs-6.2.4/index.html).
  This documentation provides guidelines on validating network configurations
  in single-node and multi-node environments to attain optimal speed and bandwidth
  in AMD Instinct-powered clusters.

* Updated the HIP runtime documentation.

  * Added a new section on how to use [HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4/how-to/hipgraph.html).

  * Added a new section about the [Stream ordered memory allocator (SOMA)](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4/how-to/stream_ordered_allocator.html).

  * Updated the [Porting CUDA driver API](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4/how-to/hip_porting_driver_api.html) section.

* Updated the [Post-installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/install/post-install.html)
  with guidance on using the `update-alternatives` utility and environment modules to help you manage multiple ROCm
  versions and streamline PATH configuration.

* Updated the [LLM inference performance validation on AMD Instinct
  MI300X](https://rocm.docs.amd.com/en/docs-6.2.4/how-to/performance-validation/mi300x/vllm-benchmark.html)
  documentation with more detailed guidance, new models, and the `float8` data type.

## Operating system and hardware support changes

ROCm 6.2.4 adds support for the [AMD Radeon PRO V710](https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v710.html) GPU for compute workloads. See 
[Supported GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/reference/system-requirements.html#supported-gpus)
for more information.

This release maintains the same operating system support as 6.2.2.

## ROCm components

The following table lists the versions of ROCm components for ROCm 6.2.4, including any version changes from 6.2.2 to 6.2.4.

Click the component's updated version to go to a detailed list of its changes. Click <i class="fab fa-github fa-lg"></i> to go to the component's source code on GitHub.

<div class="pst-scrollable-table-container">
    <table id="rocm-rn-components" class="table">
        <thead>
            <tr>
                <th>Category</th>
                <th>Group</th>
                <th>Name</th>
                <th>Version</th>
                <th></th>
            </tr>
        </thead>
        <colgroup>
            <col span="1">
            <col span="1">
        </colgroup>
        <tbody class="rocm-components-libs rocm-components-ml">
            <tr>
                <th rowspan="8">Libraries</th>
                <th rowspan="8">Machine learning and computer vision</th>
                <td><a href="https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.2.4">Composable Kernel</a>
                </td>
                <td>1.1.0</td>
                <td><a href="https://github.com/ROCm/composable_kernel/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/AMDMIGraphX/en/docs-6.2.4">MIGraphX</a></td>
                <td>2.10</td>
                <td><a href="https://github.com/ROCm/AMDMIGraphX/releases/"><i class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/MIOpen/en/docs-6.2.4">MIOpen</a></td>
                <td>3.2.0</td>
                <td><a href="https://github.com/ROCm/MIOpen/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/MIVisionX/en/docs-6.2.4">MIVisionX</a></td>
                <td>3.0.0</td>
                <td><a href="https://github.com/ROCm/MIVisionX/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocAL/en/docs-6.2.4">rocAL</a></td>
                <td>2.0.0</td>
                <td><a href="https://github.com/ROCm/rocAL/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocDecode/en/docs-6.2.4">rocDecode</a></td>
                <td>0.6.0</td>
                <td><a href="https://github.com/ROCm/rocDecode/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocPyDecode/en/docs-6.2.4">rocPyDecode</a></td>
                <td>0.1.0</td>
                <td><a href="https://github.com/ROCm/rocPyDecode/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rpp/en/docs-6.2.4">RPP</a></td>
                <td>1.8.0</td>
                <td><a href="https://github.com/ROCm/rpp/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
        <tbody class="rocm-components-libs rocm-components-communication">
            <tr>
                <th rowspan="1"></th>
                <th rowspan="1">Communication</th>
                <td><a href="https://rocm.docs.amd.com/projects/rccl/en/docs-6.2.4">RCCL</a></td>
                <td>2.20.5</td>
                <td><a href="https://github.com/ROCm/rccl/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
        <tbody class="rocm-components-libs rocm-components-math tbody-reverse-zebra">
            <tr>
                <th rowspan="16"></th>
                <th rowspan="16">Math</th>
                <td><a href="https://rocm.docs.amd.com/projects/hipBLAS/en/docs-6.2.4">hipBLAS</a></td>
                <td>2.2.0</td>
                <td><a href="https://github.com/ROCm/hipBLAS/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-6.2.4">hipBLASLt</a></td>
                <td>0.8.0</td>
                <td><a href="https://github.com/ROCm/hipBLASLt/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/hipFFT/en/docs-6.2.4">hipFFT</a></td>
                <td>1.0.15&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/hipFFT/blob/docs/6.2.4/CHANGELOG.md">1.0.16</a></td>
                <td><a href="https://github.com/ROCm/hipFFT/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/hipfort/en/docs-6.2.4">hipfort</a></td>
                <td>0.4.0</td>
                <td><a href="https://github.com/ROCm/hipfort/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/hipRAND/en/docs-6.2.4">hipRAND</a></td>
                <td>2.11.0&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/hipRAND/blob/docs/6.2.4/CHANGELOG.md">2.11.1</a></td>
                <td><a href="https://github.com/ROCm/hipRAND/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/hipSOLVER/en/docs-6.2.4">hipSOLVER</a></td>
                <td>2.2.0</td>
                <td><a href="https://github.com/ROCm/hipSOLVER/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/hipSPARSE/en/docs-6.2.4">hipSPARSE</a></td>
                <td>3.1.1</td>
                <td><a href="https://github.com/ROCm/hipSPARSE/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/hipSPARSELt/en/docs-6.2.4">hipSPARSELt</a></td>
                <td>0.2.1</td>
                <td><a href="https://github.com/ROCm/hipSPARSELt/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocALUTION/en/docs-6.2.4">rocALUTION</a></td>
                <td>3.2.0&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/rocALUTION/blob/docs/6.2.4/CHANGELOG.md">3.2.1</a></td>
                <td><a href="https://github.com/ROCm/rocALUTION/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocBLAS/en/docs-6.2.4">rocBLAS</a></td>
                <td>4.2.1&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/rocBLAS/blob/docs/6.2.4/CHANGELOG.md">4.2.4</a></td>
                <td><a href="https://github.com/ROCm/rocBLAS/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocFFT/en/docs-6.2.4">rocFFT</a></td>
                <td>1.0.29&nbsp;&Rightarrow;&nbsp;<a href="#rocfft-1-0-30">1.0.30</a></td>
                <td><a href="https://github.com/ROCm/rocFFT/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocRAND/en/docs-6.2.4">rocRAND</a></td>
                <td>3.1.0&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/rocRAND/blob/docs/6.2.4/CHANGELOG.md">3.1.1</a></td>
                <td><a href="https://github.com/ROCm/rocRAND/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocSOLVER/en/docs-6.2.4">rocSOLVER</a></td>
                <td>3.26.0&nbsp;&Rightarrow;&nbsp;<a href="#rocsolver-3-26-2">3.26.2</a></td>
                <td><a href="https://github.com/ROCm/rocSOLVER/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocSPARSE/en/docs-6.2.4">rocSPARSE</a></td>
                <td>3.2.0&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/rocSPARSE/blob/docs/6.2.4/CHANGELOG.md">3.2.1</a></td>
                <td><a href="https://github.com/ROCm/rocSPARSE/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocWMMA/en/docs-6.2.4">rocWMMA</a></td>
                <td>1.5.0</td>
                <td><a href="https://github.com/ROCm/rocWMMA/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://github.com/ROCm/Tensile">Tensile</a></td>
                <td>4.41.0</td>
                <td><a href="https://github.com/ROCm/tensile/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
        <tbody class="rocm-components-libs rocm-components-primitives tbody-reverse-zebra">
            <tr>
                <th rowspan="4"></th>
                <th rowspan="4">Primitives</th>
                <td><a href="https://rocm.docs.amd.com/projects/hipCUB/en/docs-6.2.4">hipCUB</a></td>
                <td>3.2.0&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/hipCUB/blob/docs/6.2.4/CHANGELOG.md">3.2.1</a></td>
                <td><a href="https://github.com/ROCm/hipCUB/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/hipTensor/en/docs-6.2.4">hipTensor</a></td>
                <td>1.3.0</td>
                <td><a href="https://github.com/ROCm/hipTensor/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.2.4">rocPRIM</a></td>
                <td>3.2.1&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/rocPRIM/blob/docs/6.2.4/CHANGELOG.md">3.2.2</a></td>
                <td><a href="https://github.com/ROCm/rocPRIM/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocThrust/en/docs-6.2.4">rocThrust</a></td>
                <td>3.1.0&nbsp;&Rightarrow;&nbsp;<a href="https://github.com/ROCm/rocThrust/blob/docs/6.2.4/CHANGELOG.md">3.1.1</a></td>
                <td><a href="https://github.com/ROCm/rocThrust/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
        <tbody class="rocm-components-tools rocm-components-system tbody-reverse-zebra">
            <tr>
                <th rowspan="6">Tools</th>
                <th rowspan="6">System management</th>
                <td><a href="https://rocm.docs.amd.com/projects/amdsmi/en/docs-6.2.4">AMD SMI</a></td>
                <td>24.6.3&nbsp;&Rightarrow;&nbsp;<a href="#amd-smi-24-6-3">24.6.3</a></td>
                <td><a href="https://github.com/ROCm/amdsmi/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocminfo/en/docs-6.2.4">rocminfo</a></td>
                <td>1.0.0</td>
                <td><a href="https://github.com/ROCm/rocminfo/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rdc/en/docs-6.2.4">ROCm Data Center Tool</a></td>
                <td>0.3.0</td>
                <td><a href="https://github.com/ROCm/rdc/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocm_smi_lib/en/docs-6.2.4">ROCm SMI</a></td>
                <td>7.3.0</td>
                <td><a href="https://github.com/ROCm/rocm_smi_lib/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/docs-6.2.4">ROCm Validation Suite</a></td>
                <td>1.0.0</td>
                <td><a href="https://github.com/ROCm/ROCmValidationSuite/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
        <tbody class="rocm-components-tools rocm-components-perf">
            <tr>
                <th rowspan="6"></th>
                <th rowspan="6">Performance</th>
                <td><a href="https://rocm.docs.amd.com/projects/omniperf/en/docs-6.2.4">Omniperf</a></td>
                <td>2.0.1</td>
                <td><a href="https://github.com/ROCm/omniperf/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/omnitrace/en/docs-6.2.4">Omnitrace</a></td>
                <td>1.11.2</td>
                <td><a href="https://github.com/ROCm/omnitrace/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocm_bandwidth_test/en/docs-6.2.4">ROCm Bandwidth
                        Test</a></td>
                <td>1.4.0</td>
                <td><a href="https://github.com/ROCm/rocm_bandwidth_test/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocprofiler/en/docs-6.2.4/">ROCProfiler</a></td>
                <td>2.0.0</td>
                <td><a href="https://github.com/ROCm/ROCProfiler/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-6.2.4">ROCprofiler-SDK</a></td>
                <td>0.4.0</td>
                <td><a href="https://github.com/ROCm/rocprofiler-sdk/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr >
                <td><a href="https://rocm.docs.amd.com/projects/roctracer/en/docs-6.2.4/">ROCTracer</a></td>
                <td>4.1.0</td>
                <td><a href="https://github.com/ROCm/ROCTracer/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
        <tbody class="rocm-components-tools rocm-components-dev">
            <tr>
                <th rowspan="5"></th>
                <th rowspan="5">Development</th>
                <td><a href="https://rocm.docs.amd.com/projects/HIPIFY/en/docs-6.2.4/">HIPIFY</a></td>
                <td>18.0.0</td>
                <td><a href="https://github.com/ROCm/HIPIFY/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/ROCdbgapi/en/docs-6.2.4">ROCdbgapi</a></td>
                <td>0.76.0</td>
                <td><a href="https://github.com/ROCm/ROCdbgapi/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/ROCmCMakeBuildTools/en/docs-6.2.4/">ROCm CMake</a></td>
                <td>0.13.0</td>
                <td><a href="https://github.com/ROCm/rocm-cmake/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/ROCgdb/en/docs-6.2.4">ROCm Debugger (ROCgdb)</a>
                </td>
                <td>14.2</td>
                <td><a href="https://github.com/ROCm/ROCgdb/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/rocr_debug_agent/en/docs-6.2.4">ROCr Debug Agent</a>
                </td>
                <td>2.0.3</td>
                <td><a href="https://github.com/ROCm/rocr_debug_agent/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
        <tbody class="rocm-components-compilers tbody-reverse-zebra">
            <tr>
                <th rowspan="2" colspan="2">Compilers</th>
                <td><a href="https://rocm.docs.amd.com/projects/HIPCC/en/docs-6.2.4">HIPCC</a></td>
                <td>1.1.1</td>
                <td><a href="https://github.com/ROCm/llvm-project/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.4">llvm-project</a></td>
                <td>18.0.0</td>
                <td><a href="https://github.com/ROCm/llvm-project/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
        <tbody class="rocm-components-runtimes tbody-reverse-zebra">
            <tr>
                <th rowspan="2" colspan="2">Runtimes</th>
                <td><a href="https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4">HIP</a></td>
                <td>6.2.4</a></td>
                <td><a href="https://github.com/ROCm/HIP/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
            <tr>
                <td><a href="https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4">ROCr Runtime</a></td>
                <td>1.14.0</td>
                <td><a href="https://github.com/ROCm/ROCR-Runtime/releases/"><i
                            class="fab fa-github fa-lg"></i></a></td>
            </tr>
        </tbody>
    </table>
</div>

## Detailed component changes

The following sections describe key changes to ROCm components.

### **AMD SMI** (24.6.3)

#### Resolved issues

* Fixed support for the API calls `amdsmi_get_gpu_process_isolation` and
  `amdsmi_clean_gpu_local_data`, along with the `amd-smi set
  --process-isolation <0 or 1>` command. See issue
  [#3500](https://github.com/ROCm/ROCm/issues/3500) on GitHub.

### **rocFFT** (1.0.30)

#### Optimized

* Implemented 1D kernels for factorizable sizes greater than 1024 and less than 2048.

#### Resolved issues

* Fixed plan creation failure on some even-length real-complex transforms that use Bluestein's algorithm.

### **rocSOLVER** (3.26.2)

#### Resolved issues

* Fixed synchronization issue in STEIN.

## ROCm known issues

ROCm known issues are tracked on [GitHub](https://github.com/ROCm/ROCm/labels/Verified%20Issue).
Known issues related to individual components are listed in the [Detailed component changes](#detailed-component-changes)
section.

## ROCm upcoming changes

The following changes to the ROCm software stack are anticipated for future releases.

### rocm-llvm-alt

The `rocm-llvm-alt` package will be removed in an upcoming release. Users relying on the functionality provided by the closed-source compiler should transition to the open-source compiler. Once the `rocm-llvm-alt` package is removed, any compilation requesting functionality provided by the closed-source compiler will result in a Clang warning: "*[AMD] proprietary optimization compiler has been removed*".

### rccl-rdma-sharp-plugins

The RCCL plugin package, `rccl-rdma-sharp-plugins`, will be removed in an upcoming ROCm release. 
