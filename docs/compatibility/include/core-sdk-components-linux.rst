.. matrix::
   :show-cond: os=ubuntu os=debian os=rhel os=oracle-linux os=rocky-linux os=sles

   .. matrix-head::

      .. matrix-row::
         :header:

         .. matrix-cell:: Component group

         .. matrix-cell:: Component name

   .. matrix-row::

      .. matrix-cell:: Math and compute libraries
         :rowspan: 18

      .. matrix-cell::
         :show-cond: fam=instinct fam=radeon

         `Composable Kernel 1.2.0 <https://rocm.docs.amd.com/projects/composable_kernel/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `hipBLAS 3.6.0 <https://rocm.docs.amd.com/projects/hipBLAS/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `hipBLASLt 1.4.1 <https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `hipCUB 4.6.0 <https://rocm.docs.amd.com/projects/hipCUB/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `hipFFT 1.0.25 <https://rocm.docs.amd.com/projects/hipFFT/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `hipRAND 3.4.0 <https://rocm.docs.amd.com/projects/hipRAND/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `hipSOLVER 3.6.0 <https://rocm.docs.amd.com/projects/hipSOLVER/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `hipSPARSE 4.7.0 <https://rocm.docs.amd.com/projects/hipSPARSE/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi350p gpu=mi325x gpu=mi300x gpu=mi300a

         `hipSPARSELt 0.2.9 <https://rocm.docs.amd.com/projects/hipSPARSELt/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `MIOpen 3.6.0 <https://rocm.docs.amd.com/projects/MIOpen/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `rocBLAS 5.6.0 <https://rocm.docs.amd.com/projects/rocBLAS/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `rocFFT 1.0.39 <https://rocm.docs.amd.com/projects/rocFFT/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `rocPRIM 4.6.0 <https://rocm.docs.amd.com/projects/rocPRIM/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `rocRAND 5.0.0 <https://rocm.docs.amd.com/projects/rocRAND/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `rocSOLVER 3.36.0 <https://rocm.docs.amd.com/projects/rocSOLVER/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `rocSPARSE 5.0.0 <https://rocm.docs.amd.com/projects/rocSPARSE/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `rocThrust 4.6.0 <https://rocm.docs.amd.com/projects/rocThrust/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell::

         `rocWMMA 2.2.1 <https://rocm.docs.amd.com/projects/rocWMMA/en/docs-10.0.0/index.html>`__

   .. matrix-row::

      .. matrix-cell:: Communication libraries
         :rowspan: 2
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi350p gpu=mi325x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210 gpu=ai-r9700s gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060 gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=rx-7900-xtx gpu=rx-7900-gre gpu=w7700 gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700 gpu=rx-7600

      .. matrix-cell:: Communication libraries
         :rowspan: 1
         :show-cond: gpu=mi100 gpu=v710 gpu=v620 gpu=w6800

      .. matrix-cell:: Communication libraries
         :rowspan: 1
         :show-cond: fam=ryzen

      .. matrix-cell::

         `RCCL 2.30.7 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rccl>`__

   .. matrix-row::
      :show-cond: gpu=mi355x gpu=mi350x gpu=mi350p gpu=mi325x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210 gpu=ai-r9700s gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060 gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=rx-7900-xtx gpu=rx-7900-gre gpu=w7700 gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700 gpu=rx-7600

      .. matrix-cell::

         `rocSHMEM 3.6.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocshmem>`__

   .. matrix-row::

      .. matrix-cell:: Media libraries
         :rowspan: 2
         :show-cond: fam=instinct fam=radeon

      .. matrix-cell:: Media libraries
         :rowspan: 2
         :show-cond: gfx=gfx1150 gfx=gfx1151 gfx=gfx1152 gfx=gfx1153

      .. matrix-cell::
         :show-cond: fam=instinct fam=radeon

         `rocDecode 1.9.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocdecode>`__

      .. matrix-cell::
         :show-cond: gfx=gfx1150 gfx=gfx1151 gfx=gfx1152 gfx=gfx1153

         `rocDecode 1.9.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocdecode>`__

   .. matrix-row::

      .. matrix-cell::
         :show-cond: fam=instinct fam=radeon

         `rocJPEG 1.7.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocjpeg>`__

      .. matrix-cell::
         :show-cond: gfx=gfx1150 gfx=gfx1151 gfx=gfx1152 gfx=gfx1153

         `rocJPEG 1.7.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocjpeg>`__

   .. matrix-row::

      .. matrix-cell:: Storage libraries
         :show-cond: fam=instinct

      .. matrix-cell::
         :show-cond: fam=instinct

         `hipFile 0.4.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/hipFile>`__

   .. matrix-row::

      .. matrix-cell:: Runtimes and compilers
         :rowspan: 5
         :show-cond: fam=instinct fam=radeon

      .. matrix-cell:: Runtimes and compilers
         :rowspan: 4
         :show-cond: fam=ryzen

      .. matrix-cell::

         `HIP 10.0.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/hip>`__

   .. matrix-row::

      .. matrix-cell::

         `HIPIFY 10.0.0 <https://github.com/ROCm/HIPIFY/tree/therock-10.0>`__

   .. matrix-row::

      .. matrix-cell::

         `LLVM 24.0.0 <https://github.com/ROCm/llvm-project/tree/therock-10.0>`__

   .. matrix-row::
      :show-cond: fam=instinct fam=radeon

      .. matrix-cell::

         `ROCr Runtime 1.21.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocr-runtime>`__

   .. matrix-row::

      .. matrix-cell::

         `SPIRV-LLVM-Translator 24.0.0 <https://github.com/ROCm/SPIRV-LLVM-Translator/tree/therock-10.0>`__

   .. matrix-row::

      .. matrix-cell:: Profiling and debugging tools
         :rowspan: 6
         :show-cond: fam=instinct

      .. matrix-cell:: Profiling and debugging tools
         :rowspan: 4
         :show-cond: fam=radeon

      .. matrix-cell:: Profiling and debugging tools
         :rowspan: 3
         :show-cond: gfx=gfx1150 gfx=gfx1151 gfx=gfx1152 gfx=gfx1153

      .. matrix-cell::
         :show-cond: fam=instinct

         `ROCm Compute Profiler (rocprofiler-compute) 3.8.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocprofiler-compute>`__

      .. matrix-cell::
         :show-cond: fam=radeon

         `ROCprofiler-SDK 1.3.5 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocprofiler-sdk>`__

      .. matrix-cell::
         :show-cond: gfx=gfx1150 gfx=gfx1151 gfx=gfx1152 gfx=gfx1153

         `ROCm Compute Profiler (rocprofiler-compute) 3.8.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocprofiler-compute>`__

   .. matrix-row::
      :show-cond: fam=instinct

      .. matrix-cell::

         `ROCm Systems Profiler (rocprofiler-systems) 1.8.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocprofiler-systems>`__

   .. matrix-row::
      :show-cond: gfx=gfx1150 gfx=gfx1151 gfx=gfx1152 gfx=gfx1153

      .. matrix-cell::

         `ROCm Systems Profiler (rocprofiler-systems) 1.8.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocprofiler-systems>`__

   .. matrix-row::
      :show-cond: fam=instinct

      .. matrix-cell::

         `ROCprofiler-SDK 1.3.5 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocprofiler-sdk>`__

   .. matrix-row::
      :show-cond: gfx=gfx1150 gfx=gfx1151 gfx=gfx1152 gfx=gfx1153

      .. matrix-cell::

         `ROCprofiler-SDK 1.3.5 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocprofiler-sdk>`__

   .. matrix-row::
      :show-cond: fam=instinct fam=radeon

      .. matrix-cell::

         `ROCdbgapi 0.80.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocdbgapi>`__

   .. matrix-row::
      :show-cond: fam=instinct fam=radeon

      .. matrix-cell::

         `ROCm Debugger (ROCgdb) 16.3 <https://github.com/ROCm/ROCgdb/tree/therock-10.0>`__

   .. matrix-row::
      :show-cond: fam=instinct fam=radeon

      .. matrix-cell::

         `ROCR Debug Agent 2.1.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocr-debug-agent>`__

   .. matrix-row::

      .. matrix-cell:: Control and monitoring tools
         :rowspan: 3

      .. matrix-cell::
         :show-cond: fam=instinct fam=radeon

         `AMD SMI 27.0.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/amdsmi>`__

   .. matrix-row::

      .. matrix-cell::

         `rocminfo 1.0.0 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rocminfo>`__

   .. matrix-row::
      :show-cond: fam=instinct

      .. matrix-cell::

         `ROCm Data Center Tool (RDC) 1.3.1 <https://github.com/ROCm/rocm-systems/tree/therock-10.0/projects/rdc>`__
