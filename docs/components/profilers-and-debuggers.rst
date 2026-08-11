.. meta::
   :description: AMD ROCm profiling and debugging tools for GPU application performance analysis and fault diagnosis.
   :keywords: profiler, debugger, rocprofiler, ROCgdb, ROCm, performance, tracing

**********************************
ROCm profiling and debugging tools
**********************************

ROCm profiling and debugging tools help you measure GPU application performance,
identify bottlenecks, and diagnose execution faults.

* :doc:`ROCm Compute Profiler <rocprofiler-compute:index>`
  (rocprofiler-compute) -- Kernel-level profiling for machine learning and
  high performance computing (HPC) workloads.

* :doc:`ROCm Systems Profiler <rocprofiler-systems:index>`
  (rocprofiler-systems) -- Comprehensive profiling and tracing of applications
  running on the CPU or the CPU and GPU.

* :doc:`ROCprofiler-SDK <rocprofiler-sdk:index>` -- Toolkit for developing
  analysis tools for profiling and tracing GPU compute applications.

* :doc:`ROCdbgapi <rocdbgapi:index>` -- ROCm debugger API library.

* :doc:`ROCm Debugger <rocgdb:index>` (ROCgdb) -- Source-level debugger for
  Linux, based on the GNU Debugger (GDB).

* :doc:`ROCR Debug Agent <rocr_debug_agent:index>` -- Prints the state of all
  AMD GPU wavefronts that caused a queue error by sending a SIGQUIT signal to
  the process while the program is running.

.. note::

   `ROCprof Compute Viewer <https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/latest/>`_ is a tool for visualizing and analyzing GPU thread trace data collected using :doc:`rocprofv3 <rocprofiler-sdk:index>`. Note that ROCprof Compute Viewer is in an early access state. Running production workloads is not recommended.
