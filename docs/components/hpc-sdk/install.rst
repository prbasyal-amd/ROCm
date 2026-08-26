:selector-toc2: Installation environment
:selector-toc2-icon: fa-solid fa-computer

.. meta::
   :description: Install ROCm to run high-performance computing (HPC) workloads.
   :keywords: ROCm, HPC, install, installation, Linux, AMD Instinct

.. _hpc-install:

********************
Install ROCm HPC SDK
********************

AMD ROCm HPC SDK provides high-performance computing libraries and tools for
AMD Instinct GPUs. This guide walks you through installing the HPC SDK
alongside ROCm installation on a supported Linux distribution.

The ROCm for HPC applications and containers run on a standard ROCm
installation. See the :ref:`Compatibility matrix <compat-matrix>` for details
on supported hardware and operating systems.

The HPC application containers are published through
`AMD InfinityHub-CI <https://github.com/amd/InfinityHub-CI>`_. Each container
provides parameters to specify source code branches and release versions of ROCm,
OpenMPI, UCX, and Ubuntu.

.. selector:: Device family
   :key: fam

   .. selector-option:: All
      :value: all
      :width: 6

   .. selector-option:: AMD Instinct™
      :value: instinct
      :width: 6
      :toc-label: AMD Instinct

.. datatemplate:yaml:: /data/gpus.yaml
   :template: gpu-selector.rst.jinja

.. datatemplate:yaml:: /data/gpus.yaml
   :template: misc/os-selector-no-windows.rst.jinja

.. selector:: ROCm version
   :key: rocm-ver

   .. selector-option:: 10.0.0
      :value: 10.0.0

   .. selector-option:: 7.14.0
      :value: 7.14.0

.. selector:: Installation method
   :show-cond: os=ubuntu os=debian
   :key: i

   .. selector-option:: apt
      :value: pkgman
      :width: 6

   .. selector-option:: Tarball
      :value: tar
      :width: 6

.. selector:: Installation method
   :show-cond: os=rhel os=oracle-linux os=rocky-linux
   :key: i

   .. selector-option:: dnf
      :value: pkgman
      :width: 6

   .. selector-option:: Tarball
      :value: tar
      :width: 6

.. selector:: Installation method
   :show-cond: os=sles
   :key: i

   .. selector-option:: zypper
      :value: pkgman
      :width: 6

   .. selector-option:: Tarball
      :value: tar
      :width: 6

----

Before installing the HPC SDK, make sure your system meets the ROCm hardware,
software, and driver requirements. For instructions, see :ref:`Install AMD ROCm <rocm-install-selector>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

HPC SDK includes `hipTensor <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiptensor>`_ and `rocALUTION <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocalution>`_ packaged as part of the installation.

Install HPC SDK
---------------

.. include:: ./include/rocm10.0.0-install.rst

.. include:: ./include/rocm7.14.0-install.rst

Uninstall HPC SDK
----------------------

.. include:: ./include/rocm10.0.0-uninstall.rst

.. include:: ./include/rocm7.14.0-uninstall.rst
