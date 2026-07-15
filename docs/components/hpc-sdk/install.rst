:selector-toc2: Installation environment
:selector-toc2-icon: fa-solid fa-computer

.. meta::
   :description: Install ROCm to run high-performance computing (HPC) workloads.
   :keywords: ROCm, HPC, install, installation, Linux, AMD Instinct

.. _hpc-install:

***********************
Install ROCm HPC-SDK
***********************

AMD ROCm HPC-SDK provides high-performance computing libraries and tools for AMD GPU
architectures. This guide walks you through installing the HPC-SDK alongside ROCm installation
on a supported Linux distribution.

The ROCm for HPC applications and containers run on a standard ROCm installation.
Install ROCm on a supported Linux distribution before running any of the HPC
applications under the HPC application catalog.

* :doc:`/install/rocm`

* See the
  :ref:`Compatibility matrix <compat-matrix>`
  for details on supported hardware and operating systems.

The HPC application containers are published through
`AMD InfinityHub-CI <https://github.com/amd/InfinityHub-CI>`_. Each container
provides parameters to specify source code branches and release versions of ROCm,
OpenMPI, UCX, and Ubuntu.

.. selector:: Device family
   :key: fam

   .. selector-option:: All
      :value: all w=compute
      :width: 6

   .. selector-option:: AMD Instinct™
      :value: instinct w=compute
      :width: 6
      :toc-label: AMD Instinct

.. include:: /install/include/gpu-selector.rst

.. selected:: fam=instinct

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi355x gpu=mi350x gpu=mi325x

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 20%

      .. selector-option:: Debian
         :value: debian
         :width: 20%

      .. selector-option:: RHEL
         :value: rhel
         :width: 20%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 20%

      .. selector-option:: SLES
         :value: sles
         :width: 20%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi350p

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 25%

      .. selector-option:: Debian
         :value: debian
         :width: 25%

      .. selector-option:: RHEL
         :value: rhel
         :width: 25%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 25%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi300x

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: Debian
         :value: debian
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 4

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 4

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi300a

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 20%

      .. selector-option:: Debian
         :value: debian
         :width: 20%

      .. selector-option:: RHEL
         :value: rhel
         :width: 20%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 20%

      .. selector-option:: SLES
         :value: sles
         :width: 20%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi250x gpu=mi250

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 25%

      .. selector-option:: Debian
         :value: debian
         :width: 25%

      .. selector-option:: RHEL
         :value: rhel
         :width: 25%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 25%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi210

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi100

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

.. selector:: Linux distribution
   :key: os
   :show-cond: fam=radeon

   .. selector-option:: Ubuntu
      :value: ubuntu
      :width: 6

   .. selector-option:: RHEL
      :value: rhel
      :width: 6
      :toc-label: Red Hat Enterprise Linux

.. selector:: Operating system
   :key: os
   :show-cond: fam=ryzen

   .. selector-option:: Ubuntu
      :value: ubuntu
      :width: 12

.. selected:: fam=all

   .. selector:: Operating system
      :key: os

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: Debian
         :value: debian
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 4

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 4

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

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

Before installing the HPC-SDK, make sure your system meets the ROCm hardware,
software, and driver requirements. For instructions, see :ref:`Install AMD ROCm <rocm-install-selector>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

HPC-SDK includes `hipTensor <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiptensor>`_ and `rocALUTION <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocalution>`_ packaged as part of the installation.

Install HPC-SDK
---------------

.. selected:: i=tar

   The standard ROCm tarball installation includes the HPC-SDK. No
   additional steps are required. For details on ROCm tarball installation,
   refer to :ref:`Install AMD ROCm <rocm-install>` and select Tarball
   installation method from installation environment selector.

.. selected:: i=pkgman

   1. :doc:`Install ROCm </install/rocm>`. Remember to complete the :ref:`ROCm
      installation prerequisites <rocm-prerequisites>` to install dependencies
      and configure GPU access permissions.

   .. selected:: fam=all

      2. Use the following command to install HPC-SDK for all GPU architectures:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14 amdrocm-hpc-sdk7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14 amdrocm-hpc-sdk7.14

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14 amdrocm-hpc-sdk7.14

   .. selected:: gfx=gfx950

      2. Use the following command to install HPC-SDK for your ``gfx950`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx950 amdrocm-hpc-sdk7.14-gfx950

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx950 amdrocm-hpc-sdk7.14-gfx950

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx950 amdrocm-hpc-sdk7.14-gfx950

   .. selected:: gfx=gfx942

      2. Use the following command to install HPC-SDK for your ``gfx942`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx942 amdrocm-hpc-sdk7.14-gfx942

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx942 amdrocm-hpc-sdk7.14-gfx942

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx942 amdrocm-hpc-sdk7.14-gfx942

   .. selected:: gfx=gfx90a

      2. Use the following command to install HPC-SDK for your ``gfx90a`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx90a amdrocm-hpc-sdk7.14-gfx90a

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx90a amdrocm-hpc-sdk7.14-gfx90a

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx90a amdrocm-hpc-sdk7.14-gfx90a

   .. selected:: gfx=gfx908

      2. Use the following command to install HPC-SDK for your ``gfx908`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx908 amdrocm-hpc-sdk7.14-gfx908

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx908 amdrocm-hpc-sdk7.14-gfx908

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx908 amdrocm-hpc-sdk7.14-gfx908

   .. selected:: gfx=gfx1200

      2. Use the following command to install HPC-SDK for your ``gfx1200`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1200 amdrocm-hpc-sdk7.14-gfx1200

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1200 amdrocm-hpc-sdk7.14-gfx1200

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1200 amdrocm-hpc-sdk7.14-gfx1200

   .. selected:: gfx=gfx1201

      2. Use the following command to install HPC-SDK for your ``gfx1201`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1201 amdrocm-hpc-sdk7.14-gfx1201

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1201 amdrocm-hpc-sdk7.14-gfx1201

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1201 amdrocm-hpc-sdk7.14-gfx1201

   .. selected:: gfx=gfx1100

      2. Use the following command to install HPC-SDK for your ``gfx1100`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1100 amdrocm-hpc-sdk7.14-gfx1100

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1100 amdrocm-hpc-sdk7.14-gfx1100

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1100 amdrocm-hpc-sdk7.14-gfx1100

   .. selected:: gfx=gfx1101

      2. Use the following command to install HPC-SDK for your ``gfx1101`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1101 amdrocm-hpc-sdk7.14-gfx1101

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1101 amdrocm-hpc-sdk7.14-gfx1101

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1101 amdrocm-hpc-sdk7.14-gfx1101

   .. selected:: gfx=gfx1102

      2. Use the following command to install HPC-SDK for your ``gfx1102`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1102 amdrocm-hpc-sdk7.14-gfx1102

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1102 amdrocm-hpc-sdk7.14-gfx1102

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1102 amdrocm-hpc-sdk7.14-gfx1102

   .. selected:: gfx=gfx1103

      2. Use the following command to install HPC-SDK for your ``gfx1103`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1103 amdrocm-hpc-sdk7.14-gfx1103

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1103 amdrocm-hpc-sdk7.14-gfx1103

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1103 amdrocm-hpc-sdk7.14-gfx1103

   .. selected:: gfx=gfx1030

      2. Use the following command to install HPC-SDK for your ``gfx1030`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1030 amdrocm-hpc-sdk7.14-gfx1030

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1030 amdrocm-hpc-sdk7.14-gfx1030

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1030 amdrocm-hpc-sdk7.14-gfx1030

   .. selected:: gfx=gfx1151

      2. Use the following command to install HPC-SDK for your ``gfx1151`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1151 amdrocm-hpc-sdk7.14-gfx1151

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1151 amdrocm-hpc-sdk7.14-gfx1151

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1151 amdrocm-hpc-sdk7.14-gfx1151

   .. selected:: gfx=gfx1150

      2. Use the following command to install HPC-SDK for your ``gfx1150`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1150 amdrocm-hpc-sdk7.14-gfx1150

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1150 amdrocm-hpc-sdk7.14-gfx1150

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1150 amdrocm-hpc-sdk7.14-gfx1150

   .. selected:: gfx=gfx1152

      2. Use the following command to install HPC-SDK for your ``gfx1152`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1152 amdrocm-hpc-sdk7.14-gfx1152

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1152 amdrocm-hpc-sdk7.14-gfx1152

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1152 amdrocm-hpc-sdk7.14-gfx1152

   .. selected:: gfx=gfx1153

      2. Use the following command to install HPC-SDK for your ``gfx1153`` GPU:

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

                sudo apt install amdrocm-hpc7.14-gfx1153 amdrocm-hpc-sdk7.14-gfx1153

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

                sudo dnf install amdrocm-hpc7.14-gfx1153 amdrocm-hpc-sdk7.14-gfx1153

         .. selected:: os=sles

            .. code-block:: bash

                sudo zypper install amdrocm-hpc7.14-gfx1153 amdrocm-hpc-sdk7.14-gfx1153

Uninstall HPC-SDK
----------------------

.. selected:: i=tar

   The standard ROCm uninstallation process can be followed to uninstall HPC-SDK. No additional
   steps are required to remove the HPC-SDK separately. Refer to :ref:`Uninstalling <rocm-uninstall>`
   section and select Tarball from the installation environment selector.

.. selected:: i=pkgman

   .. selected:: fam=all

      Use the following command to uninstall HPC-SDK for all GPU architectures:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14 amdrocm-hpc-sdk7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14 amdrocm-hpc-sdk7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14 amdrocm-hpc-sdk7.14

   .. selected:: gfx=gfx950

      Use the following command to uninstall HPC-SDK for your ``gfx950`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx950 amdrocm-hpc-sdk7.14-gfx950

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx950 amdrocm-hpc-sdk7.14-gfx950

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx950 amdrocm-hpc-sdk7.14-gfx950

   .. selected:: gfx=gfx942

      Use the following command to uninstall HPC-SDK for your ``gfx942`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx942 amdrocm-hpc-sdk7.14-gfx942

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx942 amdrocm-hpc-sdk7.14-gfx942

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx942 amdrocm-hpc-sdk7.14-gfx942

   .. selected:: gfx=gfx90a

      Use the following command to uninstall HPC-SDK for your ``gfx90a`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx90a amdrocm-hpc-sdk7.14-gfx90a

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx90a amdrocm-hpc-sdk7.14-gfx90a

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx90a amdrocm-hpc-sdk7.14-gfx90a

   .. selected:: gfx=gfx908

      Use the following command to uninstall HPC-SDK for your ``gfx908`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx908 amdrocm-hpc-sdk7.14-gfx908

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx908 amdrocm-hpc-sdk7.14-gfx908

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx908 amdrocm-hpc-sdk7.14-gfx908

   .. selected:: gfx=gfx1200

      Use the following command to uninstall HPC-SDK for your ``gfx1200`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1200 amdrocm-hpc-sdk7.14-gfx1200

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1200 amdrocm-hpc-sdk7.14-gfx1200

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1200 amdrocm-hpc-sdk7.14-gfx1200

   .. selected:: gfx=gfx1201

      Use the following command to uninstall HPC-SDK for your ``gfx1201`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1201 amdrocm-hpc-sdk7.14-gfx1201

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1201 amdrocm-hpc-sdk7.14-gfx1201

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1201 amdrocm-hpc-sdk7.14-gfx1201

   .. selected:: gfx=gfx1100

      Use the following command to uninstall HPC-SDK for your ``gfx1100`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1100 amdrocm-hpc-sdk7.14-gfx1100

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1100 amdrocm-hpc-sdk7.14-gfx1100

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1100 amdrocm-hpc-sdk7.14-gfx1100

   .. selected:: gfx=gfx1101

      Use the following command to uninstall HPC-SDK for your ``gfx1101`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1101 amdrocm-hpc-sdk7.14-gfx1101

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1101 amdrocm-hpc-sdk7.14-gfx1101

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1101 amdrocm-hpc-sdk7.14-gfx1101

   .. selected:: gfx=gfx1102

      Use the following command to uninstall HPC-SDK for your ``gfx1102`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1102 amdrocm-hpc-sdk7.14-gfx1102

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1102 amdrocm-hpc-sdk7.14-gfx1102

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1102 amdrocm-hpc-sdk7.14-gfx1102

   .. selected:: gfx=gfx1103

      Use the following command to uninstall HPC-SDK for your ``gfx1103`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1103 amdrocm-hpc-sdk7.14-gfx1103

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1103 amdrocm-hpc-sdk7.14-gfx1103

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1103 amdrocm-hpc-sdk7.14-gfx1103

   .. selected:: gfx=gfx1030

      Use the following command to uninstall HPC-SDK for your ``gfx1030`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1030 amdrocm-hpc-sdk7.14-gfx1030

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1030 amdrocm-hpc-sdk7.14-gfx1030

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1030 amdrocm-hpc-sdk7.14-gfx1030

   .. selected:: gfx=gfx1151

      Use the following command to uninstall HPC-SDK for your ``gfx1151`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1151 amdrocm-hpc-sdk7.14-gfx1151

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1151 amdrocm-hpc-sdk7.14-gfx1151

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1151 amdrocm-hpc-sdk7.14-gfx1151

   .. selected:: gfx=gfx1150

      Use the following command to uninstall HPC-SDK for your ``gfx1150`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1150 amdrocm-hpc-sdk7.14-gfx1150

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1150 amdrocm-hpc-sdk7.14-gfx1150

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1150 amdrocm-hpc-sdk7.14-gfx1150

   .. selected:: gfx=gfx1152

      Use the following command to uninstall HPC-SDK for your ``gfx1152`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1152 amdrocm-hpc-sdk7.14-gfx1152

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1152 amdrocm-hpc-sdk7.14-gfx1152

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1152 amdrocm-hpc-sdk7.14-gfx1152

   .. selected:: gfx=gfx1153

      Use the following command to uninstall HPC-SDK for your ``gfx1153`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hpc7.14-gfx1153 amdrocm-hpc-sdk7.14-gfx1153

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hpc7.14-gfx1153 amdrocm-hpc-sdk7.14-gfx1153

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hpc7.14-gfx1153 amdrocm-hpc-sdk7.14-gfx1153
