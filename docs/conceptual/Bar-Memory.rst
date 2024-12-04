.. meta::
   :description: Learn about BAR memory and ways to handle physical addressing limit in ROCm
   :keywords: BAR memory, MMIO, GPU memory, Physical Addressing Limit, AMD, ROCm

******************************
BAR memory overview
******************************
BAR (Base Address Register) memory in PCIe defines and maps the memory-mapped input/output (MMIO) space required by a PCIe device for its resources such as registers or device memory. CPU and other devices in the system use BAR memory to access the resources of the PCIe devices. When the system boots, the SBIOS sets up the physical address space of the hardware platform. This also includes system memory and one or more MMIO apertures. 

Handling physical address limits for BAR memory access
===============================================================
In general, there are two MMIO apertures: one set up below 4 GB physical address space for 32-bit compatibility, and one above 4 GB for devices that need more space. You can control the memory location of the high MMIO aperture with above 4 GB physical address space from the configuration options in the SBIOS. This enables you to configure the extended MMIO space to align with the physical addressing limit of the devices on the hardware platform if peer-to-peer (P2P) Direct Memory Access (DMA) is required. The physical addressing limit is important when data transfer is needed between the local memory of different devices through P2P DMA. It works only when one device can directly access the local BAR memory of another. If the BAR memory is above the physical addressing limit of the device, it will not be able to access the remote BAR. For example, if a PCIe device is limited to 44-bit of physical addressing, you should ensure that the MMIO aperture is set below 44-bit in the hardware platform address space.

You need to be aware of the physical addressing limits of the devices when setting up additional MMIO apertures. There are two ways to handle this:

* Ensure that the high MMIO aperture is within the physical addressing limits of the devices in the system. For example, if the devices have a 44-bit physical addressing limit, set the ``MMIOH Base`` and ``MMIO High size`` options in the BIOS such that the aperture is within the 44-bit address range. Also, ensure the ``Above 4G Decoding`` option is Enabled. 

* Enable the Input-Output Memory Management Unit (IOMMU). When the IOMMU is enabled in non-passthrough mode, it will create a virtual IO address space for each device on the system. It also ensures that all virtual addresses created in that space are within the physical addressing limits of the device. The driver reports the physical addressing limits to the kernel. The kernel sets the IO virtual address space for the device according to the physical addressing limits.


BAR configuration for AMD GFX9 and Vega10 GPUs
================================================

For GFX9 and Vega10 GPUs that have the 44-bit physical address and 48-bit GPU virtual address, the BARs can be configured as:

.. list-table:: 
  :widths: 25 25 50
  :header-rows: 1

  * - BAR Type
    - Value
    - Description
  * - BAR0-1 registers
    - 64-bit, Prefetchable, GPU memory
    - 8 GB or 16 GB depending on Vega10 SKU. Must be placed less than 2^44 to support P2P access from other Vega10. Prefetching enables faster read operation for high-performance computing (HPC) by fetching the contiguous data from the same data source even before requested as an anticipation of a future request.
  * - BAR2-3 registers
    - 64-bit, Prefetchable, Doorbell
    - Must be placed less than 2^44 to support P2P access from other Vega10. As a Doorbell BAR, it indicates to the GPU that a new operation is in its queue to be processed. 
  * - BAR4 register
    - Optional
    - Not a boot device
  * - BAR5 register
    - 32-bit, Non-prefetchable, MMIO
    - Must be placed less than 4 GB

Example of BAR usage on AMD GFX8 GPUs
========================================
Following is an example configuration for BARs on GFX8 GPUs with the 40-bit physical addressing limit: 

.. code:: shell 

  11:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Fiji [Radeon R9 FURY / NANO
  Series] (rev c1)

  Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Device 0b35

  Flags: bus master, fast devsel, latency 0, IRQ 119

  Memory at bf40000000 (64-bit, prefetchable) [size=256M]

  Memory at bf50000000 (64-bit, prefetchable) [size=2M]

  I/O ports at 3000 [size=256]

  Memory at c7400000 (32-bit, non-prefetchable) [size=256K]

  Expansion ROM at c7440000 [disabled] [size=128K]

Details of the BARs configured in the example are: 

1. GPU Frame Buffer BAR: ``Memory at bf40000000 (64-bit, prefetchable) [size=256M]``

The size is 256 MB, but in general, it will be the size of the
GPU memory (typically 4 GB+). This BAR has to be set below 2^40 to allow P2P access from
other AMD GFX8 GPUs. For AMD GFX9 and Vega GPUs the BAR has to be set below 2^44 to allow P2P
access from other AMD GFX9 GPUs.

2. Doorbell BAR: ``Memory at bf50000000 (64-bit, prefetchable) [size=2M]``

The size of the BAR should typically be less than 10 MB for this generation of GPUs and has been set to 2 MB in the example. This BAR has to be placed less than 2^40 to allow peer-to-peer access from other generations of AMD GPUs.

3. IO BAR: ``I/O ports at 3000 [size=256]``

This is for legacy VGA and boot device support. Since the GPUs used are not connected to a display (VGA devices), this is not a concern even if it is not set up in the SBIOS.

4. MMIO BAR: ``Memory at c7400000 (32-bit, non-prefetchable) [size=256K]``

This is required for the AMD Driver SW to access the configuration registers. Since the reminder of the BAR available is only 1 DWORD (32-bit), this should be set below 4 GB. In the example, it is fixed at 256 KB.

5. Expansion ROM: ``Expansion ROM at c7440000 [disabled] [size=128K]``

This is required for the AMD Driver SW to access the GPU video-bios. In the example, it is fixed at 128 KB.





