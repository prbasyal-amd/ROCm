.. meta::
   :description: Learn about BAR memory and ways to handle physical addressing limit in ROCm
   :keywords: BAR memory, MMIO, GPU memory, Physical Addressing Limit, AMD, ROCm

******************************
BAR memory overview
******************************
From the BIOS of an Intel Xeon E5-based system, you can enable above 4 GB PCIe addressing. You also need to set the
memory-mapped input/output (MMIO) High Base address ``MMIOH Base`` and range ``MMIO High Size`` in the BIOS.

In the Supermicro system, under the system BIOS, go to ``Advanced > PCIe/PCI/PnP configuration``. You should see the following:

* Above 4G Decoding = Enabled
* MMIOH Base = 512G
* MMIO High Size = 256G

.. Note:: 

  When we support the Large Base Address Register (BAR) Capability there is a Large BAR VBIOS that disables the IO BAR.

You need to be aware of the physical addressing limits of the devices when setting up additional Memory-Mapped Input/Output (MMIO) apertures. The physical addressing limit is important when data transfer is needed between the local memory of different devices through peer-to-peer (P2P) direct memory access (DMA). P2P DMA works only when one device can directly access the local BAR memory of another. If the BAR memory is above the physical addressing limit of the device, it will not be able to access the remote BAR. There are two ways to handle this:

* Ensure that the high MMIO aperture is within the physical addressing limits of the devices in the system. For example, if the devices have a 44-bit physical addressing limit, ``MMIOH Base`` and ``MMIO High size`` options in the BIOS should be set such that the aperture is within the 44-bit address range.

* Enable the Input-Output Memory Management Unit (IOMMU). When the IOMMU is enabled in non-passthrough mode, it will create a virtual IO address space for each device on the system. It also ensures that all virtual addresses created in that space are within the physical addressing limits of the device. The driver reports the physical addressing limits to the kernel. The kernel sets the IO virtual address space for the device according to the physical addressing limits.

For GFX9 and Vega10 GPUs that have 44-bit physical address and 48-bit virtual address, the BARs can be configured as:

.. list-table:: 
  :widths: 25 25 50
  :header-rows: 1

  * - BAR Type
    - Value
    - Description
  * - BAR0-1 registers
    - 64-bit, prefetchable, GPU memory
    - 8 GB or 16 GB depending on Vega10 SKU. Must be placed less than 2^44 to support P2P access from other Vega10. Prefetching enables faster read operation for high-performance computing (HPC) by fetching the contiguous data from the same data source even before requested as an anticipation of a future request.
  * - BAR2-3 registers
    - 64-bit, prefetchable, Doorbell
    - Must be placed less than 2^44 to support P2P access from other Vega10. Doorbell indicates the GPU that a new operation is in its queue to be processed. 
  * - BAR4 register
    - Optional
    - Not a boot device
  * - BAR5 register
    - 32-bit, non-prefetchable, MMIO
    - Must be placed less than 4 GB

Following is an example configuration for BARs on GFX8 GPUs with 40-bit physical addressing limit: 

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

The size is 256 MB, but in general this will be the size of the
GPU memory (typically 4 GB+). This BAR has to be placed less than 2^40 to allow peer-to-peer access from
other GFX8 AMD GPUs. For GFX9 (Vega GPU) the BAR has to be placed less than 2^44 to allow peer-to-peer
access from other GFX9 AMD GPUs.

2. Doorbell BAR: ``Memory at bf50000000 (64-bit, prefetchable) [size=2M]``

The size of the BAR should typically be less than 10 MB (currently set to 2 MB) for this generation of GPUs. This BAR has to be placed less than 2^40 to allow peer-to-peer access from other generations of AMD GPUs.

3. IO BAR: ``I/O ports at 3000 [size=256]``

This is for legacy VGA and boot device support. Since the GPUs used are not connected to a display (VGA devices), this is not a concern even if the SBIOS does not setup.

4. MMIO BAR: ``Memory at c7400000 (32-bit, non-prefetchable) [size=256K]``

This is required for the AMD Driver SW to access the configuration registers. Since the reminder of the BAR available is only 1 DWORD (32-bit), this is placed less than 4 GB. This is fixed at 256 KB.

5. Expansion ROM: ``Expansion ROM at c7440000 [disabled] [size=128K]``

This is required for the AMD Driver SW to access the GPU video-bios. This is currently fixed at 128 KB.





