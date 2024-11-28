.. meta::
   :description: Learn about BAR memory and how it is used to handle physical address limit in ROCm
   :keywords: BAR memory, MMIO, GPU memory, Physical Address Limit, AMD, ROCm


******************************
BAR memory overview
******************************
Base Address Registers (BAR) are a type of memory that stores memory addresses used by the devices in the system. It is used to map the memory and IO devices. 


Handling physical address limits for MMIO apertures
======================================================
You need to be aware of the physical address limits of the devices when setting up additional Memory-Mapped Input/Output (MMIO) apertures. The physical address limit is an important factor when data needs to be transferred between the local memory of different devices through peer-to-peer (P2P) direct memory access (DMA). P2P DMA works when one device can directly access the local BAR memory of another. If the BAR memory is above the physical addressing limit of the device, it will not be able to access the remote BAR. There are two ways to handle this:

* Ensure that the high MMIO aperture is within the addressing limits of the devices in the system. For example, if the devices have 44-bit physical addressing limit, the MMIO High Base address (MMIOH Base) and size (MMIO High size) options in the BIOS should be set such that the aperture ends up within the 44-bit address range.

* Enable the IOMMU. If the IOMMU is enabled in non-passthrough mode, it will create a virtual IO address space for each device on the system. It also ensures that all virtual addresses created in that space are within the physical address limits of the device. The driver reports the physical address limits to the kernel. The kernel when needed sets the IO virtual address space for the device according the physical address limits.


Configuring PCIe addressing on Xeon E5 system
=================================================
To enable above 4 GB PCIe addressing on an Intel Xeon E5 based system from the BIOS, follow these steps:  

1. Enable the Above 4G Decoding option.
2. Configure MMIO. Set the MMIOH Base to 512 GB and the size MMIO High Size to 256 GB.

Verify thr BIOS configuration
-------------------------------
In the Supermicro system BIOS, go to **Advanced** > **PCIe/PCI/PnP configuration **. You need to see the following:

* Above 4G Decoding = Enabled
* MMIOH Base = 512G
* MMIO High Size = 256G


BAR configuration for GFX9 and Vega10  
---------------------------------------
For GFX9 and Vega10 which have physical address up to 44 bit and 48 bit Virtual address.

  * BAR0-1 registers: 64bit, prefetchable, GPU memory. 8GB or 16GB depending on Vega10 SKU. Must
    be placed < 2^44 to support P2P  	access from other Vega10.
  * BAR2-3 registers: 64bit, prefetchable, Doorbell. Must be placed \< 2^44 to support P2P access from
    other Vega10.
  * BAR4 register: Optional, not a boot device.
  * BAR5 register: 32bit, non-prefetchable, MMIO. Must be placed \< 4GB.

Example of BAR usage on AMD GFX8 GPUs
----------------------------------------
A sample configuration and BAR memory placement in a GFX8 GPUs with 40-bit physical address limit can be: 

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

From the sample configuration, the BAR memories and their placement details are:

1. GPU Frame Buffer BAR: ``Memory at bf40000000 (64-bit, prefetchable) [size=256M]``

  The BAR memory is set to 256M. In general, this BAR memory should be equal to the GPU memory which is generally more than 4 GB. This BAR needs to be assigned to memory addresses which are less than 40 bit (< 2^40) to allow peer-to-peer access from other GFX8 AMD GPUs. However, For GFX9 (Vega GPU) the BAR needs to be assigned to memory addresses which are less than 44 bit (< 2^44) to allow peer-to-peer access from other GFX9 AMD GPUs.

2. Doorbell BAR: ``Memory at bf50000000 (64-bit, prefetchable) [size=2M]``

  The size of the BAR is generally less than 10 MB (<10 MB). In the example, it is fixed to 2 MB for the
  generation of GPUs used. This BAR has to be assigned to memory addresses which are less than 40 bit (< 2^40) to allow peer-to-peer access from other generations of AMD GPUs.

3. IO BAR: ``I/O ports at 3000 [size=256]``

  This BAR memory is for legacy Video Graphics Array (VGA) and boot device support. Since GPUs used in this scenario, are not connected to a display (VGA devices), this is not a concern even if the SBIOS does not setup.

4. MMIO BAR: ``Memory at c7400000 (32-bit, non-prefetchable) [size=256K]``

  This BAR is required for the AMD Driver SW to access the configuration registers. Since the
  reminder of the BAR available is only 1 DWORD (32bit), this is assigned to memory addresses which are less than 4 GB (< 4GB). This is fixed at 256KB.

5 : Expansion ROM -- This is required for the AMD Driver SW to access the GPU video-bios. This is
currently fixed at 128KB.


