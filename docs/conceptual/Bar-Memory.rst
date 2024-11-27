.. meta::
   :description: Learn about BAR memory and how it is used to handle physical address limit in ROCm
   :keywords: BAR memory, MMIO, GPU memory, Physical Address Limit, AMD, ROCm

BAR memory overview
====================================
On a Xeon E5 based system in the BIOS we can turn on above 4GB PCIe addressing, if so he need to set
memory-mapped input/output (MMIO) base address (MMIOH base) and range (MMIO high size) in the BIOS.

In the Supermicro system in the system bios you need to see the following

  * Advanced->PCIe/PCI/PnP configuration-\> Above 4G Decoding = Enabled
  * Advanced->PCIe/PCI/PnP Configuration-\>MMIOH Base = 512G
  * Advanced->PCIe/PCI/PnP Configuration-\>MMIO High Size = 256G

When we support Large Bar Capability there is a Large Bar VBIOS which also disable the IO bar.

For GFX9 and Vega10 which have Physical Address up 44 bit and 48 bit Virtual address.

  * BAR0-1 registers: 64bit, prefetchable, GPU memory. 8GB or 16GB depending on Vega10 SKU. Must
    be placed < 2^44 to support P2P  	access from other Vega10.
  * BAR2-3 registers: 64bit, prefetchable, Doorbell. Must be placed \< 2^44 to support P2P access from
    other Vega10.
  * BAR4 register: Optional, not a boot device.
  * BAR5 register: 32bit, non-prefetchable, MMIO. Must be placed \< 4GB.

Here is how our base address register (BAR) works on GFX 8 GPUs with 40 bit Physical Address Limit ::

  11:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Fiji [Radeon R9 FURY / NANO
  Series] (rev c1)

  Subsystem: Advanced Micro Devices, Inc. [AMD/ATI] Device 0b35

  Flags: bus master, fast devsel, latency 0, IRQ 119

  Memory at bf40000000 (64-bit, prefetchable) [size=256M]

  Memory at bf50000000 (64-bit, prefetchable) [size=2M]

  I/O ports at 3000 [size=256]

  Memory at c7400000 (32-bit, non-prefetchable) [size=256K]

  Expansion ROM at c7440000 [disabled] [size=128K]

Legend:

1 : GPU Frame Buffer BAR -- In this example it happens to be 256M, but typically this will be size of the
GPU memory (typically 4GB+). This BAR has to be placed \< 2^40 to allow peer-to-peer access from
other GFX8 AMD GPUs. For GFX9 (Vega GPU) the BAR has to be placed \< 2^44 to allow peer-to-peer
access from other GFX9 AMD GPUs.

2 : Doorbell BAR -- The size of the BAR is typically will be \< 10MB (currently fixed at 2MB) for this
generation GPUs. This BAR has to be placed \< 2^40 to allow peer-to-peer access from other current
generation AMD GPUs.

3 : IO BAR -- This is for legacy VGA and boot device support, but since this the GPUs in this project are
not VGA devices (headless), this is not a concern even if the SBIOS does not setup.

4 : MMIO BAR -- This is required for the AMD Driver SW to access the configuration registers. Since the
reminder of the BAR available is only 1 DWORD (32bit), this is placed \< 4GB. This is fixed at 256KB.

5 : Expansion ROM -- This is required for the AMD Driver SW to access the GPU video-bios. This is
currently fixed at 128KB.

For more information, you can review
`Overview of Changes to PCI Express 3.0 <https://www.mindshare.com/files/resources/PCIe%203-0.pdf>`_.


Setting up additional MMIO apertures
---------------------------------------

It is worth noting the physical address limits of the devices when setting up additional MMIO apertures.  This will come into play with peer to peer DMA (i.e., transfers between local memory on the devices).  For peer to peer DMA to work, one device will DMA directly to the other device’s local memory BAR.  If that BAR is above the device’s physical addressing limit, it will not be able to access the remote BAR.  There are two ways to handle this:

Make sure the high MMIO aperture is located within the addressing limits of the devices on the system.  For example, if the devices have a 44 bit physical addressing limit, the MMIO High Base and Size options in the BIOS should be set such that the aperture ends up within a 44 bit address range.
Enable the IOMMU.  If the IOMMU is enabled in non-passthrough mode, it will create a virtual IO address space for each device on the system and guarantee that all virtual addresses created in that space are within the device’s physical address limits.  The driver reports the device’s physical address limits to the kernel and the kernel takes this into account when it sets up the IO virtual address space for the device.