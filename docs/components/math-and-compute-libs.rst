.. meta::
   :description: AMD ROCm math and compute libraries for GPU-accelerated linear algebra, FFTs, random number generation, and deep learning.
   :keywords: math, compute, library, hipBLAS, rocBLAS, MIOpen, rocFFT, rocSPARSE, ROCm

*******************************
ROCm math and compute libraries
*******************************

ROCm math and compute libraries provide GPU-accelerated implementations of
common numerical operations including dense and sparse linear algebra, FFTs,
random number generation, and deep learning primitives.

Libraries prefixed with ``roc*`` are native, high-performance implementations
written in HIP specifically for AMD GPUs. Libraries prefixed with ``hip*`` are
portable wrappers that implement NVIDIA CUDA-equivalent APIs, allowing CUDA applications
to be ported to AMD GPUs with minimal code changes.

.. datatemplate:yaml:: /data/components-current.yaml

    {%- set defaults = load("/data/components-default.yaml").rocm_core_sdk.components -%}
    {%- set current = data.rocm_core_sdk.components -%}
    {%- set slug = data.rocm_core_sdk.meta.rtd_version_slug -%}
    {%- set tag = data.rocm_core_sdk.meta.release_tag -%}
    {%- for name, comp in defaults.items() | sort(attribute="0") -%}
    {%-     if comp.group == "Math and compute libraries" -%}
    {%-         set cur = current.get(name, {}) -%}
    {%-         set ver_label = " " + cur.version|string if cur.version is defined else "" -%}
    {%-         set desc = " -- " + comp.description if comp.description is defined else "" -%}
    {%-         if comp.xref is defined and comp.xref.rtd_project is defined -%}
    {%-             set url = comp.xref.rtd_project | replace("${rtd_version_slug}", slug) | replace("${release_tag}", tag) %}
    * `{{ name }}{{ ver_label }} <{{ url }}>`__{{ desc }}
    {%-         elif comp.xref is defined and comp.xref.github_repo is defined -%}
    {%-             set url = comp.xref.github_repo | replace("${rtd_version_slug}", slug) | replace("${release_tag}", tag) %}
    * `{{ name }}{{ ver_label }} <{{ url }}>`__{{ desc }}
    {%-         else %}
    * {{ name }}{{ ver_label }}{{ desc }}
    {%-         endif %}
    {%-     endif -%}
    {%- endfor %}
