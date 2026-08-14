.. meta::
   :description: AMD ROCm runtimes and compilers for GPU application development, including HIP, HIPIFY, and LLVM.
   :keywords: runtime, compiler, HIP, HIPIFY, LLVM, amdclang, ROCm

***************************
ROCm runtimes and compilers
***************************

ROCm runtimes and compilers provide the core execution environment and
programming tools for GPU application development on AMD hardware.

.. datatemplate:yaml:: /data/components-current.yaml

    {%- set defaults = load("/data/components-default.yaml").rocm_core_sdk.components -%}
    {%- set current = data.rocm_core_sdk.components -%}
    {%- set slug = data.rocm_core_sdk.meta.rtd_version_slug -%}
    {%- set tag = data.rocm_core_sdk.meta.release_tag -%}
    {%- for name, comp in defaults.items() | sort(attribute="0") -%}
    {%-     if comp.group == "Runtime and compilers" -%}
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
