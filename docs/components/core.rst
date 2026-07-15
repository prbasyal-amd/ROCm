.. meta::
   :description: AMD ROCm Core SDK - list of libraries and tools
   :keywords: component, tool, lib, library, dnn, algorithm, cli, end, machine, learning, optimization, optimize, primitive, api, binding, wrapper

************************
ROCm Core SDK components
************************

The ROCm Core SDK is the foundation of the ROCm software stack. It provides the
libraries, runtimes, compilers, and tools needed to develop and run GPU-accelerated
applications on AMD hardware.

.. datatemplate:yaml:: /_data/components.yaml

    {%- set components = data.rocm_core_sdk.components -%}
    {%- set version_slug = data.rocm_core_sdk.meta.rtd_version_slug -%}
    {%- set base_url = "https://rocm.docs.amd.com/projects" -%}
    {%- set group_order = [
        "Math and compute libraries",
        "Communication libraries",
        "Runtime and compilers",
        "Profiling and debugging tools",
        "Control and monitoring tools",
        "Media libraries",
        "Storage"
    ] -%}

    {# Collect groups not in the predefined order so they appear at the end. #}
    {%- set extra_groups = [] -%}
    {%- for name, comp in components.items() -%}
    {%-     if comp.group is defined and comp.group not in group_order and comp.group not in extra_groups -%}
    {%-         set _ = extra_groups.append(comp.group) -%}
    {%-     endif -%}
    {%- endfor -%}

    {%- for group in group_order + extra_groups -%}
    {%-     set group_items = [] -%}
    {%-     for name, comp in components.items() -%}
    {%-         if comp.group is defined and comp.group == group -%}
    {%-             set _ = group_items.append((name, comp)) -%}
    {%-         endif -%}
    {%-     endfor -%}
    {%-     if group_items %}

    {{ group }}
    {{ "=" * group | length }}
    {%- if group == "Math and compute libraries" %}

    A comprehensive set of GPU-accelerated math libraries covering dense and sparse
    linear algebra, FFTs, random number generation, and more.

    * Libraries prefixed with ``roc*`` are native, high-performance
      implementations written in HIP specifically for AMD GPUs.

    * Libraries prefixed with ``hip*`` are portable wrappers that implement
      NVIDIA CUDA-equivalent APIs, allowing CUDA applications to be ported to AMD GPUs
      with minimal code changes.

    Libraries include:
    {% endif %}
    {%- for name, comp in group_items | sort(attribute="0") -%}
    {%-     if comp.xref is defined and comp.xref.rtd_project is defined -%}
    {%-         set doc_ref = comp.xref.rtd_project -%}
    {%-     elif comp.xref is defined and comp.xref.docs is defined and "/" not in comp.xref.docs -%}
    {%-         set doc_ref = comp.xref.docs | lower -%}
    {%-     else -%}
    {%-         set doc_ref = None -%}
    {%-     endif -%}
    {%-     set desc = " -- " + comp.description if comp.description is defined else "" -%}
    {%-     if doc_ref %}
    * `{{ name }} <{{ base_url }}/{{ doc_ref }}/en/{{ version_slug }}>`__{{ desc }}
    {%-     else %}
    * {{ name }}{{ desc }}
    {%-     endif %}
    {%- endfor %}
    {%-     endif -%}
    {%- endfor %}
