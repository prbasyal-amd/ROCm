"""
Sphinx extension that merges component YAML files at build time and writes
the result to docs/_data/components.yaml for the datatemplates plugin.

Configuration values (all optional, defaults shown):
  rocdoc_components_data_default   = "data/components-default.yaml"   # rel. to repo root
  rocdoc_components_data_current   = "data/components.yaml"           # rel. to repo root
  rocdoc_components_data_previous  = "data/components-previous.yaml"  # rel. to repo root (optional)
  rocdoc_components_data_output    = "_data/components.yaml"          # rel. to srcdir

Input files:
  rocdoc_components_data_default   — component metadata (group, description, xref),
                                keyed by name under rocm_core_sdk.components.
  rocdoc_components_data_current   — current-release version numbers and
                                rocm_core_sdk.meta.
  rocdoc_components_data_previous  — previous-release version numbers and
                                rocm_core_sdk.meta (skipped if file absent).

Output structure written to rocdoc_components_data_output:

  rocm_core_sdk:
    meta: { ...current release... }
    previous_meta: { ...previous release... }   # if previous file present
    components:
      ComponentName:
        group: ...
        description: ...
        xref: ...
        version: "x.y.z"            # current release
        previous_version: "x.y.z"   # previous release, if present and different
  rocm_extras: [...]
"""

from pathlib import Path

import yaml
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Default paths — all relative to the repo root except the output, which is
# relative to the Sphinx source directory (srcdir / docs/).
DEFAULT_FILE = "data/components-default.yaml"
VERSION_FILE = "data/components.yaml"
PREVIOUS_FILE = "data/components-previous.yaml"
OUTPUT_FILE = "_data/components.yaml"


def _merge(default: dict, version: dict, previous: dict | None) -> dict:
    default_sdk: dict = default.get("rocm_core_sdk", {})
    version_sdk: dict = version.get("rocm_core_sdk", {})
    previous_sdk: dict = (previous or {}).get("rocm_core_sdk", {})

    default_components: dict = default_sdk.get("components", {})
    version_components: dict = version_sdk.get("components", {})
    previous_components: dict = previous_sdk.get("components", {})

    merged: dict = {}
    for name, base in default_components.items():
        component = dict(base)

        ver_entry = version_components.get(name)
        if isinstance(ver_entry, dict) and "version" in ver_entry:
            component["version"] = ver_entry["version"]

        prev_entry = previous_components.get(name)
        if isinstance(prev_entry, dict) and "version" in prev_entry:
            prev_ver = prev_entry["version"]
            if prev_ver != component.get("version"):
                component["previous_version"] = prev_ver

        merged[name] = component

    # Components in version file but not in defaults
    for name, vc in version_components.items():
        if name not in merged:
            merged[name] = dict(vc) if isinstance(vc, dict) else {"version": vc}

    sdk_out: dict = {
        "meta": version_sdk.get("meta", {}),
        "components": merged,
    }
    if previous_sdk.get("meta"):
        sdk_out["previous_meta"] = previous_sdk["meta"]

    result: dict = {"rocm_core_sdk": sdk_out}

    if "rocm_extras" in default:
        result["rocm_extras"] = default["rocm_extras"]

    return result


def _resolve_path(value, base: Path) -> Path:
    """Return an absolute Path for *value*, joining with *base* if relative."""
    p = Path(value)
    return p if p.is_absolute() else base / p


def merge_components(app) -> None:
    root_dir = Path(app.srcdir).parent
    default_path = _resolve_path(app.config.rocdoc_components_data_default, root_dir)
    version_path = _resolve_path(app.config.rocdoc_components_data_current, root_dir)
    previous_path = _resolve_path(app.config.rocdoc_components_data_previous, root_dir)
    output_path = _resolve_path(app.config.rocdoc_components_data_output, Path(app.srcdir))

    if not default_path.exists():
        raise FileNotFoundError(f"merge_components: not found: {default_path}")
    if not version_path.exists():
        raise FileNotFoundError(f"merge_components: not found: {version_path}")

    with open(default_path, encoding="utf-8") as f:
        default = yaml.safe_load(f)
    with open(version_path, encoding="utf-8") as f:
        version = yaml.safe_load(f)

    previous = None
    if previous_path.exists():
        with open(previous_path, encoding="utf-8") as f:
            previous = yaml.safe_load(f)
    else:
        logger.info(f"merge_components: {PREVIOUS_FILE} not found, skipping previous versions")

    merged = _merge(default, version, previous)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info(f"merge_components: written to {output_path}")


def setup(app):
    # Paths for input files are relative to the repo root (srcdir parent).
    # The output path is relative to srcdir (the docs/ directory).
    app.add_config_value("rocdoc_components_data_default", DEFAULT_FILE, "env")
    app.add_config_value("rocdoc_components_data_current", VERSION_FILE, "env")
    app.add_config_value("rocdoc_components_data_previous", PREVIOUS_FILE, "env")
    app.add_config_value("rocdoc_components_data_output", OUTPUT_FILE, "env")

    app.connect("builder-inited", merge_components)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
