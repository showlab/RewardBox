"""Model registry and factory.

Design points (compared with pyiqa's directory-scan fallback):
  - the registry comes from each adapter module's module-level SPEC/SPECS, **with no
    heavy top-level dependency**, so collecting it never imports torch
  - unknown names raise with spelling suggestions; **import failures are never
    swallowed silently**
"""
from __future__ import annotations

import difflib
import fnmatch
import importlib
import inspect
import os
import re
from pathlib import Path

from .core.spec import DependencyError, ModelSpec

_ADAPTER_MODULES = (
    "rewardbox.adapters.topreward",
    "rewardbox.adapters.gvl",
    "rewardbox.adapters.roboreward",
    "rewardbox.adapters.robodopamine",
    "rewardbox.adapters.robometer",
    "rewardbox.adapters.sole_r1",
    "rewardbox.adapters.lrm",
    "rewardbox.adapters.rynnvalue",
    "rewardbox.adapters.robovaluerl",
    "rewardbox.adapters.procvlm",
    "rewardbox.adapters.vlac",
    "rewardbox.adapters.vlac_cut",
    "rewardbox.adapters.vlac_cut_vllm",
    "rewardbox.adapters.lrm_vllm",
    "rewardbox.adapters.primo",
    "rewardbox.adapters.flora",
    "rewardbox.adapters.general_vlm",
    "rewardbox.adapters.jev",
)

_CACHE: dict[str, ModelSpec] | None = None

# A record in calibration/records/ is what marks a model as verified with real weights.
# It is a pure filesystem check without heavy imports, so list/info can always call it.
# The records ship as package data, so the verified column of `rewardbox list` stays
# correct after a non-editable install; REWARDBOX_CALIBRATION_DIR may point elsewhere.
_CALIBRATION_RECORDS = Path(
    os.environ.get("REWARDBOX_CALIBRATION_DIR")
    or Path(__file__).resolve().parent / "calibration" / "records")


def verified_status(name: str) -> dict:
    """{"score": bool, "compare": bool}: whether calibration/records/<name>.json
    (score() regression) and calibration/records/compare/<name>.json (compare()
    regression) exist.  File contents are not parsed, only presence is checked."""
    return {
        "score": (_CALIBRATION_RECORDS / f"{name}.json").exists(),
        "compare": (_CALIBRATION_RECORDS / "compare" / f"{name}.json").exists(),
    }


def _collect() -> dict[str, ModelSpec]:
    """Import every adapter module and read SPEC/SPECS.

    **Deliberately no try/except**: an adapter module that accidentally imports a heavy
    dependency at top level must blow up here instead of silently losing a model.
    """
    out: dict[str, ModelSpec] = {}
    for mod_name in _ADAPTER_MODULES:
        mod = importlib.import_module(mod_name)
        specs = getattr(mod, "SPECS", None) or (
            [mod.SPEC] if hasattr(mod, "SPEC") else [])
        if not specs:
            raise RuntimeError(f"{mod_name} defines no module-level SPEC or SPECS")
        for s in specs:
            if s.name in out:
                raise RuntimeError(f"duplicate model name: {s.name} ({mod_name})")
            out[s.name] = s
    return out


def load_manifest(force: bool = False) -> dict[str, ModelSpec]:
    global _CACHE
    if _CACHE is None or force:
        _CACHE = _collect()
    return _CACHE


def get_spec(name: str) -> ModelSpec:
    m = load_manifest()
    if name not in m:
        close = difflib.get_close_matches(name, list(m), n=5, cutoff=0.4)
        hint = (f" Did you mean: {', '.join(close)}?" if close
                else f" Available models: {', '.join(sorted(m))}")
        raise KeyError(f"unknown model {name!r}.{hint}")
    return m[name]


def _natural(s: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s.lower())]


def list_models(mode: str | None = None, signal: str | None = None,
                filter: str = "", available_only: bool = False) -> list[str]:
    """List models.  Pure table lookup, no heavy import."""
    m = load_manifest()
    names = list(m)
    if mode:
        names = [n for n in names if m[n].reward_mode.value == mode]
    if signal:
        names = [n for n in names if m[n].signal_type.value == signal]
    if filter:
        names = sorted(set(fnmatch.filter(names, filter)))
    if available_only:
        names = [n for n in names if m[n].requires.check().ok]
    return sorted(names, key=_natural)


def get_model_info(name: str) -> dict:
    s = get_spec(name)
    rep = s.requires.check()
    return {**s.to_dict(), "available": rep.ok,
            "missing": [i.spec for i in rep.failures()] + list(rep.missing_env),
            "vram_note": rep.vram_note, "verified": verified_status(name)}


def validate_model_opts(name: str, opts: dict) -> None:
    """Validate against the adapter's explicit constructor arguments; ``**opts`` is no
    licence to accept typos."""
    spec = get_spec(name)
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.cls)
    signature = inspect.signature(cls.__init__)
    allowed = {
        p.name for p in signature.parameters.values()
        if p.name not in {"self", "spec", "check_deps"}
        and p.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    unknown = set(opts) - allowed
    if unknown:
        raise TypeError(
            f"{name!r} does not support model options {sorted(unknown)}; available: {sorted(allowed)}. "
            "Refusing to silently swallow a misspelled --model-opt")


def create_reward(name: str, check_deps: bool = True, **opts):
    """Create an adapter instance.  Dependency checks run **before** any heavy import."""
    spec = get_spec(name)
    validate_model_opts(name, opts)
    if check_deps:
        rep = spec.requires.check()
        if not rep.ok:
            raise DependencyError(rep.format(name))
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.cls)
    return cls(spec, check_deps=False, **opts)
