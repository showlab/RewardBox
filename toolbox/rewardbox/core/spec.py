"""Pure-data model descriptions + dependency checks.  Must build without torch."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from packaging.requirements import Requirement as _PEP508
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .enums import ImplementationKind, RewardMode, SignalType
from .types import SamplingSpec


class DependencyError(RuntimeError):
    """Dependencies not satisfied.  The message must be a fix the user can follow as is."""


@dataclass(frozen=True)
class ReqItem:
    spec: str
    dist: str
    installed: str | None
    ok: bool
    reason: str


@dataclass
class ReqReport:
    ok: bool
    items: list[ReqItem] = field(default_factory=list)
    missing_env: list[str] = field(default_factory=list)
    vram_note: str = ""
    install_hint: tuple[str, ...] = ()

    def failures(self) -> list[ReqItem]:
        return [i for i in self.items if not i.ok]

    def format(self, name: str) -> str:
        lines = [f"RewardBox: cannot create {name!r}: dependencies not satisfied", ""]
        w = max((len(i.dist) for i in self.items), default=10)
        for i in self.items:
            got = i.installed or "not installed"
            lines.append(f"  {i.dist:<{w}}  {i.spec:<32}  installed {got:<14} "
                         f"{'✓' if i.ok else '✗ ' + i.reason}")
        for k in self.missing_env:
            lines.append(f"  environment variable {k} is not set  ✗")
        bad = " ".join(repr(i.spec) for i in self.failures())
        lines += ["", "  Fix:"]
        if self.install_hint:
            lines.extend(f"    {line}" for line in self.install_hint)
        else:
            lines += [
                "    uv pip install --override locks/shared-overrides.txt "
                "-r locks/shared-py311-linux-x86_64.lock",
                f"    (or install individually: pip install {bad})" if bad else "",
            ]
        lines.append("  Diagnose: rewardbox doctor")
        return "\n".join(x for x in lines if x is not None)


@dataclass(frozen=True)
class Requirements:
    pip: tuple[str, ...] = ()
    python: str = ">=3.11"
    system: tuple[str, ...] = ()
    env: tuple[str, ...] = ()
    vram_gb: float | None = None
    note: str = ""
    install_hint: tuple[str, ...] = ()

    def check(self) -> ReqReport:
        items = [_check_one(s) for s in self.pip]
        pyver = ".".join(str(x) for x in sys.version_info[:3])
        if self.python and not SpecifierSet(self.python).contains(pyver, prereleases=True):
            items.append(ReqItem(f"python{self.python}", "python", pyver, False, "version mismatch"))
        for cmd in self.system:
            ok = shutil.which(cmd) is not None
            items.append(ReqItem(cmd, cmd, "found" if ok else None, ok,
                                 "" if ok else "system command not found"))
        missing_env = [k for k in self.env if not os.environ.get(k)]
        return ReqReport(
            ok=all(i.ok for i in items) and not missing_env,
            items=items, missing_env=missing_env,
            # VRAM does not count towards ok: a static check cannot confirm it, it is a hint
            vram_note=f"needs about {self.vram_gb} GB of GPU memory (not probed)" if self.vram_gb else "",
            install_hint=self.install_hint,
        )

    def to_dict(self) -> dict:
        return {"pip": list(self.pip), "python": self.python,
                "system": list(self.system), "env": list(self.env),
                "vram_gb": self.vram_gb, "note": self.note,
                "install_hint": list(self.install_hint)}


def _check_one(spec: str) -> ReqItem:
    try:
        req = _PEP508(spec)
    except Exception as e:
        return ReqItem(spec, spec, None, False, f"cannot parse: {e}")
    if req.marker is not None and not req.marker.evaluate():
        return ReqItem(spec, req.name, None, True, "marker does not apply, skipped")
    try:
        installed = _dist_version(req.name)
    except PackageNotFoundError:
        return ReqItem(spec, req.name, None, False, "not installed")
    if not req.specifier:
        return ReqItem(spec, req.name, installed, True, "")
    try:
        ok = req.specifier.contains(Version(installed), prereleases=True)
    except InvalidVersion:
        return ReqItem(spec, req.name, installed, True, "non-standard version string, skipped")
    return ReqItem(spec, req.name, installed, ok, "" if ok else "version mismatch")


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    cls: str
    module: str
    reward_mode: RewardMode
    signal_type: SignalType
    requires: Requirements = field(default_factory=Requirements)
    model_opts: dict = field(default_factory=dict)
    sampling: SamplingSpec = field(default_factory=SamplingSpec)
    score_range: tuple[float, float] = (0.0, 1.0)
    higher_better: bool = True
    required_views: tuple[str, ...] = ("primary",)
    optional_views: tuple[str, ...] = ()
    supports_pairwise: bool = False
    outputs_success: bool = False
    zero_shot: bool = True
    paper: str = ""
    weights: str = ""
    upstream: str = ""          # official repository + entry point, written into result provenance
    description: str = ""
    implementation_kind: ImplementationKind = ImplementationKind.OFFICIAL_ADAPTER

    def to_dict(self) -> dict:
        return {
            "name": self.name, "cls": self.cls, "module": self.module,
            "reward_mode": self.reward_mode.value, "signal_type": self.signal_type.value,
            "requires": self.requires.to_dict(), "model_opts": self.model_opts,
            "sampling": self.sampling.to_dict(),
            "score_range": list(self.score_range), "higher_better": self.higher_better,
            "required_views": list(self.required_views),
            "optional_views": list(self.optional_views),
            "supports_pairwise": self.supports_pairwise,
            "outputs_success": self.outputs_success, "zero_shot": self.zero_shot,
            "paper": self.paper, "weights": self.weights, "upstream": self.upstream,
            "description": self.description,
            "implementation_kind": self.implementation_kind.value,
        }
