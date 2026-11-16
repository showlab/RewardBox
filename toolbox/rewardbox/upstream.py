"""Pinned official checkouts that adapters call into, and where to find them.

RewardBox does not vendor upstream algorithms; several adapters load the
official source from a git checkout at a pinned commit.  This module is the
single table of those checkouts (URL, commit, directory name, override
variable) plus the search order every adapter uses:

1. the adapter's own override, e.g. ``REWARDBOX_VLAC_ROOT`` (checked by the adapter);
2. ``$REWARDBOX_UPSTREAM_DIR/<dirname>``;
3. ``<parent>/reward_model_code/<dirname>`` for each parent of this package
   (a source clone with ``rewardbox fetch-upstream`` run in it);
4. ``${XDG_CACHE_HOME:-~/.cache}/rewardbox/upstream/<dirname>``.

``rewardbox fetch-upstream MODEL`` clones the pinned commit into the default
location.  Module top level imports nothing heavy (the registry imports
adapters, and adapters import this module).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_DIR_ENV = "REWARDBOX_UPSTREAM_DIR"
# A chmod of a checkout (shared machines, filesystems without mode bits) is not a source
# modification; "is the pinned source untouched" checks compare content only.
_IGNORE_MODE = ("-c", "core.fileMode=false")


class UpstreamNotFoundError(RuntimeError):
    """The pinned official checkout an adapter needs is not on disk.

    Fix with ``rewardbox fetch-upstream <model>`` or the adapter's ``REWARDBOX_*_ROOT``.
    """


@dataclass(frozen=True)
class Upstream:
    key: str                    # id accepted by ``rewardbox fetch-upstream``
    dirname: str                # directory adapters search for
    url: str
    commit: str
    models: tuple[str, ...]     # registry names (or name prefixes ending in "-") served
    marker: str                 # file that proves the checkout is the right repository
    env: str | None = None      # adapter-specific root override
    import_name: str | None = None  # adapter imports this top-level package from the checkout
    note: str = ""

    def serves(self, model: str) -> bool:
        return any(model == m or (m.endswith("-") and model.startswith(m)) for m in self.models)


UPSTREAMS: tuple[Upstream, ...] = (
    Upstream("topreward", "TOPReward", "https://github.com/TOPReward/TOPReward.git",
             "4877a0ee5098cbec18485125466631b1dcc4a573",
             # Only the native-client variants; model ``topreward`` (Robometer Table-I
             # baseline) imports nothing from this repository.
             ("topreward-",), "topreward/clients/qwen.py", import_name="topreward"),
    Upstream("robometer", "robometer", "https://github.com/robometer/robometer.git",
             "5b815254bf31ee1bea3753c3a2da9f9033736d9a",
             ("robometer-", "gvl", "gvl-", "topreward"), "robometer/evals/baselines/gvl.py",
             env="REWARDBOX_ROBOMETER_ROOT", import_name="robometer",
             note="Robometer heads; also hosts the GVL and TOPReward baselines"),
    Upstream("robodopamine", "Robo-Dopamine", "https://github.com/FlagOpen/Robo-Dopamine.git",
             "2c714abca66329f8e2113587226bc9556726f6c2",
             ("robodopamine-",), "examples/inference.py", env="REWARDBOX_ROBODOPAMINE_ROOT"),
    Upstream("sole-r1", "sole-r1-model", "https://github.com/Philip-MIT/sole-r1-model.git",
             "2a1286b205577a38a2ef5d9d13efa76878485f87",
             ("sole-r1-",), "reward_client/utils.py", env="REWARDBOX_SOLE_R1_ROOT",
             note="client + reward_server; the server runs in its own Python 3.10 env"),
    Upstream("lrm", "Large-Reward-Models",
             "https://github.com/physical-superintelligence-lab/Large-Reward-Models.git",
             "5845ab0f7c3c6a0bef8bd37bf7f07e9533f7ed2f",
             ("lrm-",), "vlm_reward/vlm_reward_client.py", env="REWARDBOX_LRM_ROOT"),
    Upstream("rynnvalue", "RynnValue", "https://github.com/alibaba-damo-academy/RynnValue.git",
             "10e0d333f5f3811d0d130587e50f1faf48da49e5",
             ("rynnvalue-",), "robometer/robometer/evals/baseline_eval_server.py",
             env="REWARDBOX_RYNNVALUE_ROOT"),
    Upstream("procvlm", "ProcVLM", "https://github.com/RUCKBReasoning/ProcVLM.git",
             "377523a31f05bab9c0db5ac8b9edfa7b7f03968a",
             ("procvlm-",), "pyproject.toml", env="REWARDBOX_PROCVLM_ROOT",
             note="server-side only; scripts/run_procvlm_server.py launches it"),
    Upstream("robovaluerl", "Robo-ValueRL", "https://github.com/Open-X-Humanoid/Robo-ValueRL.git",
             "1618bcb01212eb747a5d2659228b7fb5db87a563",
             ("robo-valuerl-",), "README.md",
             note="server-side only; scripts/run_robovaluerl_server.py launches it"),
    Upstream("vlac", "VLAC", "https://github.com/InternRobotics/VLAC.git",
             "dc57e57169f6ab823c397698ce8a629951d89774",
             ("vlac-2b", "vlac-8b"), "evo_vlac/utils/model_utils.py", env="REWARDBOX_VLAC_ROOT"),
    Upstream("vlac-cut", "VLAC-cut", "https://github.com/InternRobotics/VLAC-cut.git",
             "354df56ff177018f4ed3849996b3b1fc9cadedc8",
             ("vlac-cut", "vlac-cut-"), "scripts/run_example.py", env="REWARDBOX_VLAC_CUT_ROOT"),
    Upstream("primo", "PRIMO-R1", "https://github.com/10-OASIS-01/PRIMO-R1.git",
             "f746e1f83af001337dabffafa948129353ac962e",
             ("primo-",), "src/primo_prompts.py", env="REWARDBOX_PRIMO_ROOT"),
)


def get_upstream(key: str) -> Upstream:
    for u in UPSTREAMS:
        if u.key == key or u.dirname == key:
            return u
    raise KeyError(f"unknown upstream {key!r}; known: {', '.join(u.key for u in UPSTREAMS)}")


def upstreams_for_model(model: str) -> list[Upstream]:
    """Checkouts a registered model calls into (empty for API/HF-only models)."""
    return [u for u in UPSTREAMS if u.serves(model)]


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "rewardbox" / "upstream"


def source_checkout_dir() -> Path | None:
    """``<repo>/reward_model_code`` when RewardBox runs from a source clone."""
    root = Path(__file__).resolve().parent.parent
    return root / "reward_model_code" if (root / "pyproject.toml").is_file() else None


def default_upstream_dir() -> Path:
    """Where ``rewardbox fetch-upstream`` clones to when ``--dest`` is not given."""
    shared = os.environ.get(UPSTREAM_DIR_ENV)
    if shared:
        return Path(shared).expanduser()
    return source_checkout_dir() or _cache_dir()


def shared_upstream_dirs() -> list[Path]:
    """Directories outside a source clone: ``$REWARDBOX_UPSTREAM_DIR`` then the user cache."""
    shared = os.environ.get(UPSTREAM_DIR_ENV)
    return ([Path(shared).expanduser()] if shared else []) + [_cache_dir()]


def upstream_candidates(dirname: str) -> list[Path]:
    """Search order for one checkout (see module docstring); duplicates removed."""
    out: list[Path] = []
    shared = os.environ.get(UPSTREAM_DIR_ENV)
    if shared:
        out.append(Path(shared).expanduser() / dirname)
    out.extend(parent / "reward_model_code" / dirname for parent in Path(__file__).resolve().parents)
    out.append(_cache_dir() / dirname)
    # Resolve symlinks (a shared ``reward_model_code`` link is common) so that
    # identity guards comparing loaded module paths against the root agree.
    seen, unique = set(), []
    for p in out:
        p = p.resolve()
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def find_upstream(key: str) -> Path | None:
    """Resolved checkout root, honoring the adapter's explicit override first."""
    u = get_upstream(key)
    if u.env and os.environ.get(u.env):
        root = Path(os.environ[u.env]).expanduser().resolve()
        return root if (root / u.marker).exists() else None
    for candidate in upstream_candidates(u.dirname):
        if (candidate / u.marker).exists():
            return candidate.resolve()
    return None


def missing_message(key: str) -> str:
    u = get_upstream(key)
    override = f" or set {u.env}" if u.env else ""
    return (f"official {u.dirname} checkout not found (need {u.marker}). "
            f"Run `rewardbox fetch-upstream {u.key}`{override}. "
            f"Searched: {', '.join(str(p) for p in upstream_candidates(u.dirname))}")


def ensure_importable(key: str) -> None:
    """Put a package-style checkout (robometer, TOPReward) on ``sys.path`` if needed.

    Does nothing when the package is already importable (pip install -e or
    PYTHONPATH), so an explicitly installed copy always wins.  Raises
    :class:`UpstreamNotFoundError` when it is neither importable nor on disk, so
    the caller sees the fetch command instead of a bare ``ModuleNotFoundError``.
    """
    u = get_upstream(key)
    if u.import_name is None or u.import_name in sys.modules:
        return
    if importlib.util.find_spec(u.import_name) is not None:
        return
    root = find_upstream(key)
    if root is None:
        raise UpstreamNotFoundError(missing_message(key))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def fetch(key: str, dest: Path | None = None, *, dry_run: bool = False,
          retries: int = 3, log=print) -> Path:
    """Clone one pinned checkout (blobless, LFS smudge skipped) and verify it.

    Verified means: HEAD is the pinned commit, the working tree is clean and the
    marker file exists.  An existing directory is never modified; a dirty one
    (interrupted earlier fetch or local edits) is reported with the repair command.
    """
    u = get_upstream(key)
    base = Path(dest).expanduser() if dest is not None else default_upstream_dir()
    target = base / u.dirname
    repair = f"git -C {target} checkout -f --detach {u.commit}"
    if (target / ".git").exists():
        head = _git(["rev-parse", "HEAD"], cwd=target, capture=True).strip()
        if head != u.commit:
            raise RuntimeError(
                f"{target} exists at {head[:12]}, pinned commit is {u.commit[:12]}; "
                "move it away or check out the pinned commit yourself")
        if _git([*_IGNORE_MODE, "status", "--porcelain"], cwd=target, capture=True).strip():
            raise RuntimeError(
                f"{target} is at the pinned commit but its working tree is incomplete or "
                f"modified; remove it and re-run, or restore the pinned files with: {repair}")
        log(f"{u.key}: already at {u.commit[:12]} in {target}")
        return target
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"{target} exists and is not a git checkout; refusing to overwrite")
    clone = ["clone", "--filter=blob:none", "--no-checkout", u.url, str(target)]
    checkout = ["checkout", "--detach", u.commit]
    log(("[dry-run] " if dry_run else "") + "git " + " ".join(clone))
    log(("[dry-run] " if dry_run else "") + "git " + " ".join(checkout) + f"   (in {target})")
    if dry_run:
        return target
    base.mkdir(parents=True, exist_ok=True)
    _git(clone)
    # Blobs arrive lazily during checkout; retry transient network drops.  ``-f`` is
    # safe here only because this call created the directory a moment ago.
    for attempt in range(1, retries + 1):
        try:
            _git(checkout if attempt == 1 else ["checkout", "-f", "--detach", u.commit],
                 cwd=target)
            break
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise RuntimeError(
                    f"{u.key}: checkout failed {retries}x (network?); remove {target} and "
                    f"re-run, or resume with: {repair}") from None
            log(f"{u.key}: checkout attempt {attempt} failed, retrying")
    head = _git(["rev-parse", "HEAD"], cwd=target, capture=True).strip()
    dirty = _git([*_IGNORE_MODE, "status", "--porcelain"], cwd=target, capture=True).strip()
    if head != u.commit or dirty or not (target / u.marker).exists():
        raise RuntimeError(
            f"{u.key}: verification failed in {target} (HEAD {head[:12]}, "
            f"clean={not dirty}, marker {u.marker} present={(target / u.marker).exists()})")
    log(f"{u.key}: {target} @ {u.commit[:12]}")
    return target


def _git(args: list[str], cwd: Path | None = None, capture: bool = False) -> str:
    """Run git with LFS downloads skipped (adapters need source, not LFS media).

    ``GIT_LFS_SKIP_SMUDGE`` is read by git-lfs itself.  When a user-level config
    still registers ``filter.lfs`` but git-lfs is not installed, checkout of any
    path with an LFS attribute fails (git misreports it as a remote hang-up), so
    the filter is replaced by a pass-through that leaves pointer files as-is.
    """
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    lfs_off = [] if shutil.which("git-lfs") else [
        "-c", "filter.lfs.process=", "-c", "filter.lfs.smudge=cat",
        "-c", "filter.lfs.clean=cat", "-c", "filter.lfs.required=false"]
    proc = subprocess.run(["git", *lfs_off, *args], cwd=cwd, env=env, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None)
    return proc.stdout or ""
