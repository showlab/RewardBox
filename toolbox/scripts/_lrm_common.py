"""Shared provenance helpers for pinned Large-Reward-Models wrappers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _default_checkout(dirname: str) -> Path:
    """``$REWARDBOX_UPSTREAM_DIR/<dirname>`` if present, else ``<repo>/reward_model_code/<dirname>``.

    Mirrors rewardbox.upstream without importing it: this launcher may run under an
    isolated interpreter that does not have rewardbox installed.
    """
    shared = os.environ.get("REWARDBOX_UPSTREAM_DIR")
    if shared and (Path(shared).expanduser() / dirname).is_dir():
        return Path(shared).expanduser() / dirname
    return Path(__file__).resolve().parents[1] / "reward_model_code" / dirname

UPSTREAM_COMMIT = "5845ab0f7c3c6a0bef8bd37bf7f07e9533f7ed2f"
WEIGHTS_REPO = "USC-PSI-Lab/LRM-models"
WEIGHTS_REVISION = "5f8da7f3934f5d658256d77552927a9f76ad1c0e"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
    elif os.environ.get("REWARDBOX_LRM_ROOT"):
        root = Path(os.environ["REWARDBOX_LRM_ROOT"]).expanduser().resolve()
    else:
        root = _default_checkout("Large-Reward-Models")
    if not (root / "vlm_reward" / "vlm_reward_server.py").is_file():
        raise RuntimeError(
            f"{root} is not a Large-Reward-Models checkout: missing "
            "vlm_reward/vlm_reward_server.py"
        )
    return root


def git_revision(root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cannot read the Large-Reward-Models git revision: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def require_pinned_checkout(root: Path) -> str:
    revision = git_revision(root)
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(
            "Large-Reward-Models checkout revision drifted: "
            f"expected {UPSTREAM_COMMIT}, actual {revision}"
        )
    checks = (
        (["diff", "--quiet", "--"], "working tree"),
        (["diff", "--cached", "--quiet", "--"], "index"),
    )
    for args, label in checks:
        # core.fileMode=false: a permission-only change is not a source modification.
        proc = subprocess.run(["git", "-c", "core.fileMode=false", "-C", str(root), *args])
        if proc.returncode == 1:
            raise RuntimeError(
                f"Large-Reward-Models {label} has tracked modifications; "
                "the formal wrapper refuses to label dirty source as the pinned commit"
            )
        if proc.returncode != 0:
            raise RuntimeError(f"cannot check the Large-Reward-Models {label} state")
    return revision


def hf_snapshot_revision(path: str | Path) -> str | None:
    """Return ``snapshots/<revision>`` identity without dereferencing files."""
    current = Path(path).expanduser().absolute()
    for candidate in (current, *current.parents):
        if candidate.parent.name == "snapshots":
            return candidate.name
    return None


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
