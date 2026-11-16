#!/usr/bin/env python3
"""Prepare a mutable SOLE-R1 runtime checkpoint and launch the pinned server.

The upstream loader writes processor files into its checkpoint directory and
uses a cwd-relative preprocessor override.  This wrapper preserves the
immutable Hugging Face snapshot, stages the intended override explicitly, and
records the effective cwd/command/runtime identity in a JSON provenance file.
The heavyweight server dependencies remain in ``reward_server/.venv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_checkout(dirname: str) -> Path:
    """``$REWARDBOX_UPSTREAM_DIR/<dirname>`` if present, else ``<repo>/reward_model_code/<dirname>``.

    Mirrors rewardbox.upstream without importing it: this launcher may run under an
    isolated interpreter that does not have rewardbox installed.
    """
    shared = os.environ.get("REWARDBOX_UPSTREAM_DIR")
    if shared and (Path(shared).expanduser() / dirname).is_dir():
        return Path(shared).expanduser() / dirname
    return Path(__file__).resolve().parents[1] / "reward_model_code" / dirname


UPSTREAM_ROOT = _default_checkout("sole-r1-model")
SERVER_ROOT = UPSTREAM_ROOT / "reward_server"
SERVER_LOCK = REPO_ROOT / "locks" / "sole-r1-server-py310-linux-x86_64.lock"
AGENTLACE_OVERLAY = REPO_ROOT / "locks" / "sole-r1-client-no-deps.txt"
CHECKPOINT_REPO = "Philip-MIT/SOLE-R1-8B"
CHECKPOINT_REVISION = "4b322fbbff3a6b9815fc13b70803cdcb8884616b"
UPSTREAM_COMMIT = "2a1286b205577a38a2ef5d9d13efa76878485f87"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _resolve_snapshot(value: Path | None) -> Path:
    if value is None:
        from huggingface_hub import snapshot_download

        value = Path(snapshot_download(
            repo_id=CHECKPOINT_REPO,
            revision=CHECKPOINT_REVISION,
            cache_dir=str(REPO_ROOT / ".hf" / "hub"),
            local_files_only=True,
        ))
    snapshot = value.expanduser().resolve()
    if not snapshot.is_dir() or snapshot.name != CHECKPOINT_REVISION:
        raise RuntimeError(
            f"the checkpoint must be a snapshot at the pinned revision {CHECKPOINT_REVISION}: {snapshot}"
        )
    return snapshot


def _prepare_runtime(snapshot: Path, runtime: Path) -> dict:
    marker_name = ".rewardbox-runtime-source.json"
    marker = runtime / marker_name
    if runtime.exists():
        if not marker.is_file():
            raise RuntimeError(f"runtime directory exists without an identity marker; refusing to reuse it: {runtime}")
        identity = json.loads(marker.read_text(encoding="utf-8"))
        if identity.get("checkpoint_revision") != CHECKPOINT_REVISION:
            raise RuntimeError(f"runtime checkpoint revision mismatch: {runtime}")
        return identity

    runtime.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=runtime.name + ".tmp-", dir=runtime.parent))
    try:
        files: dict[str, dict] = {}
        for source in sorted(snapshot.rglob("*")):
            if not source.is_file():
                continue
            rel = source.relative_to(snapshot)
            target = temporary / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            resolved = source.resolve()
            if source.name.endswith(".safetensors"):
                target.symlink_to(resolved)
                files[str(rel)] = {
                    "storage": "symlink_to_immutable_hf_blob",
                    "blob": resolved.name,
                    "size": resolved.stat().st_size,
                }
            else:
                shutil.copy2(resolved, target)
                files[str(rel)] = {
                    "storage": "mutable_copy",
                    "sha256": _sha256(target),
                    "size": target.stat().st_size,
                }

        override = (
            SERVER_ROOT / "src" / "reward_server" / "data" / "preprocessor_config.json"
        )
        if not override.is_file():
            raise FileNotFoundError(override)
        shutil.copy2(override, temporary / "preprocessor_config.json")
        identity = {
            "checkpoint_repo": CHECKPOINT_REPO,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "immutable_snapshot": str(snapshot),
            "runtime_checkpoint": str(runtime),
            "weight_storage": "absolute symlinks to immutable HF blobs",
            "mutable_files": "all non-safetensors files copied into runtime checkpoint",
            "preprocessor_override": {
                "source": str(override),
                "sha256": _sha256(override),
                "staged_before_upstream_load": True,
            },
            "files": files,
        }
        _write_json(temporary / marker_name, identity)
        temporary.rename(runtime)
        return identity
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _ensure_server_python() -> tuple[Path, str]:
    server_python = SERVER_ROOT / ".venv" / "bin" / "python"
    find_managed = [
        "uv", "python", "find", "--managed-python", "--no-project", "3.10",
    ]
    found = subprocess.run(find_managed, capture_output=True, text=True)
    if found.returncode != 0:
        subprocess.run(["uv", "python", "install", "3.10"], check=True)
        found = subprocess.run(
            find_managed, check=True, capture_output=True, text=True,
        )
    managed_python = Path(found.stdout.strip()).resolve()

    def header_for(python: Path) -> Path | None:
        if not python.is_file():
            return None
        value = subprocess.run(
            [
                str(python), "-c",
                "import pathlib, sysconfig; "
                "print(pathlib.Path(sysconfig.get_paths()['include']) / 'Python.h')",
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        path = Path(value)
        return path if path.is_file() else None

    if header_for(server_python) is None:
        subprocess.run(
            [
                "uv", "venv", "--clear", "--python", str(managed_python),
                str(SERVER_ROOT / ".venv"),
            ],
            check=True,
        )
    version = subprocess.run(
        [str(server_python), "-c", "import platform; print(platform.python_version())"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if not version.startswith("3.10."):
        raise RuntimeError(
            f"the SOLE-R1 server must use Python 3.10 as tested upstream, actual {version}: "
            f"{server_python}"
        )
    header = header_for(server_python)
    if header is None:
        raise RuntimeError(f"the SOLE-R1 server Python lacks Python.h needed by the Triton JIT: {server_python}")
    return server_python, version


def _sync_server_environment(server_python: Path, env: dict[str, str]) -> list[str]:
    if not SERVER_LOCK.is_file() or not AGENTLACE_OVERLAY.is_file():
        raise FileNotFoundError(
            f"missing SOLE-R1 server lock/overlay: {SERVER_LOCK}, {AGENTLACE_OVERLAY}"
        )
    # The released uv.lock identifies a stale local project
    # (qwen-reward-server 0.2.0 versus reward_server 1.0.0), so current uv would
    # rewrite the pinned checkout.  Use RewardBox's runtime-only Python 3.10
    # lock, add Agentlace without its broad metadata dependencies, and execute
    # the pinned official source directly through PYTHONPATH.
    subprocess.run(
        [
            "uv", "pip", "sync", "--python", str(server_python),
            str(SERVER_LOCK),
        ],
        check=True, env=env,
    )
    subprocess.run(
        [
            "uv", "pip", "install", "--python", str(server_python),
            "--no-deps", "-r", str(AGENTLACE_OVERLAY),
        ],
        check=True, env=env,
    )
    subprocess.run(
        [
            str(server_python), "-c",
            "from reward_server.main import callback; "
            "from reward_server.vlm_batch_decode import load_model; "
            "print(callback.__name__, load_model.__name__)",
        ],
        check=True, cwd=SERVER_ROOT, env=env,
    )
    return subprocess.run(
        ["uv", "pip", "freeze", "--python", str(server_python)],
        check=True, capture_output=True, text=True, env=env,
    ).stdout.splitlines()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--runtime-dir", type=Path)
    p.add_argument("--provenance-file", type=Path)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="explicit vLLM reservation fraction; upstream default is 0.9",
    )
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--sync-only", action="store_true",
        help="prepare and validate the isolated Python 3.10 server environment without launching",
    )
    args = p.parse_args()
    if args.prepare_only and args.sync_only:
        p.error("--prepare-only and --sync-only cannot be used together")
    if (
        not math.isfinite(args.gpu_memory_utilization)
        or not 0 < args.gpu_memory_utilization <= 1
    ):
        p.error("--gpu-memory-utilization must be a finite number in (0, 1]")

    snapshot = _resolve_snapshot(args.snapshot)
    runtime = (
        args.runtime_dir
        or REPO_ROOT / ".hf" / "runtime" / "sole-r1-8b" / CHECKPOINT_REVISION
    ).expanduser().resolve()
    provenance_file = (
        args.provenance_file
        or runtime.parent / f"{CHECKPOINT_REVISION}.server-provenance.json"
    ).expanduser().resolve()
    runtime_identity = _prepare_runtime(snapshot, runtime)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"SOLE-R1 checkout commit={commit}, required {UPSTREAM_COMMIT}")
    server_python = SERVER_ROOT / ".venv" / "bin" / "python"
    command = [
        str(server_python),
        "src/reward_server/main.py",
        f"checkpoint_path={runtime}", f"server_port={args.port}",
        "load_model_params.gpu_memory_utilization="
        f"{args.gpu_memory_utilization}",
    ]
    provenance = {
        "schema_version": 1,
        "service": {"name": "rewards", "port": args.port},
        "upstream": {"commit": commit, "root": str(UPSTREAM_ROOT)},
        "checkpoint": runtime_identity,
        "startup": {
            "cwd": str(SERVER_ROOT),
            "command": command,
            "wrapper": str(Path(__file__).resolve()),
            "relative_preprocessor_cp_note": (
                "upstream cwd-relative cp does not resolve from this cwd; the exact "
                "override was pre-staged in the mutable runtime checkpoint"
            ),
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_memory_utilization_default_overridden": (
                args.gpu_memory_utilization != 0.9
            ),
        },
        "environment": {
            "hf_home": str((REPO_ROOT / ".hf").resolve()),
            "server_python": str(server_python),
            "pythonpath": str(SERVER_ROOT / "src"),
            "required_python": "3.10.x (upstream tested version)",
            "isolation": "reward_server/.venv; shared RewardBox .venv is not upgraded",
            "runtime_lock": {
                "path": str(SERVER_LOCK),
                "sha256": _sha256(SERVER_LOCK),
                "scope": "imports reachable from reward_server/main.py",
            },
            "agentlace_overlay": {
                "path": str(AGENTLACE_OVERLAY),
                "sha256": _sha256(AGENTLACE_OVERLAY),
                "install_mode": "--no-deps",
            },
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "launch_state": "prepared_not_started" if args.prepare_only else "syncing",
    }
    _write_json(provenance_file, provenance)
    print(json.dumps({
        "runtime_checkpoint": str(runtime),
        "provenance_file": str(provenance_file),
        "launch_state": provenance["launch_state"],
    }, indent=2), flush=True)
    if args.prepare_only:
        return 0

    env = dict(os.environ)
    env["HF_HOME"] = str((REPO_ROOT / ".hf").resolve())
    source_path = str(SERVER_ROOT / "src")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    server_python, python_version = _ensure_server_python()
    freeze = _sync_server_environment(server_python, env)
    provenance["environment"]["python"] = python_version
    provenance["environment"]["packages"] = freeze
    if args.sync_only:
        provenance["launch_state"] = "environment_ready_not_started"
        _write_json(provenance_file, provenance)
        print(json.dumps({
            "server_python": str(server_python),
            "package_count": len(freeze),
            "provenance_file": str(provenance_file),
            "launch_state": provenance["launch_state"],
        }, indent=2), flush=True)
        return 0
    provenance["launch_state"] = "starting"
    provenance["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(provenance_file, provenance)
    os.chdir(SERVER_ROOT)
    os.execvpe(command[0], command, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
