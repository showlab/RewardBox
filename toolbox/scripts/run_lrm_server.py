#!/usr/bin/env python3
"""Launch the pinned official Large-Reward-Models Flask server.

This wrapper does not install or update anything.  It requires a separate
Python 3.10 server environment, validates the official dependency ranges plus
the effective import closure, validates the checkpoint layout, writes a
provenance JSON, then executes the untouched
``vlm_reward/vlm_reward_server.py`` entry point.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from _lrm_common import (
    BASE_MODEL,
    UPSTREAM_COMMIT,
    WEIGHTS_REPO,
    WEIGHTS_REVISION,
    atomic_write_json,
    hf_snapshot_revision,
    repo_root,
    require_pinned_checkout,
    sha256_file,
    utc_now,
)

HEADS = ("contrastive", "progress", "completion")
OFFICIAL_REQUIREMENTS = {
    "torch": ">=2.4.1,<2.6",
    "transformers": ">=4.57,<5",
    "flask": "",
    "numpy": "",
    "Pillow": "",
    "requests": "",
    "qwen-vl-utils": "",
    "peft": "",
    "safetensors": "",
}
# The pinned server passes ``device_map=device`` to Transformers.  Transformers
# rejects that code path when Accelerate is absent, but the upstream
# requirements.txt does not list it.  Keep this runtime-closure repair separate
# from the official dependency declaration so provenance does not blur them.
IMPORT_CLOSURE_REQUIREMENTS = {"accelerate": ">=0.29"}
REQUIREMENTS = {**OFFICIAL_REQUIREMENTS, **IMPORT_CLOSURE_REQUIREMENTS}


def _server_python_path(path: Path) -> Path:
    """Make an interpreter path absolute without dereferencing a venv symlink."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _environment_identity(python: Path) -> dict:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"server Python does not exist or is not executable: {python}")
    code = """
import importlib.metadata as m, json, platform, sys
names = ['torch','transformers','flask','numpy','Pillow','requests','qwen-vl-utils','peft','safetensors','accelerate']
out = {'python': platform.python_version(), 'executable': sys.executable, 'packages': {}}
for name in names:
    try: out['packages'][name] = m.version(name)
    except m.PackageNotFoundError: out['packages'][name] = None
print(json.dumps(out, sort_keys=True))
"""
    proc = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cannot inspect the server Python: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    try:
        identity = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"server Python returned invalid environment JSON: {proc.stdout!r}") from exc
    py = Version(identity["python"])
    if (py.major, py.minor) != (3, 10):
        raise RuntimeError(
            f"the official server environment requires Python 3.10, actual {identity['python']}"
        )
    problems = []
    for package, constraint in REQUIREMENTS.items():
        installed = identity["packages"].get(package)
        if installed is None:
            problems.append(f"{package} is not installed")
        elif constraint and not SpecifierSet(constraint).contains(
            Version(installed), prereleases=True
        ):
            problems.append(f"{package}=={installed} does not satisfy {constraint}")
    if problems:
        raise RuntimeError(
            "the separate LRM server environment does not satisfy the official requirements / actual import closure: "
            + "; ".join(problems)
        )
    return identity


def _checkpoint_layout(path: Path) -> str:
    adapter_config = path / "adapter_config.json"
    adapter_weights = path / "adapter_model.safetensors"
    if adapter_config.is_file() and adapter_weights.is_file():
        return "peft_adapter"
    config = path / "config.json"
    single_weights = path / "model.safetensors"
    index_path = path / "model.safetensors.index.json"
    if config.is_file() and single_weights.is_file():
        return "full_model"
    if config.is_file() and index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"invalid full-model safetensors index: {index_path}: {exc}") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"full-model safetensors index.weight_map is empty: {index_path}")
        shards = set(weight_map.values())
        if not all(isinstance(name, str) and name for name in shards):
            raise RuntimeError(f"invalid shard names in the full-model safetensors index: {index_path}")
        missing = sorted(name for name in shards if not (path / name).is_file())
        if missing:
            raise RuntimeError(f"the full-model snapshot lacks shards referenced by the index: {missing}")
        return "full_model"
    raise RuntimeError(
        f"the checkpoint is neither an official PEFT adapter nor a complete Transformers model: {path}"
    )


def _checkpoint_identities(
    paths: dict[str, Path],
    layouts: dict[str, str],
    *,
    allow_unverified: bool,
) -> dict[str, dict]:
    """Bind public checkpoints to their immutable Hub snapshot identity."""
    identities = {}
    for head, path in paths.items():
        snapshot_revision = hf_snapshot_revision(path)
        pinned_public_snapshot = (
            snapshot_revision == WEIGHTS_REVISION and path.name == head
        )
        if not pinned_public_snapshot and not allow_unverified:
            raise RuntimeError(
                f"the {head} checkpoint is not the same-name directory under the {WEIGHTS_REPO}@{WEIGHTS_REVISION} "
                f"snapshot: {path}. If this is an official PEFT export not yet published, "
                "pass --allow-unverified-weights explicitly and keep the unverified marker in the results."
            )

        files = {}
        for artifact in sorted(path.iterdir(), key=lambda item: item.name):
            if not artifact.is_file():
                continue
            if not (
                artifact.name.endswith(".json")
                or artifact.name.endswith(".safetensors")
                or artifact.name.startswith("tokenizer")
                or artifact.name.startswith("vocab")
                or artifact.name.startswith("merges")
            ):
                continue
            resolved = artifact.resolve()
            record = {
                "size": resolved.stat().st_size,
                "storage": "symlink" if artifact.is_symlink() else "regular_file",
            }
            if artifact.is_symlink():
                # Hugging Face cache symlinks resolve to content-addressed blob names.
                record["resolved_blob"] = resolved.name
            else:
                record["sha256"] = sha256_file(artifact)
            files[artifact.name] = record
        identities[head] = {
            "path": str(path),
            "layout": layouts[head],
            "snapshot_revision": snapshot_revision,
            "pinned_public_snapshot": pinned_public_snapshot,
            "files": files,
        }
    return identities


def _auxiliary_path_identity(path: Path) -> dict:
    if path.is_file():
        return {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
    files = {}
    for artifact in sorted(path.rglob("*"), key=lambda item: str(item)):
        if artifact.is_file() and not any(
            part.startswith(".") for part in artifact.relative_to(path).parts
        ):
            files[str(artifact.relative_to(path))] = {
                "sha256": sha256_file(artifact),
                "size": artifact.stat().st_size,
            }
    if not files:
        raise RuntimeError(f"processor path directory is empty: {path}")
    return {"path": str(path), "files": files}


def _health(url: str, timeout: float = 2.0) -> dict | None:
    try:
        request = Request(f"{url}/health", headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _stop_process(process: subprocess.Popen, *, signal_number: int = 15) -> int:
    """Stop only the server child and never leave a timed-out launch behind."""
    if process.poll() is not None:
        return int(process.returncode or 0)
    process.send_signal(signal_number)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=30)


def _resolve_paths(args, root: Path) -> tuple[dict[str, Path], dict[str, str]]:
    if args.mode == "tri":
        paths = {
            "contrastive": args.contrastive_adapter,
            "progress": args.progress_adapter,
            "completion": args.completion_adapter,
        }
        if args.weights_root:
            weights_root = args.weights_root.expanduser().resolve()
            paths = {
                head: value or weights_root / head for head, value in paths.items()
            }
        if any(value is None for value in paths.values()):
            raise RuntimeError(
                "tri needs --weights-root or the three --*-adapter paths"
            )
        resolved = {head: Path(value).expanduser().resolve() for head, value in paths.items()}
    else:
        model_path = args.model_path
        if model_path is None and args.weights_root is not None:
            model_path = args.weights_root / args.mode
        if model_path is None:
            raise RuntimeError("single-head mode needs --model-path or --weights-root")
        resolved = {args.mode: Path(model_path).expanduser().resolve()}

    layouts = {}
    for head, path in resolved.items():
        if not path.is_dir():
            raise RuntimeError(f"{head} checkpoint directory does not exist: {path}")
        layouts[head] = _checkpoint_layout(path)
    if args.mode == "tri" and set(layouts.values()) != {"peft_adapter"}:
        raise RuntimeError(
            "the official single-backbone tri loader only accepts three PEFT adapters; the pinned HF "
            f"revision {WEIGHTS_REVISION} publishes complete model directories. Do not pass full-model "
            "paths to the tri loader; start three single-head servers instead and let the lrm-tri adapter "
            "fuse them with fixed weights via contrastive_url/progress_url/completion_url."
        )
    return resolved, layouts


def _build_command(args, root: Path, python: Path, paths: dict[str, Path]) -> list[str]:
    command = [
        str(python),
        str(root / "vlm_reward" / "vlm_reward_server.py"),
        "--host", args.host,
        "--port", str(args.port),
        "--gpu_id", str(args.gpu_id),
    ]
    if args.mode == "tri":
        command.extend(
            [
                "--tri_contrastive_adapter", str(paths["contrastive"]),
                "--tri_progress_adapter", str(paths["progress"]),
                "--tri_completion_adapter", str(paths["completion"]),
                "--base_model_path", args.base_model_path,
            ]
        )
    else:
        command.extend(["--model_path", str(paths[args.mode])])
        if args.base_model_path:
            command.extend(["--base_model_path", args.base_model_path])
        if args.processor_path:
            command.extend(["--processor_path", str(args.processor_path.resolve())])
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=[*HEADS, "tri"], required=True)
    parser.add_argument("--lrm-root", type=Path)
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--contrastive-adapter", type=Path)
    parser.add_argument("--progress-adapter", type=Path)
    parser.add_argument("--completion-adapter", type=Path)
    parser.add_argument("--base-model-path", default=BASE_MODEL)
    parser.add_argument("--processor-path", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--provenance-file", type=Path)
    parser.add_argument(
        "--allow-unverified-weights",
        action="store_true",
        help="allow non-pinned/private checkpoint paths, recorded as unverified",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.gpu_id < 0:
        parser.error("--gpu-id must be >= 0")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("--startup-timeout must be > 0")
    if not args.host.strip():
        parser.error("--host must not be empty")
    if not isinstance(args.base_model_path, str) or not args.base_model_path.strip():
        parser.error("--base-model-path must not be empty")

    try:
        root = repo_root(args.lrm_root)
        revision = require_pinned_checkout(root)
        # ``venv/bin/python`` is normally a symlink to the base interpreter.
        # Path.resolve() would bypass pyvenv.cfg and inspect the package-empty
        # base Python instead of the requested isolated environment.
        python = _server_python_path(args.server_python)
        environment = _environment_identity(python)
        paths, layouts = _resolve_paths(args, root)
        checkpoint_identities = _checkpoint_identities(
            paths, layouts, allow_unverified=args.allow_unverified_weights
        )
        processor_identity = None
        if args.processor_path is not None:
            processor_path = args.processor_path.expanduser().resolve()
            if not (processor_path.is_file() or processor_path.is_dir()):
                raise RuntimeError(f"processor path does not exist or is not a regular file/directory: {processor_path}")
            args.processor_path = processor_path
            processor_identity = _auxiliary_path_identity(processor_path)
        command = _build_command(args, root, python, paths)
    except RuntimeError as exc:
        parser.error(str(exc))

    provenance_file = args.provenance_file
    if provenance_file is None:
        provenance_file = (
            Path(__file__).resolve().parents[1]
            / ".hf"
            / "runtime"
            / "lrm"
            / f"{args.mode}-{args.port}.server-provenance.json"
        )
    provenance_file = provenance_file.expanduser().resolve()
    source_files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in (
            root / "vlm_reward" / "vlm_reward_server.py",
            root / "vlm_reward" / "vlm_reward_client.py",
            root / "vlm_reward" / "requirements.txt",
        )
    }
    provenance = {
        "schema": "rewardbox.lrm-server-provenance.v1",
        "status": "dry_run" if args.dry_run else "launching",
        "created_at": utc_now(),
        "mode": args.mode,
        "upstream": {
            "repo": "https://github.com/physical-superintelligence-lab/Large-Reward-Models",
            "commit": revision,
            "expected_commit": UPSTREAM_COMMIT,
            "root": str(root),
            "source_sha256": source_files,
        },
        "weights": {
            "expected_repo": WEIGHTS_REPO,
            "expected_revision": WEIGHTS_REVISION,
            "all_paths_pinned_public_snapshot": all(
                item["pinned_public_snapshot"]
                for item in checkpoint_identities.values()
            ),
            "allow_unverified_weights": args.allow_unverified_weights,
            "checkpoints": checkpoint_identities,
            "base_model_path": args.base_model_path,
            "base_model_used": any(
                layout == "peft_adapter" for layout in layouts.values()
            ),
            "processor_override": processor_identity,
        },
        "runtime": {
            **environment,
            "dependency_contract": {
                "official_requirements": OFFICIAL_REQUIREMENTS,
                "rewardbox_import_closure": IMPORT_CLOSURE_REQUIREMENTS,
            },
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_id_argument": args.gpu_id,
            "host": args.host,
            "port": args.port,
            "command": command,
            "cwd": str(root / "vlm_reward"),
        },
    }
    if args.dry_run:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return 0

    atomic_write_json(provenance_file, provenance)
    print(f"LRM provenance: {provenance_file}", flush=True)
    process = subprocess.Popen(command, cwd=root / "vlm_reward")
    deadline = time.monotonic() + args.startup_timeout
    health_url = f"http://127.0.0.1:{args.port}"
    try:
        payload = None
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                provenance.update(
                    status="failed_before_ready",
                    finished_at=utc_now(),
                    returncode=returncode,
                )
                atomic_write_json(provenance_file, provenance)
                return returncode or 1
            payload = _health(health_url)
            if payload and payload.get("model_loaded", False):
                break
            time.sleep(2.0)
        else:
            returncode = _stop_process(process)
            provenance.update(
                status="startup_timeout",
                finished_at=utc_now(),
                returncode=returncode,
            )
            atomic_write_json(provenance_file, provenance)
            return 1

        if args.mode == "tri" and not set(HEADS) <= set(payload.get("adapters", [])):
            returncode = _stop_process(process)
            provenance.update(
                status="invalid_tri_health",
                finished_at=utc_now(),
                health=payload,
                returncode=returncode,
            )
            atomic_write_json(provenance_file, provenance)
            return 1
        provenance.update(status="ready", ready_at=utc_now(), health=payload)
        atomic_write_json(provenance_file, provenance)
        print(f"LRM server ready: {health_url}/health", flush=True)
        returncode = process.wait()
        provenance.update(
            status="exited", finished_at=utc_now(), returncode=returncode
        )
        atomic_write_json(provenance_file, provenance)
        return returncode
    except KeyboardInterrupt:
        returncode = _stop_process(process, signal_number=2)
        provenance.update(
            status="interrupted", finished_at=utc_now(), returncode=returncode
        )
        atomic_write_json(provenance_file, provenance)
        return 130


if __name__ == "__main__":
    sys.exit(main())
