#!/usr/bin/env python3
"""Launch the untouched pinned official RynnValue Robometer server.

This wrapper installs or updates nothing.  It validates a separately created
Python 3.10 environment, the pinned upstream checkout, a complete immutable
Hugging Face snapshot, and the selected official inference profile.  It then
executes ``robometer/evals/baseline_eval_server.py`` and writes a machine-
readable provenance record after the official health/model-info endpoints are
ready.
"""
from __future__ import annotations

import argparse
import hashlib
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


def _default_checkout(dirname: str) -> Path:
    """``$REWARDBOX_UPSTREAM_DIR/<dirname>`` if present, else ``<repo>/reward_model_code/<dirname>``.

    Mirrors rewardbox.upstream without importing it: this launcher may run under an
    isolated interpreter that does not have rewardbox installed.
    """
    shared = os.environ.get("REWARDBOX_UPSTREAM_DIR")
    if shared and (Path(shared).expanduser() / dirname).is_dir():
        return Path(shared).expanduser() / dirname
    return Path(__file__).resolve().parents[1] / "reward_model_code" / dirname

UPSTREAM_COMMIT = "10e0d333f5f3811d0d130587e50f1faf48da49e5"
UPSTREAM_REPO = "https://github.com/alibaba-damo-academy/RynnValue"
PROVENANCE_SCHEMA = "rewardbox.rynnvalue-server-provenance.v1"
MODELS = {
    "4b": {
        "model_id": "Alibaba-DAMO-Academy/RynnValue-4B",
        "revision": "3f73b5d2b5e53b21f248c8791004dde6a8cf2b92",
        # This snapshot predates persistence of the runtime-only Transformers
        # field.  Its shipped RynnValueLangConfig defaults a missing value to
        # pred_slot_isolated_eager.
        "exported_attention": None,
        "effective_attention": "pred_slot_isolated_eager",
    },
    "8b": {
        "model_id": "Alibaba-DAMO-Academy/RynnValue-8B",
        "revision": "8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e",
        "exported_attention": "pred_slot_isolated_eager",
        "effective_attention": "pred_slot_isolated_eager",
    },
}
PROFILES = {
    "official-isolated": {
        "attn_override": None,
        "effective_attention": "pred_slot_isolated_eager",
    },
    "paper-eager": {
        "attn_override": "eager",
        "effective_attention": "eager",
    },
}
RUNTIME_REQUIREMENTS = {
    "torch": ">=2.8,<2.9",
    "transformers": ">=4.57,<5",
    "accelerate": ">=1",
    "fastapi": ">=0.116",
    "uvicorn": ">=0.35",
    "hydra-core": ">=1.3",
    "omegaconf": ">=2.3",
    "pydantic": ">=2",
    "python-multipart": ">=0.0.20",
    "numpy": ">=1.21",
    "Pillow": ">=9",
    "requests": ">=2.32",
    "loguru": ">=0.7",
}


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _repo_root(explicit: Path | None = None) -> Path:
    root = (
        explicit.expanduser().resolve()
        if explicit is not None
        else _default_checkout("RynnValue")
    )
    required = (
        root / "robometer" / "robometer" / "evals" / "baseline_eval_server.py",
        root / "robometer" / "robometer" / "evals" / "baselines" / "rynnvalue.py",
        root / "robometer" / "uv.lock",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(
            f"incomplete RynnValue checkout: {root}; needs the official Robometer server, adapter and uv.lock"
        )
    return root


def _require_pinned_checkout(root: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    actual = revision.stdout.strip()
    if revision.returncode != 0 or actual != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"RynnValue checkout is not at the pinned commit {UPSTREAM_COMMIT}: "
            f"{actual or revision.stderr.strip() or '<unavailable>'}"
        )
    # core.fileMode=false: a permission-only change is not a source modification.
    git = ["git", "-c", "core.fileMode=false", "-C", str(root)]
    for command, label in (
        ([*git, "diff", "--quiet", "HEAD", "--"], "tracked"),
        ([*git, "diff", "--cached", "--quiet", "HEAD", "--"], "staged"),
    ):
        result = subprocess.run(command)
        if result.returncode == 1:
            raise RuntimeError(f"the official RynnValue checkout has {label} modifications; refusing to start")
        if result.returncode != 0:
            raise RuntimeError(f"cannot verify the {label} state of the RynnValue checkout")
    return actual


def _server_python_path(path: Path) -> Path:
    """Make a venv interpreter absolute without dereferencing its symlink."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _environment_identity(python: Path) -> dict:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"server Python does not exist or is not executable: {python}")
    packages = list(RUNTIME_REQUIREMENTS)
    code = f"""
import importlib.metadata as m, json, platform, sys
names = {packages!r}
out = {{'python': platform.python_version(), 'executable': sys.executable, 'packages': {{}}}}
for name in names:
    try: out['packages'][name] = m.version(name)
    except m.PackageNotFoundError: out['packages'][name] = None
print(json.dumps(out, sort_keys=True))
"""
    process = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True
    )
    if process.returncode != 0:
        raise RuntimeError(
            "cannot inspect the RynnValue server Python: "
            + (process.stderr.strip() or process.stdout.strip())
        )
    try:
        identity = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"server Python returned invalid environment JSON: {process.stdout!r}") from exc
    version = Version(identity["python"])
    if (version.major, version.minor) != (3, 10):
        raise RuntimeError(f"the official RynnValue server requires Python 3.10, actual {version}")
    failures = []
    for package, constraint in RUNTIME_REQUIREMENTS.items():
        installed = identity["packages"].get(package)
        if installed is None:
            failures.append(f"{package} is not installed")
            continue
        if not SpecifierSet(constraint).contains(Version(installed), prereleases=True):
            failures.append(f"{package}=={installed} does not satisfy {constraint}")
    if failures:
        raise RuntimeError(
            "the separate RynnValue environment does not satisfy the pinned Robometer runtime: " + "; ".join(failures)
        )
    return identity


def _hf_snapshot_revision(path: Path) -> str | None:
    parts = path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _checkpoint_identity(
    path: Path,
    expected: dict,
    *,
    allow_unverified: bool,
) -> dict:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"RynnValue model snapshot directory does not exist: {path}")
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    single_path = path / "model.safetensors"
    if not config_path.is_file() or not (index_path.is_file() or single_path.is_file()):
        raise RuntimeError(
            "the RynnValue snapshot must contain config.json and complete safetensors weights"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid RynnValue config.json: {exc}") from exc
    expected_config = {"model_type": "rynn_value_lang", "use_meta": True}
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    exported_attention = config.get("attn_implementation")
    if exported_attention != expected["exported_attention"]:
        mismatches["attn_implementation"] = {
            "expected_exported": expected["exported_attention"],
            "actual": exported_attention,
        }
    configuration_path = path / "configuration_rynn_value_lang.py"
    if not configuration_path.is_file():
        mismatches["configuration_rynn_value_lang.py"] = {
            "expected": "official remote config source",
            "actual": "missing",
        }
    elif expected["exported_attention"] is None:
        try:
            configuration_source = configuration_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                f"cannot read the RynnValue remote config source: {configuration_path}: {exc}"
            ) from exc
        required_default = (
            'if attn_implementation is None:',
            'attn_implementation = "pred_slot_isolated_eager"',
            'self._attn_implementation = attn_implementation',
        )
        missing_default = [
            line for line in required_default if line not in configuration_source
        ]
        if missing_default:
            mismatches["effective_attn_implementation"] = {
                "expected": expected["effective_attention"],
                "missing_source_contract": missing_default,
            }
    tokenizer = config.get("value_tokenizer_config")
    if not isinstance(tokenizer, dict) or tokenizer.get("min_value") != 0.0 or tokenizer.get(
        "max_value"
    ) != 512.0:
        mismatches["value_tokenizer_config"] = {
            "expected": {"min_value": 0.0, "max_value": 512.0},
            "actual": tokenizer,
        }
    if mismatches:
        raise RuntimeError(
            "the RynnValue snapshot config does not match the official value model: "
            + json.dumps(mismatches, sort_keys=True)
        )

    weight_files: list[Path]
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(index["weight_map"].values()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"invalid RynnValue safetensors index: {exc}") from exc
        if not shard_names or not all(isinstance(name, str) and name for name in shard_names):
            raise RuntimeError("RynnValue safetensors index.weight_map is empty or invalid")
        weight_files = [path / name for name in shard_names]
    else:
        weight_files = [single_path]
    missing = [artifact.name for artifact in weight_files if not artifact.is_file()]
    if missing:
        raise RuntimeError(f"the RynnValue snapshot lacks weight shards: {missing}")
    broken_links = sorted(
        artifact.name
        for artifact in path.iterdir()
        if artifact.is_symlink() and not artifact.exists()
    )
    if broken_links:
        raise RuntimeError(f"the RynnValue snapshot contains broken symbolic links: {broken_links}")

    revision = _hf_snapshot_revision(path)
    expected_cache_name = "models--" + expected["model_id"].replace("/", "--")
    pinned = (
        revision == expected["revision"]
        and path.parent.name == "snapshots"
        and path.parent.parent.name == expected_cache_name
    )
    if not pinned and not allow_unverified:
        raise RuntimeError(
            f"model path is not a Hub snapshot of {expected['model_id']}@{expected['revision']}: "
            f"{path}. Copied/private weights require passing --allow-unverified-weights explicitly."
        )

    artifacts = [
        artifact
        for artifact in path.iterdir()
        if artifact.is_file()
        and (
            artifact.suffix in {".json", ".safetensors", ".py", ".jinja"}
            or artifact.name.startswith(("tokenizer", "vocab", "merges"))
        )
    ]
    files = {}
    for artifact in sorted(artifacts, key=lambda item: item.name):
        resolved = artifact.resolve()
        item = {
            "size": resolved.stat().st_size,
            "storage": "symlink" if artifact.is_symlink() else "regular_file",
        }
        if artifact.is_symlink():
            item["resolved_blob"] = resolved.name
        else:
            item["sha256"] = _sha256_file(artifact)
        files[artifact.name] = item
    return {
        "path": str(path),
        "model_id": expected["model_id"],
        "expected_revision": expected["revision"],
        "snapshot_revision": revision,
        "expected_hf_cache_name": expected_cache_name,
        "pinned_public_snapshot": pinned,
        "config": {
            **expected_config,
            "exported_attn_implementation": exported_attention,
            "effective_attn_implementation": expected["effective_attention"],
        },
        "weight_shards": [artifact.name for artifact in weight_files],
        "files": files,
    }


def _request_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _stop_process(process: subprocess.Popen, signal_number: int = 15) -> int:
    if process.poll() is not None:
        return int(process.returncode or 0)
    process.send_signal(signal_number)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=30)


def _build_command(
    *,
    root: Path,
    python: Path,
    model_path: Path,
    profile: str,
    camera_lookup: Path,
    host: str,
    port: int,
) -> list[str]:
    server = root / "robometer" / "robometer" / "evals" / "baseline_eval_server.py"
    command = [
        str(python),
        str(server),
        "reward_model=rynnvalue",
        f"model_path={model_path}",
        "max_frames=8",
        # Keep the HTTP request mode and released RBM-EVAL-OOD sampler profile
        # explicit.  The custom_eval fields do not resample an already-uploaded
        # multipart trajectory, but recording them in /model_info prevents this
        # server from being confused with the upstream pad_frames=true default.
        "use_frame_steps=false",
        "custom_eval.use_frame_steps=false",
        "custom_eval.pad_frames=false",
        "batch_size=16",
        f"server_url={host}",
        f"server_port={port}",
        "model_config.checkpoint_path=null",
        "model_config.mode=absolute",
        "model_config.stride=2",
        "model_config.num_frames=8",
        "model_config.max_new_tokens=128",
        "model_config.confusion_score_mode=match_binary",
        f"model_config.camera_desc_lookup_path={camera_lookup}",
    ]
    attention = PROFILES[profile]["attn_override"]
    if attention is not None:
        command.append(f"model_config.attn_implementation={attention}")
    return command


def _validate_ready_identity(model_info: dict, profile: str) -> None:
    if model_info.get("reward_model") != "rynnvalue":
        raise RuntimeError(f"wrong /model_info reward_model identity: {model_info!r}")
    config = model_info.get("config")
    model_config = config.get("model_config") if isinstance(config, dict) else None
    if not isinstance(model_config, dict):
        raise RuntimeError("/model_info lacks config.model_config")
    expected = {
        "mode": "absolute",
        "stride": 2,
        "num_frames": 8,
        "max_new_tokens": 128,
        "use_fuse": False,
    }
    mismatches = {
        key: {"expected": value, "actual": model_config.get(key)}
        for key, value in expected.items()
        if model_config.get(key) != value
    }
    if config.get("use_frame_steps") is not False:
        mismatches["use_frame_steps"] = {
            "expected": False,
            "actual": config.get("use_frame_steps"),
        }
    if config.get("max_frames") != 8:
        mismatches["max_frames"] = {
            "expected": 8,
            "actual": config.get("max_frames"),
        }
    custom_eval = config.get("custom_eval")
    if not isinstance(custom_eval, dict):
        mismatches["custom_eval"] = {
            "expected": "object with use_frame_steps=false and pad_frames=false",
            "actual": custom_eval,
        }
    else:
        if custom_eval.get("use_frame_steps") is not False:
            mismatches["custom_eval.use_frame_steps"] = {
                "expected": False,
                "actual": custom_eval.get("use_frame_steps"),
            }
        if custom_eval.get("pad_frames") is not False:
            mismatches["custom_eval.pad_frames"] = {
                "expected": False,
                "actual": custom_eval.get("pad_frames"),
            }
    expected_override = PROFILES[profile]["attn_override"]
    actual_override = model_config.get("attn_implementation")
    if expected_override is None:
        if actual_override not in {None, "pred_slot_isolated_eager"}:
            mismatches["attn_implementation"] = {
                "expected": None,
                "actual": actual_override,
            }
    elif actual_override != expected_override:
        mismatches["attn_implementation"] = {
            "expected": expected_override,
            "actual": actual_override,
        }
    if mismatches:
        raise RuntimeError(
            "the ready server does not match the requested profile: "
            + json.dumps(mismatches, sort_keys=True)
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rynnvalue-root", type=Path)
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--model-size", choices=sorted(MODELS), default="4b")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="official-isolated"
    )
    parser.add_argument("--camera-description-lookup", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--provenance-file", type=Path)
    parser.add_argument("--allow-unverified-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if not args.host.strip():
        parser.error("--host must not be empty")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("--startup-timeout must be a finite positive number")

    try:
        root = _repo_root(args.rynnvalue_root)
        revision = _require_pinned_checkout(root)
        python = _server_python_path(args.server_python)
        environment = _environment_identity(python)
        expected_model = MODELS[args.model_size]
        weights = _checkpoint_identity(
            args.model_path,
            expected_model,
            allow_unverified=args.allow_unverified_weights,
        )
        model_path = Path(weights["path"])
        camera_lookup = (
            args.camera_description_lookup.expanduser().resolve()
            if args.camera_description_lookup is not None
            else root / "robometer" / "extracted_meta_with_descriptions.json"
        )
        if not camera_lookup.is_file():
            raise RuntimeError(f"the official camera-description lookup does not exist: {camera_lookup}")
        command = _build_command(
            root=root,
            python=python,
            model_path=model_path,
            profile=args.profile,
            camera_lookup=camera_lookup,
            host=args.host,
            port=args.port,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    provenance_path = (
        args.provenance_file.expanduser().resolve()
        if args.provenance_file is not None
        else Path(__file__).resolve().parents[1]
        / ".hf"
        / "runtime"
        / "rynnvalue"
        / f"{args.model_size}-{args.profile}-{args.port}.server-provenance.json"
    )
    source_paths = (
        root / "robometer" / "robometer" / "evals" / "baseline_eval_server.py",
        root / "robometer" / "robometer" / "evals" / "baselines" / "rynnvalue.py",
        root / "rynn_value" / "processing_rynn_value_lang.py",
        root / "rynn_value" / "conversations.py",
        root / "rynn_value" / "attention_impl.py",
        root / "robometer" / "uv.lock",
    )
    source_identity = {
        str(path.relative_to(root)): _sha256_file(path) for path in source_paths
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "status": "dry_run" if args.dry_run else "launching",
        "created_at": _utc_now(),
        "upstream": {
            "repo": UPSTREAM_REPO,
            "root": str(root),
            "commit": revision,
            "expected_commit": UPSTREAM_COMMIT,
            "source_sha256": source_identity,
        },
        "weights": weights,
        "profile": {
            "name": args.profile,
            "max_frames": 8,
            "sampling": "np.rint(linspace(0,T-1,8)); first/last included",
            "use_frame_steps": False,
            "pad_frames": False,
            "mode": "absolute",
            "stride": 2,
            "num_frames": 8,
            "attn_override": PROFILES[args.profile]["attn_override"],
            "effective_attention": PROFILES[args.profile]["effective_attention"],
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 1,
            "use_fuse": False,
            "camera_description_lookup": {
                "path": str(camera_lookup),
                "sha256": _sha256_file(camera_lookup),
            },
        },
        "runtime": {
            **environment,
            "dependency_contract": RUNTIME_REQUIREMENTS,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "host": args.host,
            "port": args.port,
            "cwd": str(root / "robometer"),
            "command": command,
        },
    }
    if args.dry_run:
        print(json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False))
        return 0

    _atomic_write_json(provenance_path, provenance)
    print(f"RynnValue provenance: {provenance_path}", flush=True)
    environment_vars = os.environ.copy()
    environment_vars.setdefault("HF_HUB_OFFLINE", "1")
    environment_vars.setdefault("TRANSFORMERS_OFFLINE", "1")
    process = subprocess.Popen(
        command,
        cwd=root / "robometer",
        env=environment_vars,
    )
    base_url = f"http://127.0.0.1:{args.port}"
    deadline = time.monotonic() + args.startup_timeout
    try:
        health = None
        model_info = None
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                provenance.update(
                    status="failed_before_ready",
                    finished_at=_utc_now(),
                    returncode=returncode,
                )
                _atomic_write_json(provenance_path, provenance)
                return returncode or 1
            health = _request_json(f"{base_url}/health")
            if health and health.get("status") == "healthy":
                model_info = _request_json(f"{base_url}/model_info")
                if model_info:
                    break
            time.sleep(2.0)
        else:
            returncode = _stop_process(process)
            provenance.update(
                status="startup_timeout",
                finished_at=_utc_now(),
                returncode=returncode,
            )
            _atomic_write_json(provenance_path, provenance)
            return 1

        try:
            if health.get("reward_model") != "rynnvalue":
                raise RuntimeError(f"wrong /health reward_model identity: {health!r}")
            _validate_ready_identity(model_info, args.profile)
        except RuntimeError as exc:
            returncode = _stop_process(process)
            provenance.update(
                status="invalid_ready_identity",
                finished_at=_utc_now(),
                returncode=returncode,
                health=health,
                model_info=model_info,
                identity_error=str(exc),
            )
            _atomic_write_json(provenance_path, provenance)
            return 1

        provenance.update(
            status="ready",
            ready_at=_utc_now(),
            health=health,
            model_info=model_info,
        )
        _atomic_write_json(provenance_path, provenance)
        print(f"RynnValue server ready: {base_url}", flush=True)
        returncode = process.wait()
        provenance.update(
            status="exited",
            finished_at=_utc_now(),
            returncode=returncode,
        )
        _atomic_write_json(provenance_path, provenance)
        return returncode
    except KeyboardInterrupt:
        returncode = _stop_process(process, signal_number=2)
        provenance.update(
            status="interrupted",
            finished_at=_utc_now(),
            returncode=returncode,
        )
        _atomic_write_json(provenance_path, provenance)
        return 130


if __name__ == "__main__":
    sys.exit(main())
