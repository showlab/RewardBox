#!/usr/bin/env python3
"""Run pinned official ProcVLM inference behind a small NPZ/HTTP boundary.

Execute this file with ``reward_model_code/ProcVLM/.venv/bin/python``.  It does
not install or update anything.  Before loading vLLM it verifies the upstream
commit, official Python/package versions, immutable Hugging Face revision, all
runtime weight hashes, and the released procedural prompt.  The upstream
``evqa.inference`` and ``evqa.model`` functions remain the source of inference,
windowing, prompting, parsing, and carry-forward behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _default_checkout(dirname: str) -> Path:
    """``$REWARDBOX_UPSTREAM_DIR/<dirname>`` if present, else ``<repo>/reward_model_code/<dirname>``.

    Mirrors rewardbox.upstream without importing it: this launcher may run under an
    isolated interpreter that does not have rewardbox installed.
    """
    shared = os.environ.get("REWARDBOX_UPSTREAM_DIR")
    if shared and (Path(shared).expanduser() / dirname).is_dir():
        return Path(shared).expanduser() / dirname
    return Path(__file__).resolve().parents[1] / "reward_model_code" / dirname

UPSTREAM_REPO = "https://github.com/RUCKBReasoning/ProcVLM"
UPSTREAM_COMMIT = "377523a31f05bab9c0db5ac8b9edfa7b7f03968a"
WEIGHTS_REPO = "ce-amtic/ProcVLM-2B"
WEIGHTS_REVISION = "a7f5c18c6128a9c2a478a62ecc356f19f0e0bcf5"
WEIGHT_FILES = {
    "model-00001-of-00002.safetensors": {
        "sha256": "b56d35efd5c2d5ec38f6cbb2bd83c337a1425acf1ec5e45ffc52f7ee260715a5",
        "size_bytes": 3_106_948_088,
    },
    "model-00002-of-00002.safetensors": {
        "sha256": "00addfac3617d77d3db4f0957201824700ebb3ed32ef7a684d894af5e38e6055",
        "size_bytes": 1_770_807_984,
    },
    "procvlm_extra/procvlm_head.pt": {
        "sha256": "a1ae24192f79e63e42846b23aed82820f73e473fc9c799d6e73a0b39e1afb875",
        "size_bytes": 69_278_406,
    },
}
PROMPT_SHA256 = "0dd106e10c778ab3b963d23af18ccde5a16a4023695ffa0f8a3f2e079db52c69"
PROFILE = "official-vllm-text-progress-v1"
PROVENANCE_SCHEMA = "rewardbox.procvlm-server-provenance.v1"
SCORE_SCHEMA = "rewardbox.procvlm-score.v1"
MAX_REQUEST_BYTES = 2 * 1024**3

PROFILE_FIELDS = {
    "name": PROFILE,
    "backend": "vllm",
    "model_class": "Qwen3VLForConditionalGeneration",
    "torch_dtype": "bfloat16",
    "window_size": 8,
    "max_sampled_frames": 512,
    "max_new_tokens": 4096,
    "temperature": 0.0,
    "tp": 1,
    "enable_value_head": False,
    "use_lora": False,
    "frame_selection": "evqa.inference.build_uniform_indices",
    "window_policy": "evqa.inference.build_window_values-left-repeat",
    "prompt_template": "evqa/templates/procedural_c.txt",
    "prompt_sha256": PROMPT_SHA256,
    "image_boundary": "RGB-to-JPEG-quality-95",
    "progress_parser": "evqa.inference.extract_progress-clamp-0-100",
    "parse_failure": "carry-previous-initial-0",
}

RUNTIME_REQUIREMENTS = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "transformers": "4.57.3",
    "vllm": "0.18.1",
    "qwen-vl-utils": "0.0.14",
    "numpy": "2.2.6",
    "Pillow": "12.0.0",
    "moviepy": "1.0.3",
}


def _utc_now() -> str:
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


def _snapshot_revision(path: Path) -> str | None:
    parts = path.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            return parts[index + 1]
    return None


def _require_pinned_checkout(root: Path, *, allow_unverified: bool = False) -> dict:
    root = root.expanduser().resolve()
    required = (
        root / "evqa" / "inference.py",
        root / "evqa" / "model.py",
        root / "evqa" / "templates" / "procedural_c.txt",
        root / "core" / "backends" / "vllm.py",
        root / "uv.lock",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"incomplete ProcVLM checkout: {root}")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    actual = revision.stdout.strip()
    source_verified = revision.returncode == 0 and actual == UPSTREAM_COMMIT
    dirty = subprocess.run(
        # core.fileMode=false: a permission-only change is not a source modification.
        ["git", "-c", "core.fileMode=false", "-C", str(root),
         "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    )
    tracked_clean = dirty.returncode == 0 and not dirty.stdout.strip()
    prompt_sha256 = _sha256_file(required[2])
    prompt_verified = prompt_sha256 == PROMPT_SHA256
    if not allow_unverified:
        if not source_verified:
            raise RuntimeError(
                f"ProcVLM checkout is not at the pinned commit {UPSTREAM_COMMIT}: {actual!r}"
            )
        if not tracked_clean:
            raise RuntimeError("the official ProcVLM checkout has tracked modifications; refusing to start")
        if not prompt_verified:
            raise RuntimeError("ProcVLM procedural_c prompt content drifted")
    return {
        "repo": UPSTREAM_REPO,
        "root": str(root),
        "commit": actual,
        "expected_commit": UPSTREAM_COMMIT,
        "source_verified": source_verified and tracked_clean and prompt_verified,
        "tracked_clean": tracked_clean,
        "prompt_sha256": prompt_sha256,
    }


def _environment_identity(*, allow_unverified: bool = False) -> dict:
    import importlib.metadata as metadata

    packages: dict[str, str | None] = {}
    failures: list[str] = []
    for name, expected in RUNTIME_REQUIREMENTS.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        packages[name] = actual
        if actual != expected:
            failures.append(f"{name}=={actual!r}, expected {expected}")
    try:
        packages["flash-attn"] = metadata.version("flash-attn")
    except metadata.PackageNotFoundError:
        packages["flash-attn"] = None
    python_version = platform.python_version()
    if sys.version_info[:2] != (3, 10):
        failures.append(f"Python=={python_version}, the official environment requires 3.10")
    if failures and not allow_unverified:
        raise RuntimeError("isolated ProcVLM environment drifted: " + "; ".join(failures))
    # Never touch ``torch.cuda`` in this process before vLLM starts.  The
    # upstream async engine forks EngineCore, and a parent-side CUDA probe would
    # make that fork fail with "Cannot re-initialize CUDA".  A short child gives
    # us build metadata without contaminating the long-lived server process.
    torch_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'build': torch.__version__, "
                "'cuda': torch.version.cuda}))"
            ),
        ],
        capture_output=True,
        text=True,
    )
    try:
        torch_identity = json.loads(torch_probe.stdout) if torch_probe.returncode == 0 else {}
    except json.JSONDecodeError:
        torch_identity = {}
    return {
        "python": python_version,
        "executable": sys.executable,
        "packages": packages,
        "torch_build": torch_identity.get("build"),
        "torch_cuda": torch_identity.get("cuda"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "verified": not failures,
        # flash-attn is required by the optional Transformers/value-head path,
        # but the released default here is the upstream vLLM text path.
        "flash_attn_required_for_profile": False,
    }


def _checkpoint_identity(path: Path, *, allow_unverified: bool = False) -> dict:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"ProcVLM checkpoint directory does not exist: {path}")
    revision = _snapshot_revision(path)
    support_files = (
        "config.json",
        "generation_config.json",
        "chat_template.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    missing = [name for name in support_files if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"ProcVLM checkpoint lacks files: {missing}")
    broken = [str(item) for item in path.rglob("*") if item.is_symlink() and not item.exists()]
    if broken:
        raise RuntimeError(f"ProcVLM checkpoint contains broken links: {broken[:5]}")

    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        index = json.loads(
            (path / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid ProcVLM checkpoint JSON: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("ProcVLM safetensors index lacks a non-empty weight_map")
    mapped_shards = sorted(set(weight_map.values()))
    expected_shards = sorted(name for name in WEIGHT_FILES if name.endswith(".safetensors"))
    if mapped_shards != expected_shards:
        raise RuntimeError(
            f"ProcVLM safetensors shards drifted: {mapped_shards}, expected {expected_shards}"
        )

    files: dict[str, dict] = {}
    failures: list[str] = []
    for name, expected in WEIGHT_FILES.items():
        file_path = path / name
        if not file_path.is_file():
            failures.append(f"missing {name}")
            files[name] = {"sha256": None, "size_bytes": None}
            continue
        actual = {
            "sha256": _sha256_file(file_path),
            "size_bytes": file_path.stat().st_size,
        }
        files[name] = actual
        if actual != expected:
            failures.append(f"{name} content mismatch")
    if revision != WEIGHTS_REVISION:
        failures.append(f"snapshot revision={revision!r}")
    if config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        failures.append("config.architectures drifted")
    if config.get("model_type") != "qwen3_vl":
        failures.append("config.model_type drifted")
    if failures and not allow_unverified:
        raise RuntimeError("ProcVLM checkpoint identity check failed: " + "; ".join(failures))
    return {
        "repo_id": WEIGHTS_REPO,
        "path": str(path),
        "expected_revision": WEIGHTS_REVISION,
        "snapshot_revision": revision,
        "files": files,
        "content_verified": not failures,
        "pinned_public_snapshot": revision == WEIGHTS_REVISION,
        "safetensors_shards": mapped_shards,
        "safetensors_tensor_count": len(weight_map),
        "architecture": config.get("architectures"),
        "model_type": config.get("model_type"),
    }


def _decode_request(body: bytes) -> dict:
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as archive:
            required = {"primary", "frame_indices", "instruction"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"NPZ lacks keys: {sorted(missing)}")
            frames = np.asarray(archive["primary"])
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            instruction_value = np.asarray(archive["instruction"])
            fps_value = np.asarray(archive["fps"]) if "fps" in archive.files else None
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid ProcVLM NPZ: {exc}") from exc
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"primary must be [T,H,W,3] uint8 RGB, actual {frames.shape}/{frames.dtype}")
    if len(frames) == 0:
        raise ValueError("primary must not be empty")
    if frame_indices.shape != (len(frames),):
        raise ValueError("frame_indices must have the same length as primary")
    if np.any(frame_indices < 0) or (
        len(frame_indices) > 1 and np.any(np.diff(frame_indices) != 1)
    ):
        raise ValueError("frame_indices must be a non-negative, strictly contiguous complete-episode grid")
    if instruction_value.ndim != 0 or instruction_value.dtype.kind not in {"U", "S"}:
        raise ValueError("instruction must be a string scalar")
    instruction = str(instruction_value.item()).strip()
    if not instruction:
        raise ValueError("instruction must not be empty")
    fps = None
    if fps_value is not None:
        if fps_value.ndim != 0:
            raise ValueError("fps must be a scalar")
        fps = float(fps_value.item())
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
    return {
        "frames": np.ascontiguousarray(frames),
        "frame_indices": frame_indices,
        "instruction": instruction,
        "fps": fps,
    }


class OfficialProcVLMRuntime:
    def __init__(
        self,
        *,
        upstream_root: Path,
        model_path: Path,
        work_dir: Path,
        gpu_memory_utilization: float,
        max_model_len: int,
        process_workers: int,
        pipeline_chunk_size: int,
    ):
        self.upstream_root = upstream_root
        self.model_path = model_path
        self.work_dir = work_dir
        self.engine_kwargs = {
            "dtype": "bfloat16",
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
        }
        self.pipeline_kwargs = {
            **self.engine_kwargs,
            "process_workers": process_workers,
            "pipeline_chunk_size": pipeline_chunk_size,
        }
        self.pipeline_chunk_size = pipeline_chunk_size
        self._lock = threading.Lock()
        self._functions: dict[str, Any] = {}
        self._engine: Any | None = None

    def load(self) -> None:
        root = str(self.upstream_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from core.backends.vllm import get_vllm
        from evqa.inference import (
            build_uniform_indices,
            build_window_values,
            extract_progress,
            extract_reasoning_text,
            load_prompt_template,
            run_batch_progress_inference,
        )

        # Initialize the exact upstream cached vLLM engine before advertising
        # readiness. Capacity knobs affect allocation only, not model semantics.
        self._engine, _ = get_vllm(
            str(self.model_path),
            tp=PROFILE_FIELDS["tp"],
            engine_kwargs=dict(self.engine_kwargs),
        )
        self._functions = {
            "build_uniform_indices": build_uniform_indices,
            "build_window_values": build_window_values,
            "extract_progress": extract_progress,
            "extract_reasoning_text": extract_reasoning_text,
            "load_prompt_template": load_prompt_template,
            "run_batch_progress_inference": run_batch_progress_inference,
        }

    def close(self) -> None:
        """Stop vLLM before its upstream atexit handler stops the event loop."""
        engine, self._engine = self._engine, None
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def score(self, body: bytes) -> dict:
        request = _decode_request(body)
        frames = request["frames"]
        frame_indices = request["frame_indices"]
        count = min(len(frames), PROFILE_FIELDS["max_sampled_frames"])
        positions = self._functions["build_uniform_indices"](len(frames), count)
        selected_indices = [int(frame_indices[position]) for position in positions]
        prompt = self._functions["load_prompt_template"](request["instruction"])
        self.work_dir.mkdir(parents=True, exist_ok=True)

        with self._lock, tempfile.TemporaryDirectory(
            prefix="request-", dir=self.work_dir
        ) as temporary:
            frame_root = Path(temporary)
            selected_paths: list[str] = []
            for order, position in enumerate(positions):
                frame_path = frame_root / f"frame_{order:06d}.jpg"
                Image.fromarray(frames[position]).save(str(frame_path), quality=95)
                selected_paths.append(str(frame_path))

            accumulated_paths: list[str] = []
            accumulated_indices: list[int] = []
            items: list[dict] = []
            window_indices: list[list[int]] = []
            for path, frame_index in zip(
                selected_paths, selected_indices, strict=True
            ):
                accumulated_paths.append(path)
                accumulated_indices.append(frame_index)
                window_paths = self._functions["build_window_values"](
                    accumulated_paths,
                    len(accumulated_paths) - 1,
                    PROFILE_FIELDS["window_size"],
                )
                window = self._functions["build_window_values"](
                    accumulated_indices,
                    len(accumulated_indices) - 1,
                    PROFILE_FIELDS["window_size"],
                )
                window_indices.append([int(value) for value in window])
                items.append(
                    {
                        "image": window_paths,
                        "conversations": [{"from": "human", "value": prompt}],
                    }
                )

            answers = self._functions["run_batch_progress_inference"](
                batch_items=items,
                model_path=str(self.model_path),
                max_new_tokens=PROFILE_FIELDS["max_new_tokens"],
                temperature=PROFILE_FIELDS["temperature"],
                enable_value_head=PROFILE_FIELDS["enable_value_head"],
                use_lora=PROFILE_FIELDS["use_lora"],
                torch_dtype="bf16",
                tp=PROFILE_FIELDS["tp"],
                sampling_kwargs=None,
                engine_kwargs=dict(self.pipeline_kwargs),
            )
        if len(answers) != len(selected_indices):
            raise RuntimeError(
                f"official ProcVLM returned {len(answers)} answers, expected {len(selected_indices)}"
            )

        records: list[dict] = []
        progress_values: list[float] = []
        previous = 0.0
        for sample_index, answer in enumerate(answers):
            parsed = self._functions["extract_progress"](answer)
            if parsed is None:
                progress = previous
                source = "previous"
            else:
                progress = float(parsed)
                previous = progress
                source = "model"
            progress_values.append(progress)
            records.append(
                {
                    "sample_index": sample_index,
                    "source_position": int(positions[sample_index]),
                    "frame_index": selected_indices[sample_index],
                    "window_frame_indices": window_indices[sample_index],
                    "progress": progress,
                    "parsed_progress": None if parsed is None else float(parsed),
                    "progress_source": source,
                    "reasoning": self._functions["extract_reasoning_text"](answer),
                    "model_output": str(answer),
                }
            )
        return {
            "schema": SCORE_SCHEMA,
            "profile": PROFILE,
            "source_frame_count": len(frames),
            "source_positions": [int(value) for value in positions],
            "frame_indices": selected_indices,
            "progress": progress_values,
            "records": records,
            "vllm_batches": math.ceil(len(items) / self.pipeline_chunk_size),
        }


class ProcVLMHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, runtime, model_info):
        super().__init__(address, handler)
        self.runtime = runtime
        self.model_info = model_info


class Handler(BaseHTTPRequestHandler):
    server_version = "RewardBoxProcVLM/1"

    def log_message(self, fmt: str, *args) -> None:
        message = fmt % args
        sys.stderr.write(
            f"{self.address_string()} - - [{self.log_date_time_string()}] {message}\n"
        )

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "healthy", "reward_model": "procvlm"})
        elif self.path == "/model_info":
            self._send_json(200, self.server.model_info)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/score_npz":
            self._send_json(404, {"error": "not found"})
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/x-npz":
            self._send_json(415, {"error": "Content-Type must be application/x-npz"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(411, {"error": "valid Content-Length required"})
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "request size out of bounds"})
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._send_json(400, {"error": "truncated request body"})
            return
        try:
            result = self.server.runtime.score(body)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(500, {"error": f"ProcVLM inference failed: {exc}"})
            return
        self._send_json(200, result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=_default_checkout("ProcVLM"),
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--process-workers", type=int, default=16)
    parser.add_argument("--pipeline-chunk-size", type=int, default=8)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=repo_root / ".hf" / "runtime" / "procvlm" / "requests",
    )
    parser.add_argument(
        "--provenance-file",
        type=Path,
        default=repo_root / ".hf" / "runtime" / "procvlm" / "server-provenance.json",
    )
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be in (0,1)")
    if args.max_model_len < PROFILE_FIELDS["max_new_tokens"] + 1024:
        parser.error("--max-model-len is too small for the official generation budget")
    if args.process_workers < 1 or args.pipeline_chunk_size < 1:
        parser.error("worker and chunk sizes must be positive")
    return args


def main() -> None:
    args = _parse_args()
    upstream = _require_pinned_checkout(
        args.upstream_root, allow_unverified=args.allow_unverified
    )
    environment = _environment_identity(allow_unverified=args.allow_unverified)
    weights = _checkpoint_identity(
        args.model_path, allow_unverified=args.allow_unverified
    )
    runtime_engine = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "process_workers": args.process_workers,
        "pipeline_chunk_size": args.pipeline_chunk_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "status": "loading",
        "started_at": _utc_now(),
        "upstream": upstream,
        "weights": weights,
        "environment": environment,
        "profile": PROFILE_FIELDS,
        "runtime_engine": runtime_engine,
        "endpoint": {"host": args.host, "port": args.port},
    }
    _atomic_write_json(args.provenance_file.expanduser().resolve(), provenance)

    runtime = OfficialProcVLMRuntime(
        upstream_root=args.upstream_root.expanduser().resolve(),
        model_path=args.model_path.expanduser().resolve(),
        work_dir=args.work_dir.expanduser().resolve(),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        process_workers=args.process_workers,
        pipeline_chunk_size=args.pipeline_chunk_size,
    )
    try:
        runtime.load()
    except Exception as exc:
        provenance.update(status="error", finished_at=_utc_now(), error=str(exc))
        _atomic_write_json(args.provenance_file.expanduser().resolve(), provenance)
        raise

    provenance.update(status="ready", ready_at=_utc_now())
    _atomic_write_json(args.provenance_file.expanduser().resolve(), provenance)
    model_info = {
        "reward_model": "procvlm",
        "upstream_commit": UPSTREAM_COMMIT,
        "weights": {
            "repo_id": WEIGHTS_REPO,
            "revision": WEIGHTS_REVISION,
            "files": weights["files"],
        },
        "profile": PROFILE_FIELDS,
        "runtime_engine": runtime_engine,
        "environment": environment,
    }
    server = ProcVLMHTTPServer(
        (args.host, args.port), Handler, runtime=runtime, model_info=model_info
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "url": f"http://{args.host}:{args.port}",
                "provenance_file": str(args.provenance_file.expanduser().resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.server_close()
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
