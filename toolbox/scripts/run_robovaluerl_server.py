#!/usr/bin/env python3
"""Run the pinned Robo-ValueRL value estimator in an isolated environment.

This script must be executed with the dedicated Robo-ValueRL interpreter, not
RewardBox's shared ``.venv``.  It imports the upstream model, transforms,
tokenizer, HL-Gauss decoder, and interpolation function without copying their
model logic into RewardBox.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from types import SimpleNamespace

import numpy as np

UPSTREAM_COMMIT = "1618bcb01212eb747a5d2659228b7fb5db87a563"
WEIGHTS_REPO = "X-Humanoid/Robo-ValueRL"
WEIGHTS_REVISION = "b44f33eaa5892233174d596cc7d8a7e685967eef"
WEIGHTS_FILE = "value_estimator/model.safetensors"
WEIGHTS_SHA256 = "97c8d7060349fbc675a76cd43006fc52267647cbe67e092096ceac78d550c09b"
WEIGHTS_SIZE = 6_267_669_712
PROFILE_NAME = "official-release-short-history-v1"
PROVENANCE_SCHEMA = "rewardbox.robovaluerl-server-provenance.v1"
SCORE_SCHEMA = "rewardbox.robovaluerl-score.v1"

PROFILE = {
    "name": PROFILE_NAME,
    "config_name": "all_value_function_training",
    "history_length": 5,
    "stride": 10,
    "interpolation_method": "cubic",
    "smooth": True,
    "smooth_window": 21,
    "smooth_polyorder": 3,
    "bc_type": "natural",
    "num_bins": 256,
    "remaining_steps_min": 0.0,
    "remaining_steps_max": 5000.0,
    "hl_gauss_sigma": 3.0,
    "progress_transform": "clip(1-remaining_steps/5000,0,1)",
    "history_view": "primary",
    "views": {
        "primary": "base_0_rgb",
        "left_wrist": "left_wrist_0_rgb",
        "right_wrist": "right_wrist_0_rgb",
    },
    "released_output_field": "remain_time",
    "config_loader": "exact-release-Pi0Config-fields",
    "interpolation_loader": "AST-executed-exact-pinned-functions",
}

REQUIRED_SOURCE_FILES = (
    "robo_valuerl/agents/openpi_x_humanoid_critic.py",
    "robo_valuerl/generate_action_quality/annotate_value_function_sparse_part.py",
    "robo_valuerl/transforms/transforms.py",
    "robo_valuerl/transforms/tokenizer.py",
    "openpi/src/openpi/training/config.py",
    "openpi/src/openpi/models/hl_gauss_distribution.py",
    "openpi/src/openpi/models_pytorch/x_humanoid_pi0_pytorch_critic_with_history.py",
    "openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/check.py",
)


class InputError(ValueError):
    """A malformed client request, suitable for HTTP 400."""


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_output(root: Path, *args: str) -> str:
    # core.fileMode=false: a permission-only change is not a source modification.
    completed = subprocess.run(
        ["git", "-c", "core.fileMode=false", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_upstream(root: Path, *, allow_dirty: bool) -> dict:
    root = root.expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"upstream-root is not a Git checkout: {root}")
    head = git_output(root, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Robo-ValueRL upstream commit drifted: {head} != {UPSTREAM_COMMIT}"
        )
    missing = [name for name in REQUIRED_SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            "incomplete Robo-ValueRL checkout, missing pinned source files: "
            + ", ".join(missing)
        )
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
    dirty_paths = git_output(root, "diff", "--name-only", "HEAD").splitlines()
    source_dirty_paths = sorted(set(dirty_paths) & set(REQUIRED_SOURCE_FILES))
    if source_dirty_paths and not allow_dirty:
        raise RuntimeError(
            "Robo-ValueRL pinned inference source was modified: "
            + ", ".join(source_dirty_paths)
            + "; pass --allow-dirty-upstream explicitly only for diagnostics"
        )
    source_hashes = {
        name: sha256_file(root / name) for name in REQUIRED_SOURCE_FILES
    }
    return {
        "repo": "https://github.com/Open-X-Humanoid/Robo-ValueRL",
        "root": str(root),
        "commit": head,
        "tracked_worktree_clean": not bool(dirty),
        "tracked_dirty_paths": dirty_paths,
        "required_source_files_clean": not bool(source_dirty_paths),
        "source_verified": (
            head == UPSTREAM_COMMIT
            and not bool(missing)
            and (not source_dirty_paths or allow_dirty)
        ),
        "required_source_sha256": source_hashes,
    }


def validate_weights(
    checkpoint_dir: Path,
    *,
    declared_revision: str,
    allow_unverified: bool,
) -> dict:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    model_path = checkpoint_dir / "model.safetensors"
    if not model_path.is_file():
        raise RuntimeError(
            f"no model.safetensors under checkpoint-dir: {checkpoint_dir}"
        )
    size = model_path.stat().st_size
    digest = sha256_file(model_path)
    revision_ok = declared_revision == WEIGHTS_REVISION
    content_ok = size == WEIGHTS_SIZE and digest == WEIGHTS_SHA256
    if (not revision_ok or not content_ok) and not allow_unverified:
        raise RuntimeError(
            "Robo-ValueRL weight pin check failed: "
            + json.dumps(
                {
                    "expected_revision": WEIGHTS_REVISION,
                    "declared_revision": declared_revision,
                    "expected_size": WEIGHTS_SIZE,
                    "actual_size": size,
                    "expected_sha256": WEIGHTS_SHA256,
                    "actual_sha256": digest,
                },
                sort_keys=True,
            )
        )
    return {
        "repo_id": WEIGHTS_REPO,
        "expected_revision": WEIGHTS_REVISION,
        "snapshot_revision": declared_revision if revision_ok else None,
        "filename": WEIGHTS_FILE,
        "local_checkpoint_dir": str(checkpoint_dir),
        "local_model_path": str(model_path),
        "size_bytes": size,
        "sha256": digest,
        "content_verified": content_ok,
        "pinned_public_snapshot": revision_ok and content_ok,
    }


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def key_frame_indices(length: int, stride: int = 10) -> list[int]:
    if length <= 0:
        raise InputError("an episode needs at least 1 frame")
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def decode_npz(body: bytes) -> dict[str, Any]:
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as archive:
            expected = {
                "primary",
                "left_wrist",
                "right_wrist",
                "frame_indices",
                "instruction",
            }
            actual = set(archive.files)
            if actual != expected:
                raise InputError(
                    f"NPZ fields must be exactly {sorted(expected)}, actual {sorted(actual)}"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in expected}
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"cannot decode the allow_pickle=False NPZ: {exc}") from exc

    views = {name: arrays[name] for name in ("primary", "left_wrist", "right_wrist")}
    lengths: set[int] = set()
    for name, frames in views.items():
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise InputError(
                f"{name} must be [T,H,W,3] uint8 RGB, actual {frames.shape}/{frames.dtype}"
            )
        if len(frames) == 0:
            raise InputError(f"{name} must not be empty")
        lengths.add(len(frames))
    if len(lengths) != 1:
        raise InputError("the three views must have the same frame count")
    length = lengths.pop()
    frame_indices = arrays["frame_indices"]
    if frame_indices.dtype.kind not in {"i", "u"} or frame_indices.shape != (length,):
        raise InputError("frame_indices must be a 1-D integer array of length T")
    frame_indices = frame_indices.astype(np.int64, copy=False)
    expected_indices = np.arange(length, dtype=np.int64)
    if not np.array_equal(frame_indices, expected_indices):
        raise InputError("the official profile needs complete contiguous frame_indices starting at 0")
    instruction_array = arrays["instruction"]
    if instruction_array.ndim != 0 or instruction_array.dtype.kind not in {"U", "S"}:
        raise InputError("instruction must be an NPZ string scalar")
    instruction = str(instruction_array.item()).strip()
    if not instruction:
        raise InputError("instruction must not be empty")
    return {
        "views": views,
        "frame_indices": frame_indices,
        "instruction": instruction,
    }


def load_official_interpolator(source_path: Path):
    """Compile the four interpolation helpers verbatim from pinned source.

    Importing the released annotation module is impossible in a clean runtime:
    it imports ``openpi.training.config``, which in turn loads an author-local
    absolute-path ``.npy`` file at module import time.  AST extraction keeps the
    exact official function bodies and dependencies without executing unrelated
    broken module-level code.
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter

    wanted = {
        "linear_interpolate",
        "cubic_spline_interpolate",
        "savitzky_golay_smooth",
        "interpolate_and_smooth",
    }
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    found = {node.name for node in functions}
    if found != wanted:
        raise RuntimeError(
            f"pinned annotation interpolation functions drifted: {sorted(found)}"
        )
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {
        "np": np,
        "CubicSpline": CubicSpline,
        "savgol_filter": savgol_filter,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["interpolate_and_smooth"]


class RoboValueRuntime:
    """Official-model owner.  All heavy imports stay inside this constructor."""

    def __init__(self, upstream_root: Path, checkpoint_dir: Path, *, allow_cpu: bool):
        openpi_src = upstream_root / "openpi" / "src"
        robo_src = upstream_root / "robo_valuerl"
        sys.path.insert(0, str(openpi_src))
        sys.path.insert(0, str(robo_src))

        import albumentations as A
        import torch
        from openpi.models.hl_gauss_distribution import HLGaussDistribution
        from openpi.models.pi0_config import Pi0Config
        from agents.openpi_x_humanoid_critic import OpenPi_X_Humanoid_Critic
        import transforms.transforms as official_transforms
        from transforms.tokenizer import PaligemmaTokenizer

        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError(
                "the official 6.3GB Robo-ValueRL checkpoint needs CUDA; pass --allow-cpu only for diagnostics"
            )
        self.torch = torch
        self.lock = threading.Lock()
        self.interpolate_and_smooth = load_official_interpolator(
            upstream_root
            / "robo_valuerl"
            / "generate_action_quality"
            / "annotate_value_function_sparse_part.py"
        )
        self.decoder = HLGaussDistribution(
            torch.linspace(0, 5000, PROFILE["num_bins"]),
            sigma=PROFILE["hl_gauss_sigma"],
        )

        # Exact fields from the pinned ``all_value_function_training`` block.
        # A SimpleNamespace is sufficient because the official critic consumes
        # only these TrainConfig attributes at inference construction time.
        model_config = Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_100m",
            consistency=False,
            action_dim=32,
            freeze_paligemma=True,
            num_value_bins=256,
            use_self_preprocess=False,
            use_history_image=True,
            only_predict_remain_time=True,
        )
        config = SimpleNamespace(
            model=model_config,
            pytorch_training_precision="bfloat16",
            pytorch_weight_path=str(checkpoint_dir),
            only_load_paligemma=False,
            history_length=PROFILE["history_length"],
        )
        data_transforms = official_transforms.compose(
            [
                official_transforms.InjectDefaultPrompt(
                    prompt="separate the blocks and sort by color into different plates"
                ),
                official_transforms.TokenizePrompt(PaligemmaTokenizer(max_len=48)),
                official_transforms.ResizeImages(256, 256),
                official_transforms.PadStatesAndActions(config.model.action_dim),
            ]
        )
        image_transforms = A.Compose(
            [A.CenterCrop(height=224, width=224, p=1.0)]
        )
        self.agent = OpenPi_X_Humanoid_Critic(
            config,
            chunk_size=10,
            data_transforms=data_transforms,
            image_transforms=image_transforms,
        )
        self.agent.eval()
        self.config = config
        self.device = str(self.agent.device)
        from safetensors import safe_open

        weight_path = checkpoint_dir / "model.safetensors"
        model_state = self.agent.model.state_dict()
        model_shapes = {
            name: list(tensor.shape) for name, tensor in model_state.items()
        }
        with safe_open(weight_path, framework="pt", device="cpu") as archive:
            weight_keys = set(archive.keys())
            weight_shapes = {
                name: list(archive.get_slice(name).get_shape()) for name in weight_keys
            }
        model_keys = set(model_shapes)
        missing_model_keys = model_keys - weight_keys
        tied_aliases: dict[str, list[str]] = {}
        for missing_name in sorted(missing_model_keys):
            missing_tensor = model_state[missing_name]
            aliases = [
                loaded_name
                for loaded_name in sorted(model_keys & weight_keys)
                if model_state[loaded_name].untyped_storage().data_ptr()
                == missing_tensor.untyped_storage().data_ptr()
                and list(model_state[loaded_name].shape)
                == list(missing_tensor.shape)
            ]
            if aliases:
                tied_aliases[missing_name] = aliases
        effective_missing = sorted(missing_model_keys - set(tied_aliases))
        shape_mismatches = {
            name: {"model": model_shapes[name], "weight": weight_shapes[name]}
            for name in sorted(model_keys & weight_keys)
            if model_shapes[name] != weight_shapes[name]
        }
        self.load_contract = {
            "model_state_key_count": len(model_keys),
            "weight_key_count": len(weight_keys),
            "missing_model_keys": sorted(missing_model_keys),
            "missing_keys_satisfied_by_tied_weight_alias": tied_aliases,
            "effective_missing_model_keys": effective_missing,
            "unexpected_weight_keys": sorted(weight_keys - model_keys),
            "shape_mismatches": shape_mismatches,
            "all_model_keys_present_with_matching_shapes": not (
                effective_missing or shape_mismatches
            ),
        }

    def score(self, request: dict[str, Any]) -> dict:
        views = request["views"]
        length = len(views["primary"])
        indices = key_frame_indices(length, PROFILE["stride"])
        probability_rows: list[list[float]] = []
        decoded: dict[int, float] = {}

        with self.lock, self.torch.inference_mode():
            for index in indices:
                history_index = max(0, index - PROFILE["history_length"])
                batch = {
                    "image": {
                        "base_0_rgb": np.expand_dims(views["primary"][index], 0),
                        "left_wrist_0_rgb": np.expand_dims(
                            views["left_wrist"][index], 0
                        ),
                        "right_wrist_0_rgb": np.expand_dims(
                            views["right_wrist"][index], 0
                        ),
                    },
                    "history_image": np.expand_dims(
                        views["primary"][history_index], 0
                    ),
                    # Required structurally by Observation.from_dict.  The
                    # released value forward extracts state but never consumes
                    # it in prefix/suffix computation.
                    "state": np.zeros((1, 16), dtype=np.float32),
                    "prompt": request["instruction"],
                    "image_mask": {
                        "base_0_rgb": np.asarray([True]),
                        "left_wrist_0_rgb": np.asarray([True]),
                        "right_wrist_0_rgb": np.asarray([True]),
                    },
                    "frame_index": np.asarray([index], dtype=np.int64),
                }
                prediction = self.agent.predict_value(batch)
                if set(prediction) != {"remain_time"}:
                    raise RuntimeError(
                        "released critic output contract drifted: "
                        f"expected ['remain_time'], actual {sorted(prediction)}"
                    )
                probabilities = (
                    prediction["remain_time"].detach().to(dtype=self.torch.float32).cpu()
                )
                if tuple(probabilities.shape) != (1, PROFILE["num_bins"]):
                    raise RuntimeError(
                        "remain_time distribution shape drifted: "
                        f"{tuple(probabilities.shape)}"
                    )
                row = probabilities.numpy()[0]
                probability_rows.append(row.astype(np.float64).tolist())
                decoded[index] = float(self.decoder.batch_decode(probabilities).item())

        remaining = np.asarray(
            self.interpolate_and_smooth(
                decoded,
                length,
                method=PROFILE["interpolation_method"],
                smooth=PROFILE["smooth"],
                smooth_window=PROFILE["smooth_window"],
                smooth_polyorder=PROFILE["smooth_polyorder"],
                bc_type=PROFILE["bc_type"],
            ),
            dtype=np.float64,
        )
        if remaining.shape != (length,) or not np.all(np.isfinite(remaining)):
            raise RuntimeError("official interpolation returned invalid remaining_steps")
        progress = np.clip(
            1.0 - remaining / PROFILE["remaining_steps_max"], 0.0, 1.0
        )
        probability_array = np.asarray(probability_rows, dtype="<f4")
        return {
            "schema": SCORE_SCHEMA,
            "profile": PROFILE_NAME,
            "frame_indices": request["frame_indices"].tolist(),
            "progress": progress.tolist(),
            "remaining_steps": remaining.tolist(),
            "key_frame_indices": indices,
            "key_history_frame_indices": [
                max(0, index - PROFILE["history_length"]) for index in indices
            ],
            "key_remaining_steps": [decoded[index] for index in indices],
            "key_probabilities": probability_rows,
            "key_probabilities_float32_sha256": hashlib.sha256(
                probability_array.tobytes()
            ).hexdigest(),
            "model_forwards": len(indices),
            "state_unused_by_release_value_forward": True,
            "release_annotation_contract": "remain_time-only compatibility path",
        }


class RoboValueHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        runtime: RoboValueRuntime,
        model_info: dict,
        *,
        max_request_bytes: int,
    ):
        self.runtime = runtime
        self.model_info = model_info
        self.max_request_bytes = max_request_bytes
        super().__init__(address, RoboValueHandler)


class RoboValueHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RewardBox-RoboValueRL/1"

    @property
    def app(self) -> RoboValueHTTPServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(
            f"[{utc_now()}] {self.client_address[0]} " + format % args + "\n"
        )

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "healthy",
                    "reward_model": "robo-valuerl",
                    "profile": PROFILE_NAME,
                },
            )
        elif self.path == "/model_info":
            self.send_json(HTTPStatus.OK, self.app.model_info)
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/score_npz":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/x-npz":
            self.send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Content-Type must be application/x-npz"},
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            content_length = -1
        if content_length < 1 or content_length > self.app.max_request_bytes:
            self.send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {
                    "error": "invalid Content-Length",
                    "max_request_bytes": self.app.max_request_bytes,
                },
            )
            return
        try:
            request = decode_npz(self.rfile.read(content_length))
            started = time.perf_counter()
            output = self.app.runtime.score(request)
            output["server_latency_s"] = time.perf_counter() - started
            self.send_json(HTTPStatus.OK, output)
        except InputError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Robo-ValueRL inference failed: {type(exc).__name__}: {exc}"},
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="directory containing the pinned value_estimator/model.safetensors",
    )
    parser.add_argument("--weights-revision", default=WEIGHTS_REVISION)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--provenance-file", type=Path, required=True)
    parser.add_argument(
        "--max-request-bytes", type=int, default=4 * 1024 * 1024 * 1024
    )
    parser.add_argument("--allow-dirty-upstream", action="store_true")
    parser.add_argument("--allow-unverified-weights", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in 1..65535")
    if args.max_request_bytes <= 0:
        raise SystemExit("--max-request-bytes must be > 0")

    provenance: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "status": "validating",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "profile": PROFILE,
        "server": {"host": args.host, "port": args.port},
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "packages_before_model_import": package_versions(
                (
                    "torch",
                    "transformers",
                    "jax",
                    "flax",
                    "safetensors",
                    "albumentations",
                    "scipy",
                    "sentencepiece",
                )
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "compatibility": {
            "annotation_script_requested_fields": ["value", "rtg", "remain_time"],
            "released_history_critic_fields": ["remain_time"],
            "selected_path": "released remain_time plus official decode/interpolation",
            "state_unused_by_release_value_forward": True,
            "training_config_import_blocker": (
                "openpi.policies.x_humaniod_policy loads an author-local absolute-path "
                "HL-Gauss bins .npy during openpi.training.config import"
            ),
            "config_resolution": (
                "construct official Pi0Config from exact pinned "
                "all_value_function_training fields"
            ),
            "interpolation_resolution": (
                "AST-compile exact four helper functions from pinned annotation source"
            ),
        },
    }
    atomic_write_json(args.provenance_file, provenance)
    try:
        provenance["upstream"] = validate_upstream(
            args.upstream_root, allow_dirty=args.allow_dirty_upstream
        )
        provenance["weights"] = validate_weights(
            args.checkpoint_dir,
            declared_revision=args.weights_revision,
            allow_unverified=args.allow_unverified_weights,
        )
        provenance["status"] = "loading"
        atomic_write_json(args.provenance_file, provenance)

        runtime = RoboValueRuntime(
            args.upstream_root.expanduser().resolve(),
            args.checkpoint_dir.expanduser().resolve(),
            allow_cpu=args.allow_cpu,
        )
        provenance["runtime"]["packages_after_model_import"] = package_versions(
            (
                "torch",
                "transformers",
                "jax",
                "flax",
                "safetensors",
                "albumentations",
                "scipy",
                "sentencepiece",
            )
        )
        provenance["runtime"]["device"] = runtime.device
        provenance["weights"]["load_contract"] = runtime.load_contract
        if runtime.torch.cuda.is_available():
            provenance["runtime"]["cuda_device_name"] = (
                runtime.torch.cuda.get_device_name(0)
            )

        model_info = {
            "reward_model": "robo-valuerl",
            "upstream_commit": UPSTREAM_COMMIT,
            "weights": {
                "repo_id": WEIGHTS_REPO,
                "revision": WEIGHTS_REVISION,
                "filename": WEIGHTS_FILE,
                "sha256": WEIGHTS_SHA256,
            },
            "profile": PROFILE,
            "device": runtime.device,
            "output_contract": {
                "schema": SCORE_SCHEMA,
                "progress": "paper clip(1-remaining_steps/5000,0,1)",
                "raw": "remaining steps and key-frame 256-bin distributions",
            },
        }
        server = RoboValueHTTPServer(
            (args.host, args.port),
            runtime,
            model_info,
            max_request_bytes=args.max_request_bytes,
        )
        provenance["status"] = "ready"
        provenance["ready_at"] = utc_now()
        atomic_write_json(args.provenance_file, provenance)

        stopping = threading.Event()

        def request_shutdown(signum: int, _frame: Any) -> None:
            provenance["status"] = "stopping"
            provenance["signal"] = signum
            provenance["stopping_at"] = utc_now()
            atomic_write_json(args.provenance_file, provenance)
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, request_shutdown)
        signal.signal(signal.SIGINT, request_shutdown)
        print(
            json.dumps(
                {
                    "status": "ready",
                    "url": f"http://{args.host}:{args.port}",
                    "provenance_file": str(args.provenance_file.resolve()),
                    "device": runtime.device,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.25)
        server.server_close()
        provenance["status"] = "stopped" if stopping.is_set() else "exited"
        provenance["stopped_at"] = utc_now()
        atomic_write_json(args.provenance_file, provenance)
        return 0
    except Exception as exc:
        provenance["status"] = "failed"
        provenance["failed_at"] = utc_now()
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(args.provenance_file, provenance)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
