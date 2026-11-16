"""Static and preflight contracts for the RynnValue server wrapper."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_rynnvalue_server as server  # noqa: E402


def _checkpoint(path: Path, *, attention="pred_slot_isolated_eager") -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "rynn_value_lang",
                "use_meta": True,
                "attn_implementation": attention,
                "value_tokenizer_config": {
                    "min_value": 0.0,
                    "max_value": 512.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "configuration_rynn_value_lang.py").write_text(
        'if attn_implementation is None:\n'
        '    attn_implementation = "pred_slot_isolated_eager"\n'
        'self._attn_implementation = attn_implementation\n',
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"weights")
    return path


class TestPinnedSource:
    @pytest.mark.upstream("rynnvalue")
    def test_checkout_is_exact_official_commit(self):
        root = server._repo_root()
        assert server._require_pinned_checkout(root) == server.UPSTREAM_COMMIT

    @pytest.mark.upstream("rynnvalue")
    def test_official_launch_files_lock_reproduction_profile(self):
        root = server._repo_root()
        readme = (root / "README.md").read_text(encoding="utf-8")
        baseline = (
            root
            / "robometer"
            / "robometer"
            / "evals"
            / "baselines"
            / "rynnvalue.py"
        ).read_text(encoding="utf-8")
        assert "'max_frames=8'" in readme
        assert "'custom_eval.use_frame_steps=false'" in readme
        assert "'custom_eval.pad_frames=false'" in readme
        assert "'model_config.stride=2'" in readme
        assert "'model_config.num_frames=8'" in readme
        assert 'if mode == "absolute"' in baseline
        assert "max_new_tokens=self.max_new_tokens" in baseline
        assert "do_sample=False" in baseline
        assert "num_beams=1" in baseline


class TestServerPreflight:
    def test_server_python_path_does_not_dereference_venv_symlink(self, tmp_path):
        base = tmp_path / "python3.10"
        base.write_text("", encoding="utf-8")
        venv_python = tmp_path / "venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.symlink_to(base)
        actual = server._server_python_path(venv_python)
        assert actual == venv_python.absolute()
        assert actual != venv_python.resolve()

    def test_pinned_hub_snapshot_is_accepted(self, tmp_path):
        expected = server.MODELS["8b"]
        cache_name = "models--" + expected["model_id"].replace("/", "--")
        path = _checkpoint(
            tmp_path / cache_name / "snapshots" / expected["revision"]
        )
        identity = server._checkpoint_identity(
            path, expected, allow_unverified=False
        )
        assert identity["pinned_public_snapshot"] is True
        assert identity["snapshot_revision"] == expected["revision"]

    def test_copied_weights_require_explicit_unverified_flag(self, tmp_path):
        expected = server.MODELS["4b"]
        path = _checkpoint(tmp_path / "copied-model", attention=None)
        with pytest.raises(RuntimeError, match="--allow-unverified-weights"):
            server._checkpoint_identity(path, expected, allow_unverified=False)
        identity = server._checkpoint_identity(path, expected, allow_unverified=True)
        assert identity["pinned_public_snapshot"] is False
        assert identity["config"]["exported_attn_implementation"] is None
        assert identity["config"]["effective_attn_implementation"] == (
            "pred_slot_isolated_eager"
        )

    def test_revision_named_directory_outside_expected_hub_cache_is_rejected(
        self, tmp_path
    ):
        expected = server.MODELS["8b"]
        path = _checkpoint(tmp_path / "snapshots" / expected["revision"])
        with pytest.raises(RuntimeError, match="Hub snapshot"):
            server._checkpoint_identity(path, expected, allow_unverified=False)

    def test_snapshot_attention_or_value_support_drift_is_rejected(self, tmp_path):
        expected = server.MODELS["8b"]
        cache_name = "models--" + expected["model_id"].replace("/", "--")
        path = _checkpoint(
            tmp_path / cache_name / "snapshots" / expected["revision"],
            attention="sdpa",
        )
        with pytest.raises(RuntimeError, match="snapshot config"):
            server._checkpoint_identity(path, expected, allow_unverified=False)

    @pytest.mark.upstream("rynnvalue")
    def test_command_explicitly_selects_whole_trajectory_official_profile(self):
        root = server._repo_root()
        command = server._build_command(
            root=root,
            python=Path("/tmp/rynn-python"),
            model_path=Path("/tmp/model"),
            profile="official-isolated",
            camera_lookup=root / "robometer" / "extracted_meta_with_descriptions.json",
            host="127.0.0.1",
            port=8123,
        )
        assert "max_frames=8" in command
        assert "use_frame_steps=false" in command
        assert "custom_eval.use_frame_steps=false" in command
        assert "custom_eval.pad_frames=false" in command
        assert "model_config.checkpoint_path=null" in command
        assert "model_config.mode=absolute" in command
        assert "model_config.stride=2" in command
        assert "model_config.num_frames=8" in command
        assert not any("attn_implementation=" in item for item in command)

        eager = server._build_command(
            root=root,
            python=Path("/tmp/rynn-python"),
            model_path=Path("/tmp/model"),
            profile="paper-eager",
            camera_lookup=root / "robometer" / "extracted_meta_with_descriptions.json",
            host="127.0.0.1",
            port=8123,
        )
        assert "model_config.attn_implementation=eager" in eager

    def test_ready_identity_rejects_mode_drift(self):
        info = {
            "reward_model": "rynnvalue",
            "config": {
                "max_frames": 8,
                "use_frame_steps": False,
                "custom_eval": {
                    "use_frame_steps": False,
                    "pad_frames": False,
                },
                "model_config": {
                    "mode": "normalized",
                    "stride": 2,
                    "num_frames": 8,
                    "max_new_tokens": 128,
                    "use_fuse": False,
                    "attn_implementation": None,
                },
            },
        }
        with pytest.raises(RuntimeError, match="requested profile"):
            server._validate_ready_identity(info, "official-isolated")

    def test_ready_identity_rejects_padding_profile_drift(self):
        info = {
            "reward_model": "rynnvalue",
            "config": {
                "max_frames": 8,
                "use_frame_steps": False,
                "custom_eval": {
                    "use_frame_steps": False,
                    "pad_frames": True,
                },
                "model_config": {
                    "mode": "absolute",
                    "stride": 2,
                    "num_frames": 8,
                    "max_new_tokens": 128,
                    "use_fuse": False,
                    "attn_implementation": None,
                },
            },
        }
        with pytest.raises(RuntimeError, match="pad_frames"):
            server._validate_ready_identity(info, "official-isolated")
