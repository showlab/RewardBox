"""rewardbox command line: list / info / doctor / run / bench / fetch-upstream.

Rule 2: this module imports no torch/transformers/vllm/unsloth at top level.  list/info
must run without those packages (doctor exists to diagnose what is missing and must not
force imports that trigger global side effects such as unsloth in its default mode).

Exit codes: 0 success; 2 user input error (unknown model, malformed argument); 1 runtime
failure (missing dependency, inference error, ...).  Machine-readable output (--json,
run results) goes to stdout; human-readable errors/progress go to stderr.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
import sys
from pathlib import Path

from ..core.secrets import sensitive_option_paths

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2

_SAMPLING_ARG_NAMES = ("num_frames", "strategy", "fps", "interval", "resize", "include_last")


# ---------------------------------------------------------------- argument parsing helpers
def _parse_model_opt(pair: str) -> tuple[str, object]:
    if "=" not in pair:
        raise ValueError(f"--model-opt needs the form KEY=JSON_VALUE, got {pair!r}")
    k, v = pair.split("=", 1)
    if not k:
        raise ValueError(f"--model-opt key must not be empty: {pair!r}")
    try:
        val = json.loads(v)
    except json.JSONDecodeError:
        val = v   # not valid JSON: keep it as a plain string, so --model-opt mode=terminal works
    return k, val


def _parse_model_opts(pairs: list[str] | None) -> dict:
    return dict(_parse_model_opt(p) for p in (pairs or []))


def _reject_sensitive_cli_opts(model: str, opts: dict) -> None:
    """Credentials must not appear in shell history or worker process argv."""
    paths = sensitive_option_paths(opts)
    if paths:
        raise ValueError(
            f"--model-opt of {model!r} contains credential fields {paths}; do not put credentials on the command line. "
            "GVL reads GEMINI_API_KEY and jev-* JEV_API_KEY from the environment; the gvl-gemini-3.8-flash*, "
            "vlm-* and jev-* profiles also read the variable named by api_key_env"
        )


def _parse_worker_pythons(pairs: list[str] | None, models: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    selected = set(models)
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(
                f"--worker-python needs the form MODEL=PYTHON_PATH, got {pair!r}")
        model, path = pair.split("=", 1)
        if not model or not path:
            raise ValueError(
                f"--worker-python needs both a model name and a path: {pair!r}")
        if model not in selected:
            raise ValueError(
                f"--worker-python names model {model!r}, which was not selected with --model")
        if model in out:
            raise ValueError(f"--worker-python gives model {model!r} more than once")
        out[model] = path
    return out


def _parse_bench_model_opts(
    pairs: list[str] | None, models: list[str],
) -> dict[str, dict]:
    """Parse repeatable ``MODEL.KEY=JSON_VALUE`` options for a multi-model run."""
    selected = set(models)
    out: dict[str, dict] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(
                f"bench --model-opt needs MODEL.KEY=JSON_VALUE, got {pair!r}")
        target, raw = pair.split("=", 1)
        if "." not in target:
            raise ValueError(
                f"bench --model-opt lacks the model-name prefix, got {pair!r}")
        model, key = target.rsplit(".", 1)
        if model not in selected:
            raise ValueError(
                f"bench --model-opt names model {model!r}, which was not selected with --model")
        if not key:
            raise ValueError(f"bench --model-opt key must not be empty: {pair!r}")
        if key in out.setdefault(model, {}):
            raise ValueError(
                f"bench --model-opt gives {model}.{key} more than once")
        _, value = _parse_model_opt(f"{key}={raw}")
        out[model][key] = value
    return out


def _add_sampling_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--num-frames", type=int, default=None, dest="num_frames")
    sp.add_argument(
        "--strategy",
        choices=["uniform", "uniform_floor", "all", "fps", "interval"],
        default=None,
    )
    sp.add_argument("--fps", type=float, default=None)
    sp.add_argument("--interval", type=int, default=None)
    sp.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    sp.add_argument("--include-last", dest="include_last",
                    action=argparse.BooleanOptionalAction, default=None)


def _sampling_from_args(args: argparse.Namespace, default_sampling):
    from ..core.types import SamplingSpec

    touched = any(getattr(args, a, None) is not None for a in _SAMPLING_ARG_NAMES)
    if not touched:
        return default_sampling
    if args.fps is not None and args.interval is not None:
        raise ValueError("--fps and --interval belong to different sampling strategies; give only one")
    inferred_strategy = (
        args.strategy
        or ("fps" if args.fps is not None else None)
        or ("interval" if args.interval is not None else None)
        or (default_sampling.strategy if default_sampling is not None else "uniform")
    )
    if args.fps is not None and inferred_strategy != "fps":
        raise ValueError("--fps can only be used with --strategy fps")
    if args.interval is not None and inferred_strategy != "interval":
        raise ValueError("--interval can only be used with --strategy interval")

    base = default_sampling or SamplingSpec()
    kw = {"strategy": inferred_strategy}
    if inferred_strategy != base.strategy:
        # SamplingSpec.key() records every field; switching strategy clears the old
        # strategy's parameters, avoiding constructible but ambiguous objects such as uniform + fps.
        kw["fps"] = None
        kw["interval"] = None
    if args.num_frames is not None:
        kw["num_frames"] = args.num_frames
    if args.fps is not None:
        kw["fps"] = args.fps
    if args.interval is not None:
        kw["interval"] = args.interval
    if args.resize is not None:
        kw["resize"] = (int(args.resize[0]), int(args.resize[1]))
    if args.include_last is not None:
        kw["include_last"] = args.include_last
    try:
        return replace(base, **kw)
    except TypeError:
        return SamplingSpec(**{**base.__dict__, **kw})


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------- list
def _cmd_list(args: argparse.Namespace) -> int:
    import rewardbox

    names = rewardbox.list_models(mode=args.mode, signal=args.signal,
                                  filter=args.filter or "",
                                  available_only=args.available_only)
    rows = []
    for n in names:
        info = rewardbox.get_model_info(n)
        v = info["verified"]
        verified = ("score+compare" if v["score"] and v["compare"]
                    else "score" if v["score"] else "-")
        rows.append({"name": n, "reward_mode": info["reward_mode"],
                     "signal_type": info["signal_type"],
                     "implementation_kind": info["implementation_kind"],
                     "verified": verified, "available": info["available"]})

    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK

    if not rows:
        print("(no matching models)")
        return EXIT_OK
    w_name = max(len(r["name"]) for r in rows)
    w_mode = max(len(r["reward_mode"]) for r in rows)
    w_sig = max(len(r["signal_type"]) for r in rows)
    w_kind = max(len(r["implementation_kind"]) for r in rows)
    for r in rows:
        print(f"{r['name']:<{w_name}}  {r['reward_mode']:<{w_mode}}  "
              f"{r['signal_type']:<{w_sig}}  {r['implementation_kind']:<{w_kind}}  "
              f"verified={r['verified']:<13}  available={'yes' if r['available'] else 'no'}")
    return EXIT_OK


# ---------------------------------------------------------------- info
def _cmd_info(args: argparse.Namespace) -> int:
    import rewardbox

    try:
        info = rewardbox.get_model_info(args.model)
    except KeyError as e:
        _err(str(e.args[0] if e.args else e))
        return EXIT_USAGE_ERROR

    if args.json:
        print(json.dumps(info, indent=2))
        return EXIT_OK

    print(f"name: {info['name']}")
    print(f"implementation_kind: {info['implementation_kind']}")
    print(f"reward_mode: {info['reward_mode']}   signal_type: {info['signal_type']}")
    print(f"weights: {info['weights']}")
    print(f"upstream: {info['upstream']}")
    print(f"paper: {info['paper']}")
    print(f"sampling (default): {info['sampling']}")
    print(f"score_range: {info['score_range']}   higher_better: {info['higher_better']}")
    print(f"supports_pairwise: {info['supports_pairwise']}   "
          f"outputs_success: {info['outputs_success']}")
    v = info["verified"]
    print(f"verified: score={v['score']}  compare={v['compare']}")
    print(f"available: {info['available']}")
    if info["missing"]:
        print(f"missing: {', '.join(info['missing'])}")
    if info["vram_note"]:
        print(info["vram_note"])
    if info["requires"].get("note"):
        print(f"compatibility_note: {info['requires']['note']}")
    if info["description"]:
        print(f"\n{info['description']}")
    return EXIT_OK


# ---------------------------------------------------------------- doctor
def _runtime_version(dist_key: str) -> str | None:
    """Match how the calibration record environment block was filled: each package's
    ``__version__`` attribute, not importlib.metadata.  The two give different strings for
    CUDA wheels such as torch ("2.8.0" vs "2.8.0+cu128"); comparing calibration records
    against importlib.metadata would systematically report "not verified"."""
    try:
        if dist_key == "python":
            import platform
            return platform.python_version()
        if dist_key == "torch":
            import torch
            return torch.__version__
        if dist_key == "transformers":
            import transformers
            return transformers.__version__
    except ImportError:
        return None
    return None


def _version_tri_state(model_name: str, dist_key: str, installed: str | None) -> str:
    """verified point | in range, unverified | not installed.  Only the dimensions recorded in
    the calibration record environment block (python/torch/transformers) are compared; other
    packages have no "verified point" and only report installation state."""
    if installed is None:
        return "not installed"
    from ..registry import _CALIBRATION_RECORDS
    record_path = _CALIBRATION_RECORDS / f"{model_name}.json"
    if not record_path.exists():
        return "in range, unverified (no calibration record for this model)"
    try:
        env = json.loads(record_path.read_text()).get("environment", {})
    except Exception:
        return "in range, unverified"
    if env.get(dist_key) == _runtime_version(dist_key):
        return "verified point"
    return "in range, unverified"


def _cmd_doctor(args: argparse.Namespace) -> int:
    import rewardbox

    if args.model:
        try:
            rewardbox.get_spec(args.model)
        except KeyError as e:
            _err(str(e.args[0] if e.args else e))
            return EXIT_USAGE_ERROR
        names = [args.model]
    else:
        names = rewardbox.list_models()
    all_ok = True
    for name in names:
        spec = rewardbox.get_spec(name)
        rep = spec.requires.check()
        print(f"\n=== {name} ({spec.implementation_kind.value}) ===")
        for item in rep.items:
            status = "OK" if item.ok else f"FAIL ({item.reason})"
            tri = (_version_tri_state(name, item.dist, item.installed)
                   if item.dist in ("python", "torch", "transformers") else "")
            print(f"  {item.dist:<16} declared={item.spec:<20} installed={item.installed or 'not installed':<20} "
                  f"{status}" + (f"  [{tri}]" if tri else ""))
        for k in rep.missing_env:
            print(f"  environment variable {k} is not set -- FAIL")
            all_ok = False
        if rep.vram_note:
            print(f"  {rep.vram_note}")
        if spec.requires.note:
            print(f"  compatibility note: {spec.requires.note}")
        if not rep.ok:
            all_ok = False
            if rep.install_hint:
                print("  Fix:")
                for line in rep.install_hint:
                    print(f"    {line}")
            else:
                print(f"  Fix: uv pip install --override locks/shared-overrides.txt "
                      f"-r locks/shared-py311-linux-x86_64.lock")

    from ..upstream import (
        UpstreamNotFoundError,
        ensure_importable,
        find_upstream,
        upstreams_for_model,
    )

    def _importable(key: str) -> bool:
        try:
            ensure_importable(key)
        except UpstreamNotFoundError:
            return False
        return importlib.util.find_spec(key) is not None

    # Only the topreward-* native variants use the TOPReward package; topreward itself runs the Robometer baseline.
    selected_topreward = any(name.startswith("topreward-") for name in names)
    if selected_topreward:
        if not _importable("topreward"):
            print("\nTOPReward official package: not found -- FAIL (`rewardbox fetch-upstream topreward`, "
                  "or add TOPReward to PYTHONPATH / install the official package)")
            all_ok = False
        else:
            print("\nTOPReward official package: found")
    selected_gvl = any(name == "gvl" or name.startswith("gvl-") for name in names)
    selected_robometer = any(name.startswith("robometer") or name == "topreward"
                             for name in names)
    if selected_robometer:
        if not _importable("robometer"):
            print("\nRobometer official package: not found -- FAIL (`rewardbox fetch-upstream robometer`, "
                  "or set REWARDBOX_ROBOMETER_ROOT / install the official package)")
            all_ok = False
        else:
            print("\nRobometer official package: found")
    if any(name.startswith("robodopamine-") for name in names):
        try:
            from ..adapters.robodopamine import _repo_root
            root = _repo_root()
            print(f"\nRobo-Dopamine official repository: {root}")
        except Exception as e:
            print(f"\nRobo-Dopamine official repository: not found -- FAIL ({e})")
            all_ok = False
    # vlac-cut* load the separate VLAC-cut checkout (reported from the pinned table below).
    selected_vlac = any(name in ("vlac-2b", "vlac-8b") for name in names)
    if selected_vlac:
        try:
            from ..adapters.vlac import _repo_root
            root = _repo_root()
            print(f"\nVLAC official repository: {root}")
        except Exception as e:
            print(f"\nVLAC official repository: not found -- FAIL ({e})")
            all_ok = False
    selected_sole_r1 = "sole-r1-8b" in names
    if selected_sole_r1:
        try:
            from ..adapters.sole_r1 import _repo_root
            root = _repo_root()
            print(f"\nSOLE-R1 official repository: {root}")
            print("SOLE-R1 service: doctor does not connect; score() uses "
                  "REWARDBOX_SOLE_R1_HOST or --model-opt host=...")
        except Exception as e:
            print(f"\nSOLE-R1 official repository: not found -- FAIL ({e})")
            all_ok = False
    selected_lrm = any(name.startswith("lrm-") for name in names)
    if selected_lrm:
        try:
            from ..adapters.lrm import _repo_root
            root = _repo_root()
            print(f"\nLarge-Reward-Models official repository: {root}")
            print("LRM HTTP service: doctor does not connect; score() uses "
                  "REWARDBOX_LRM_SERVER_URL or --model-opt server_url=...")
        except Exception as e:
            print(f"\nLarge-Reward-Models official repository: not found -- FAIL ({e})")
            all_ok = False
    selected_rynnvalue = any(name.startswith("rynnvalue-") for name in names)
    if selected_rynnvalue:
        try:
            from ..adapters.rynnvalue import _repo_root
            root = _repo_root()
            print(f"\nRynnValue official repository: {root}")
            print("RynnValue HTTP service: doctor does not connect; score() requires "
                  "server_provenance_file by default and uses REWARDBOX_RYNNVALUE_SERVER_URL "
                  "or --model-opt server_url=...")
        except Exception as e:
            print(f"\nRynnValue official repository: not found -- FAIL ({e})")
            all_ok = False
    # Checkouts without a family block above come straight from the pinned table.  The
    # ProcVLM / Robo-ValueRL clients only speak HTTP, so their checkout is needed only on
    # the machine that runs the server launcher and its absence here is not a failure.
    reported = {"topreward", "robometer", "robodopamine", "vlac", "sole-r1", "lrm", "rynnvalue"}
    server_side_only = {"procvlm", "robovaluerl"}
    for u in dict.fromkeys(u for name in names for u in upstreams_for_model(name)):
        if u.key in reported:
            continue
        root = find_upstream(u.key)
        override = f", or set {u.env}" if u.env else ""
        if root is not None:
            print(f"\n{u.dirname} official repository: {root}")
        elif u.key in server_side_only:
            print(f"\n{u.dirname} official repository: not found here (needed only where the "
                  f"model server is launched: `rewardbox fetch-upstream {u.key}`{override})")
        else:
            print(f"\n{u.dirname} official repository: not found -- FAIL "
                  f"(`rewardbox fetch-upstream {u.key}`{override})")
            all_ok = False
    if selected_gvl:
        from ..adapters.gvl import credential_env_for_profile

        try:
            from ..adapters.gvl import _repo_root
            root = _repo_root()
            print(f"\nGVL pinned Robometer repository: {root}")
        except Exception as e:
            print(f"\nGVL pinned Robometer repository: not found -- FAIL ({e})")
            all_ok = False
        credential_envs = {
            credential_env_for_profile(name)
            for name in names if name == "gvl" or name.startswith("gvl-")
        }
        for env_name in sorted(credential_envs):
            label = "GVL Gemini"
            if os.environ.get(env_name):
                print(f"{label} credential: {env_name} is set (value not shown)")
            else:
                print(f"{label} credential: {env_name} is not set (the Python constructor may pass api_key "
                      "explicitly; CLI/bench accept only environment variables to avoid shell/argv leaks; "
                      "available only means the dependencies are present, not that an API request can be made)")

    if any(name.startswith(("vlm-", "jev-")) for name in names):
        print("\nGeneral-VLM / Jev profiles: hosted API, no checkout. The key is read at request "
              "time from OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY (vlm-*, by api) or "
              "JEV_API_KEY (jev-*, which also need base_url), or from the variable named by api_key_env; doctor "
              "does not contact the endpoint (docs/guides/general_vlm_and_jev.md)")

    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _v
        try:
            if selected_topreward:
                print(f"\nflash-attn: {_v('flash-attn')} (installed, needed by TOPReward)")
        except PackageNotFoundError:
            if selected_topreward:
                print("\nflash-attn: not installed -- FAIL. TOPReward's QwenClient hard-codes "
                      "attn_implementation='flash_attention_2' and raises ImportError without it. "
                      "Fix: see the prebuilt wheel command in docs/installation.md.")
                all_ok = False
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nGPU: available, {torch.cuda.device_count()} visible"
                  f" (subject to CUDA_VISIBLE_DEVICES)")
        else:
            print("\nGPU: torch is installed but CUDA is not available")
            if any(rewardbox.get_spec(name).requires.vram_gb for name in names):
                all_ok = False
    except ImportError:
        print("\nGPU: torch is not installed, cannot probe (list/info still work, but no model can run)")
        if any(rewardbox.get_spec(name).requires.vram_gb for name in names):
            all_ok = False

    if args.deep:
        print("\n--deep: lightweight import of official entry points (has side effects, see the notes below)")
        modules = []
        if selected_topreward:
            modules.append(("topreward", ""))
        if selected_robometer:
            modules.append(
                ("robometer.utils.setup_utils",
                 "triggers unsloth's global monkeypatch of torch/transformers"))
        if selected_gvl:
            try:
                from ..adapters.gvl import _upstream_gvl_class
                _upstream_gvl_class()
                print("  robometer.evals.baselines.gvl: import ok (pinned checkout)")
            except Exception as e:
                print(f"  robometer.evals.baselines.gvl: import failed -- {e}")
                all_ok = False
        if selected_sole_r1:
            try:
                from ..adapters.sole_r1 import _upstream_client_class
                _upstream_client_class()
                print("  sole-r1 reward_client.utils.InferenceClient: import ok"
                      " (service not contacted)")
            except Exception as e:
                print("  sole-r1 reward_client.utils.InferenceClient: "
                      f"import failed -- {e}")
                all_ok = False
        if selected_lrm:
            try:
                from ..adapters.lrm import _upstream_client_class
                _upstream_client_class()
                print("  Large-Reward-Models VLMRewardClient: import ok (service not contacted)")
            except Exception as e:
                print(f"  Large-Reward-Models VLMRewardClient: import failed -- {e}")
                all_ok = False
        if selected_vlac:
            try:
                from ..adapters.vlac import _repo_root
                root = _repo_root()
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
            except Exception:
                pass
            modules.append(
                ("evo_vlac",
                 "loads ms-swift/torch; run it in an isolated worker"))
        for mod_name, side_effect in modules:
            try:
                imported = importlib.import_module(mod_name)
                if mod_name == "evo_vlac":
                    import inspect
                    critic_params = inspect.signature(
                        imported.GAC_model.get_trajectory_critic).parameters
                    required = {
                        "task", "image_list", "ref_image_list", "batch_num",
                        "skip", "rich", "reverse_eval", "frame_skip",
                        "value_simple",
                    }
                    missing = required - set(critic_params)
                    if missing:
                        raise RuntimeError(
                            "GAC_model.get_trajectory_critic lacks arguments the adapter needs: "
                            f"{sorted(missing)}")
                    init_params = inspect.signature(
                        imported.GAC_model.init_model).parameters
                    missing_init = {
                        "model_path", "model_type", "device_map",
                    } - set(init_params)
                    if missing_init:
                        raise RuntimeError(
                            "GAC_model.init_model lacks arguments the adapter needs: "
                            f"{sorted(missing_init)}")
                note = f" (note: {side_effect})" if side_effect else ""
                print(f"  {mod_name}: import ok{note}")
            except Exception as e:
                print(f"  {mod_name}: import failed -- {e}")
                all_ok = False

    print()
    return EXIT_OK if all_ok else EXIT_RUNTIME_ERROR


# ---------------------------------------------------------------- run
def _cmd_run(args: argparse.Namespace) -> int:
    import rewardbox
    from ..core.results import ResultSet
    from ..core.types import Trajectory

    try:
        model_opts = _parse_model_opts(args.model_opt)
        _reject_sensitive_cli_opts(args.model, model_opts)
    except ValueError as e:
        _err(f"error: {e}")
        return EXIT_USAGE_ERROR

    try:
        spec = rewardbox.get_spec(args.model)
    except KeyError as e:
        _err(str(e.args[0] if e.args else e))
        return EXIT_USAGE_ERROR
    try:
        from ..registry import validate_model_opts
        validate_model_opts(args.model, model_opts)
    except TypeError as e:
        _err(f"error: {e}")
        return EXIT_USAGE_ERROR

    try:
        sampling = _sampling_from_args(args, spec.sampling)
    except ValueError as e:
        _err(f"error: {e}")
        return EXIT_USAGE_ERROR

    try:
        traj = Trajectory.from_video(args.video, args.instruction, sampling)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        _err(f"error: cannot read video {args.video!r}: {e}")
        return EXIT_USAGE_ERROR

    rs = ResultSet.from_trajectory(traj)

    rm = None
    runtime_error = None
    try:
        rm = rewardbox.create_reward(args.model, **model_opts)
        out = rm.score(traj)
        rs.add(out)
    except Exception as e:
        rs.add_error(args.model, str(e))
        _err(f"run failed: {args.model}: {e}")
        runtime_error = e
    finally:
        if rm is not None:
            try:
                rm.close()
            except Exception as e:
                _err(f"failed to release model resources: {args.model}: {e}")
                runtime_error = runtime_error or e

    if args.output:
        try:
            rs.save(
                args.output, merge=not args.no_merge,
                overwrite_models=args.overwrite_models)
        except Exception as e:
            _err(f"save failed: {args.output}: {e}")
            return EXIT_RUNTIME_ERROR
        print(f"saved to {args.output}")
    elif runtime_error is None:
        print(json.dumps(rs.to_dict(), indent=2))
    return EXIT_RUNTIME_ERROR if runtime_error is not None else EXIT_OK


# ---------------------------------------------------------------- bench
def _cmd_bench(args: argparse.Namespace) -> int:
    from ..bench import aggregate, run_benchmark
    from ..data import Manifest

    try:
        manifest = Manifest.load(args.manifest)
    except (FileNotFoundError, ValueError) as e:
        _err(f"error: {e}")
        return EXIT_USAGE_ERROR

    protocol_args = ("num_frames", "strategy", "fps", "interval")
    if not any(getattr(args, name) is not None for name in protocol_args):
        _err("error: bench needs an explicit sampling protocol; --resize/--include-last alone "
             "cannot define the shared frame grid. uniform needs at least --num-frames")
        return EXIT_USAGE_ERROR
    try:
        sampling = _sampling_from_args(args, None)
    except ValueError as e:
        _err(f"error: {e}")
        return EXIT_USAGE_ERROR
    if sampling is None:
        _err("error: bench needs explicit sampling parameters (at least --num-frames): models "
             "have no common default SamplingSpec, and mixing their own defaults makes results incomparable")
        return EXIT_USAGE_ERROR
    if sampling.strategy == "uniform" and args.num_frames is None:
        _err("error: uniform sampling in bench needs an explicit --num-frames")
        return EXIT_USAGE_ERROR

    for name in args.model:
        try:
            import rewardbox
            rewardbox.get_spec(name)
        except KeyError as e:
            _err(str(e.args[0] if e.args else e))
            return EXIT_USAGE_ERROR

    try:
        worker_pythons = _parse_worker_pythons(args.worker_python, args.model)
        model_opts = _parse_bench_model_opts(args.model_opt, args.model)
        for model, opts in model_opts.items():
            _reject_sensitive_cli_opts(model, opts)
        from ..registry import validate_model_opts
        for model, opts in model_opts.items():
            validate_model_opts(model, opts)
    except ValueError as e:
        _err(f"error: {e}")
        return EXIT_USAGE_ERROR
    except TypeError as e:
        _err(f"error: {e}")
        return EXIT_USAGE_ERROR

    try:
        shards = run_benchmark(
            args.model, args.manifest, sampling, args.output,
            isolate=args.isolate, resume=not args.no_resume,
            retry_errors=args.retry_errors,
            model_opts=model_opts,
            worker_pythons=worker_pythons,
            log=lambda m: print(m, flush=True))
        result_sets = aggregate(shards, args.manifest, sampling)
    except (ValueError, FileNotFoundError) as e:
        _err(f"bench configuration/resume failed: {e}")
        return EXIT_USAGE_ERROR
    except Exception as e:
        _err(f"bench run failed: {e}")
        return EXIT_RUNTIME_ERROR
    out_dir = Path(args.output)
    for traj_id, rs in result_sets.items():
        # merge=False: the shard files (with their own resume logic) are the source of truth;
        # this is a complete view regenerated from the shards every time, not an incremental
        # merge.  merge=True would be blocked on re-runs/--resume by ResultSet's own "refuse to
        # overwrite same-name models" rule, which protects data written elsewhere and does not
        # apply to regenerating a derived view.
        try:
            rs.save(out_dir / f"trajectory_{traj_id}.json", merge=False)
        except Exception as e:
            _err(f"failed to save the bench aggregate ({traj_id}): {e}")
            return EXIT_RUNTIME_ERROR

    n_ok = sum(len(s.results) for s in shards.values())
    n_err = sum(len(s.errors) for s in shards.values())
    print(f"bench finished: {len(args.model)} models x {len(manifest)} trajectories "
          f"({n_ok} ok, {n_err} failed), results in {args.output}")
    return EXIT_RUNTIME_ERROR if n_err or any(s.fatal_error for s in shards.values()) else EXIT_OK


# ---------------------------------------------------------------- fetch-upstream
def _cmd_fetch_upstream(args: argparse.Namespace) -> int:
    """Clone the pinned official checkouts that adapters call into."""
    import rewardbox

    from .. import upstream as up

    if args.list or not args.targets:
        rows = []
        for u in up.UPSTREAMS:
            found = up.find_upstream(u.key)
            rows.append({"key": u.key, "dirname": u.dirname, "url": u.url, "commit": u.commit,
                         "models": list(u.models), "env": u.env,
                         "found": str(found) if found else None})
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for r in rows:
                state = r["found"] or "missing"
                print(f"{r['key']:<13} {r['commit'][:12]}  {r['url']}\n"
                      f"{'':<13} models: {', '.join(r['models'])}   -> {state}")
            print(f"\ndefault destination: {up.default_upstream_dir()}  "
                  f"(override with --dest or {up.UPSTREAM_DIR_ENV})")
        return EXIT_OK

    keys: list[str] = []
    for target in args.targets:
        if target == "all":
            keys.extend(u.key for u in up.UPSTREAMS)
            continue
        # A name can be both an upstream key and a model (``topreward``: the key is
        # the TOPReward repo, the model runs Robometer's baseline); fetch both.
        try:
            keys.append(up.get_upstream(target).key)
            is_key = True
        except KeyError:
            is_key = False
        try:
            rewardbox.get_spec(target)
        except KeyError as e:
            if is_key:
                continue
            _err(f"error: {target!r} is neither an upstream key nor a model. "
                 f"{e.args[0] if e.args else e}")
            return EXIT_USAGE_ERROR
        needed = up.upstreams_for_model(target)
        if not needed and not is_key:
            print(f"{target}: no source checkout needed "
                  "(weights come from the Hugging Face Hub or an API)")
        keys.extend(u.key for u in needed)
    failed = False
    for key in dict.fromkeys(keys):
        try:
            up.fetch(key, Path(args.dest) if args.dest else None, dry_run=args.dry_run)
        except Exception as e:  # keep going; report every failure
            _err(f"fetch-upstream {key} failed: {e}")
            failed = True
    if args.dest and not args.dry_run and not failed:
        print(f"\nexport {up.UPSTREAM_DIR_ENV}={Path(args.dest).expanduser().resolve()}")
    return EXIT_RUNTIME_ERROR if failed else EXIT_OK


# ---------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    from .. import __version__

    p = argparse.ArgumentParser(
        prog="rewardbox",
        description="Unified adapters and evaluation for robot reward models "
                    "(list / info / doctor / run / bench / fetch-upstream)")
    p.add_argument("--version", action="version", version=f"rewardbox {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available models")
    p_list.add_argument("--mode", default=None)
    p_list.add_argument("--signal", default=None)
    p_list.add_argument("--filter", default=None)
    p_list.add_argument("--available-only", action="store_true")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_list)

    p_info = sub.add_parser("info", help="show details of one model")
    p_info.add_argument("model")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=_cmd_info)

    p_doctor = sub.add_parser("doctor", help="check dependencies / environment / GPU state")
    p_doctor.add_argument("model", nargs="?", default=None)
    p_doctor.add_argument("--deep", action="store_true",
                          help="also import official entry points lightly (has side effects, see the output)")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_run = sub.add_parser("run", help="run one model on one video")
    p_run.add_argument("model")
    p_run.add_argument("video")
    p_run.add_argument("--instruction", required=True)
    _add_sampling_args(p_run)
    p_run.add_argument("--model-opt", action="append", default=[], dest="model_opt",
                       metavar="KEY=JSON_VALUE",
                       help="argument passed to create_reward(); repeatable, e.g. num_samples=8")
    p_run.add_argument("-o", "--output", default=None)
    p_run.add_argument("--no-merge", action="store_true")
    p_run.add_argument(
        "--overwrite-models", action="store_true",
        help="when merging output, explicitly allow overwriting same-name model results")
    p_run.set_defaults(func=_cmd_run)

    p_bench = sub.add_parser(
        "bench", help="batch evaluation of several models x a manifest dataset (models in the outer loop)")
    p_bench.add_argument("--manifest", required=True)
    p_bench.add_argument("--model", action="append", required=True, dest="model")
    _add_sampling_args(p_bench)
    p_bench.add_argument("-o", "--output", required=True, help="shard/result output directory")
    p_bench.add_argument("--isolate", action="store_true",
                         help="run each model in its own subprocess (isolates global side effects such as unsloth)")
    p_bench.add_argument(
        "--worker-python", action="append", default=[],
        metavar="MODEL=PYTHON_PATH",
        help="separate Python environment for a model; repeatable, implies subprocess isolation for that model",
    )
    p_bench.add_argument(
        "--model-opt", action="append", default=[],
        metavar="MODEL.KEY=JSON_VALUE",
        help="create_reward() argument for one model; repeatable, e.g. vlac-2b.batch_size=2",
    )
    p_bench.add_argument("--no-resume", action="store_true",
                         help="ignore existing shards and re-run everything (by default --resume skips finished manifest entries)")
    p_bench.add_argument(
        "--retry-errors", action="store_true",
        help="on resume, retry entries recorded as failed in the shard (successes are still skipped)")
    p_bench.set_defaults(func=_cmd_bench)

    p_fetch = sub.add_parser(
        "fetch-upstream",
        help="clone the pinned official source an adapter calls into (git, blobless)")
    p_fetch.add_argument("targets", nargs="*",
                         help="model names (robometer-4b), upstream keys (robometer) or 'all'")
    p_fetch.add_argument("--dest", default=None,
                         help="parent directory for checkouts (default: REWARDBOX_UPSTREAM_DIR, "
                              "<source clone>/reward_model_code, or ~/.cache/rewardbox/upstream)")
    p_fetch.add_argument("--list", action="store_true", help="show the pinned table and what is found")
    p_fetch.add_argument("--json", action="store_true")
    p_fetch.add_argument("--dry-run", action="store_true", help="print git commands only")
    p_fetch.set_defaults(func=_cmd_fetch_upstream)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
