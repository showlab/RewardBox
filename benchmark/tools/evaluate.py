#!/usr/bin/env python3
"""Official RRM-Bench-1K evaluator.

    python tools/evaluate.py --predictions preds.jsonl --name my-model
    python tools/evaluate.py --predictions runs/exp1/robometer-4b.json --mode interpolate
    python tools/evaluate.py --predictions preds.jsonl --cohort multiview

Predictions (one of):
  * JSONL, one row per case:
      {"id": CASE_ID, "frame_indices": [...], "progress": [...]}   # a curve on source frames
      {"id": CASE_ID, "directions": [d_1, ..., d_{G-1}]}           # one judgement per 1 FPS
                                                                  # transition, -1/0/+1 or null
  * a RewardBox ``rewardbox bench`` shard (``<output>/<model>.json``). Results of Pair-protocol
    models (``vlm-pair``, ``jev-1.14-pair``) are read as ``directions``, the others as curves.

Metrics (definitions match the paper; see README.md):
  direction  Each case is scored on its 1 FPS grid (inputs/eval_grid_1fps.json). The GT direction
             of a transition is the sign of the label curve difference (|d| <= 1e-9 is flat);
             flat transitions are excluded. A curve's direction is the sign of its value
             difference on the grid; a tie counts as wrong. All transitions are pooled; report
             Rising/Falling precision, recall, F1, Macro-F1 (mean of the two F1) and accuracy.
             Transitions without a prediction are excluded from P/R/F1 and counted as missing.
             ``--mode exact`` uses only grid frames that carry a native value; ``--mode
             interpolate`` linearly interpolates inside the native range (no extrapolation) and
             is flagged as interpolated.
  by suite   The same direction metrics per category. (A) Action-Effect is scored on the views
             that show the effect (``action_effect_scored``); the other suites use every member
             video. Each cell comes with the always-rising Macro-F1 on the same transitions.
             (D) Visibility also pairs each P/X video with its V counterpart of the same clip.
  ranking    On the native points as given (no interpolation): VOC = mean over successful
             cases of Pearson r(time, value) (undefined -> 0); Kendall tau_a = class-balanced
             failure < suboptimal < successful ranking of the last value within each task, then
             task mean (ties 0); Succ-Fail Diff = mean successful minus mean failure last value
             within each task, then task mean. Outcome classes are GT endpoints 0 / (0,1) / 1.
             Two scopes: all single views (1,117) and one main camera per main clip (201).

Only numpy and scipy are required. Inputs are trusted fixed files; nothing is hashed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

EPSILON = 1e-9
QUALITY = ("failure", "suboptimal", "successful")       # ordinal order for Kendall
CATEGORIES = ("action_effect", "semantic_ordered", "persistence_reversal",
              "occlusion_viewpoint", "control", "no_task_view")
A_CATEGORY = "action_effect"
D_CATEGORY = "occlusion_viewpoint"
CONTROL_GROUPS = {"clean_success_control": "control_clean_success",
                  "obvious_failure_control": "control_obvious_failure"}
D_PAIRED = ("occlusion_viewpoint_paired", "paired_result_view")
BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED = 2000, 20261018
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------- loading
def load_benchmark(root: Path = ROOT, cohort: str = "single_view") -> dict:
    """Ground truth, 1 FPS grid and per-case GT transitions for one cohort."""
    gt_file = {"single_view": "ground_truth.json", "multiview": "multiview_ground_truth.json"}[cohort]
    labels = json.loads((root / "inputs" / gt_file).read_text(encoding="utf-8"))
    grid = json.loads((root / "inputs" / "eval_grid_1fps.json").read_text(encoding="utf-8"))["frames"]
    cases = {}
    for case_id, label in labels.items():
        frames = np.asarray(grid[case_id], dtype=np.int64)
        times = frames / float(label["fps"])
        curve = np.interp(times, [p["time"] for p in label["curve"]],
                          [p["value"] for p in label["curve"]])
        cases[case_id] = {**label, "grid": frames, "gt_directions": sign(np.diff(curve))}
    main_camera = set()
    if cohort == "single_view":
        by_clip = defaultdict(list)
        for case_id, label in labels.items():
            if label["condition"] == "main":
                by_clip[label["clip_id"]].append(case_id)
        for clip_id, members in sorted(by_clip.items()):
            chosen = [c for c in members if labels[c]["main_camera"]]
            if len(chosen) != 1:
                raise ValueError(f"main camera is not unique for {clip_id}")
            main_camera.add(chosen[0])
    return {"root": root, "cohort": cohort, "cases": cases, "main_camera": main_camera}


def load_predictions(path: Path) -> tuple[str, dict]:
    """Return (default display name, {case_id: {"frame_indices","progress"} | {"directions"}})."""
    path = Path(path)
    if path.suffix == ".json":
        shard = json.loads(path.read_text(encoding="utf-8"))
        if "results" not in shard:
            raise ValueError(f"{path} is not a RewardBox bench shard (no 'results')")
        preds = {}
        for cid, r in shard["results"].items():
            if (r.get("meta") or {}).get("output_kind") == "transition_directions":
                # Pair-protocol models (vlm-pair, jev-1.14-pair) judge transitions directly;
                # their per-transition answers are in raw, with null for a missing one.
                preds[cid] = {"directions": r["raw"]["transition_directions"],
                              "frame_indices": r["frame_indices"]}
            else:
                preds[cid] = {"frame_indices": r["frame_indices"], "progress": r["progress"]}
        return shard.get("model") or path.stem, preds
    preds = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row.get("id") or row.get("case_id")
        if case_id in preds:
            raise ValueError(f"{path}:{n}: duplicate id {case_id!r}")
        if "directions" in row:
            preds[case_id] = {"directions": row["directions"]}
        else:
            preds[case_id] = {"frame_indices": row["frame_indices"], "progress": row["progress"]}
    return path.stem, preds


# ---------------------------------------------------------------------------- primitives
def sign(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values) <= EPSILON, 0, np.sign(values)).astype(int)


def validate_curve(case_id: str, pred: dict, case: dict) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(pred["frame_indices"])
    values = np.asarray(pred["progress"], dtype=float)
    if (frames.ndim != 1 or values.ndim != 1 or len(frames) == 0 or len(frames) != len(values)
            or not np.issubdtype(frames.dtype, np.integer) or not np.isfinite(values).all()
            or np.any(np.diff(frames) <= 0) or frames[0] < 0 or frames[-1] >= case["frames"]):
        raise ValueError(f"invalid curve for {case_id}: integer, strictly increasing frames "
                         f"in [0, {case['frames']}) and finite values required")
    return frames.astype(np.int64), values


def curve_directions(frames, values, grid, mode: str) -> np.ndarray:
    """Per-transition predicted direction on the grid; NaN marks a missing transition."""
    if mode == "exact":
        lookup = dict(zip(frames.tolist(), values.tolist(), strict=True))
        sampled = np.array([lookup.get(int(f), np.nan) for f in grid], dtype=float)
    elif mode == "interpolate":
        sampled = np.interp(grid, frames, values, left=np.nan, right=np.nan)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    present = np.isfinite(sampled[:-1]) & np.isfinite(sampled[1:])
    out = np.full(max(0, len(grid) - 1), np.nan)
    out[present] = sign(np.diff(sampled)[present])
    return out


def metrics(points) -> dict:
    """points: (gt, prediction) with gt in {-1, +1}, prediction in {-1, 0, +1}."""
    if not points:
        return {"n": 0}
    y, p = (np.array(column) for column in zip(*points, strict=True))
    out = {"n": len(y), "n_rising": int(np.sum(y == 1)), "n_falling": int(np.sum(y == -1)),
           "pred_rising": int(np.sum(p == 1)), "pred_falling": int(np.sum(p == -1)),
           "pred_tie": int(np.sum(p == 0)), "accuracy": float(np.mean(y == p))}
    for k, name in ((1, "rising"), (-1, "falling")):
        tp = int(np.sum((y == k) & (p == k)))
        fp = int(np.sum((y != k) & (p == k)))
        fn = int(np.sum((y == k) & (p != k)))
        out[name] = {"precision": tp / (tp + fp) if tp + fp else None,
                     "recall": tp / (tp + fn) if tp + fn else None,
                     "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
                     "tp": tp, "fp": fp, "fn": fn}
    f1s = [out["rising"]["f1"], out["falling"]["f1"]]
    out["macro_f1"] = float(np.mean([f or 0.0 for f in f1s])) if all(f is not None for f in f1s) else None
    out["pred_falling_share"] = out["pred_falling"] / out["n"]
    out["pred_tie_share"] = out["pred_tie"] / out["n"]
    return out


def _confusion_scores(counts):
    scores = []
    for g, p in ((1, 2), (0, 0)):
        tp = counts[..., g, p]
        fp = counts[..., 1 - g, p]
        fn = counts[..., g, :].sum(-1) - tp
        scores.append(np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(tp.shape),
                                where=(2 * tp + fp + fn) > 0))
    accuracy = (counts[..., 1, 2] + counts[..., 0, 0]) / counts.sum((-1, -2))
    return (scores[0] + scores[1]) / 2, accuracy


def d_versus_paired_v(cases: dict, by_case: dict, repetitions=BOOTSTRAP_REPETITIONS,
                      seed=BOOTSTRAP_SEED) -> dict | None:
    """Each P/X video of the Visibility suite minus its paired V view on identical GT transitions,
    with a 95% bootstrap CI over resampled episodes (group_id)."""
    # Canonical (ground_truth.json) case order: bootstrap draws index episodes by position,
    # so the CI must not depend on how a predictions file happens to be sorted.
    groups = {}
    for case_id in (c for c in cases if c in by_case):
        points = by_case[case_id]
        case = cases[case_id]
        partner = case.get("result_case_id")
        if D_CATEGORY not in case["categories"] or partner in (None, case_id) or partner not in by_case:
            continue
        common = points.keys() & by_case[partner].keys()
        if not common:
            continue
        table = groups.setdefault(case["group_id"], np.zeros((2, 2, 3)))
        for j in common:
            for side, (gt, pred) in enumerate((points[j], by_case[partner][j])):
                table[side, (gt + 1) // 2, pred + 1] += 1
    if not groups:
        return None
    stacked = np.stack(list(groups.values()))
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(0, len(stacked), len(stacked)), minlength=len(stacked))
                        for _ in range(repetitions)])
    sampled = np.einsum("rg,gsyp->rsyp", weights, stacked)
    macro, accuracy = _confusion_scores(sampled)
    point_macro, point_accuracy = _confusion_scores(stacked.sum(0))
    out = {"episodes": len(stacked), "transitions": int(stacked[:, 0].sum()),
           "repetitions": repetitions, "seed": seed, "resampling_unit": "episode (group_id)"}
    for name, point, draws in (("macro_f1", point_macro, macro), ("accuracy", point_accuracy, accuracy)):
        delta = draws[:, 0] - draws[:, 1]
        out[name] = {"d": float(point[0]), "paired_v": float(point[1]),
                     "delta": float(point[0] - point[1]),
                     "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))]}
    return out


# ---------------------------------------------------------------------------- scoring
def _with_always_rising(points) -> dict:
    """Metrics of one group plus the always-rising Macro-F1 on the same transitions."""
    result = metrics(points)
    if result.get("n"):
        result["always_rising_macro_f1"] = metrics([(g, 1) for g, _ in points])["macro_f1"]
    return result


def score_direction(bench: dict, preds: dict, mode: str = "exact", bootstrap: bool = True) -> dict:
    cases = bench["cases"]
    unknown = sorted(set(preds) - set(cases))
    if unknown:
        raise ValueError(f"{len(unknown)} prediction ids are not in the {bench['cohort']} cohort, "
                         f"e.g. {unknown[:3]}")
    by_case, used_interpolation = {}, False
    for case_id, pred in preds.items():
        case = cases[case_id]
        if "directions" in pred:
            raw = pred["directions"]
            if "frame_indices" in pred and list(pred["frame_indices"]) != case["grid"].tolist():
                raise ValueError(f"{case_id}: transition judgements were not made on the 1 FPS grid; "
                                 "run the model with --strategy fps --fps 1 --no-include-last")
            if len(raw) != len(case["grid"]) - 1:
                raise ValueError(f"{case_id}: {len(raw)} directions for {len(case['grid']) - 1} transitions")
            directions = np.array([np.nan if d is None else d for d in raw], dtype=float)
            if not np.isin(directions[np.isfinite(directions)], (-1, 0, 1)).all():
                raise ValueError(f"{case_id}: directions must be -1, 0, +1 or null")
        else:
            frames, values = validate_curve(case_id, pred, case)
            directions = curve_directions(frames, values, case["grid"], mode)
            used_interpolation |= mode == "interpolate"
        by_case[case_id] = {j: (int(g), int(directions[j]))
                            for j, g in enumerate(case["gt_directions"])
                            if g and np.isfinite(directions[j])}
    points = [pt for entries in by_case.values() for pt in entries.values()]
    expected = sum(int(np.count_nonzero(c["gt_directions"])) for c in cases.values())
    groups = defaultdict(list)
    for case_id in (c for c in cases if c in by_case):
        entries = by_case[case_id]
        case = cases[case_id]
        for category in case["categories"]:
            if category == A_CATEGORY and not case.get("action_effect_scored", True):
                continue        # (A) is scored on the views that show the effect
            groups[category].extend(entries.values())
        control = CONTROL_GROUPS.get(case.get("human_suite"))
        if control:
            groups[control].extend(entries.values())
        partner = case.get("result_case_id")
        if D_CATEGORY in case["categories"] and partner not in (None, case_id) and partner in by_case:
            if not np.array_equal(case["gt_directions"], cases[partner]["gt_directions"]):
                raise ValueError(f"{case_id} and its result view differ in GT")
            common = sorted(entries.keys() & by_case[partner].keys())
            groups[D_PAIRED[0]].extend(entries[j] for j in common)
            groups[D_PAIRED[1]].extend(by_case[partner][j] for j in common)
    return {
        "mode": mode, "interpolated": used_interpolation,
        "coverage": {"cases": len(cases), "cases_with_predictions": len(preds),
                     "nonflat_transitions": expected, "scored": len(points),
                     "missing": expected - len(points)},
        "overall": metrics(points),
        "always_rising_same_support": metrics([(g, 1) for g, _ in points]),
        "by_category": {k: _with_always_rising(groups[k])
                        for k in (*CATEGORIES, *CONTROL_GROUPS.values(), *D_PAIRED) if k in groups},
        "d_versus_paired_v": d_versus_paired_v(cases, by_case) if bootstrap else None,
    }


def pearson_with_time(times, values) -> tuple[float, bool]:
    if len(times) < 2 or np.ptp(times) == 0 or np.ptp(values) == 0:
        return 0.0, True          # upstream maps an undefined per-trajectory Pearson to zero
    from scipy.stats import pearsonr
    return float(pearsonr(times, values).statistic), False


def ranking_aggregate(rows: list[dict]) -> dict:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    tasks = {}
    for task, members in sorted(by_task.items()):
        by_q = {q: np.asarray([r["last"] for r in members if r["quality"] == q]) for q in QUALITY}
        present = [q for q in QUALITY if len(by_q[q])]
        pair_scores = [float(np.sign(by_q[upper][:, None] - by_q[lower][None, :]).mean())
                       for lower, upper in combinations(present, 2)]
        tasks[task] = {
            "quality_counts": dict(Counter(r["quality"] for r in members)),
            "kendall_tau_a": float(np.mean(pair_scores)) if pair_scores else None,
            "succ_fail_diff": (float(by_q["successful"].mean() - by_q["failure"].mean())
                               if len(by_q["successful"]) and len(by_q["failure"]) else None)}
    successful = [r for r in rows if r["quality"] == "successful"]
    taus = [t["kendall_tau_a"] for t in tasks.values() if t["kendall_tau_a"] is not None]
    diffs = [t["succ_fail_diff"] for t in tasks.values() if t["succ_fail_diff"] is not None]
    return {"n_cases": len(rows),
            "quality_counts": dict(Counter(r["quality"] for r in rows)),
            "voc_pearson_r": float(np.mean([r["voc"] for r in successful])) if successful else None,
            "voc_n_successful": len(successful),
            "voc_undefined_mapped_to_zero": sum(r["voc_undefined"] for r in successful),
            "kendall_tau_a": float(np.mean(taus)) if taus else None, "kendall_n_tasks": len(taus),
            "succ_fail_diff": float(np.mean(diffs)) if diffs else None, "succ_fail_n_tasks": len(diffs),
            "tasks": tasks}


def score_ranking(bench: dict, preds: dict, value_scale: float = 1.0) -> dict | None:
    if bench["cohort"] != "single_view":
        return None
    curves = {cid: p for cid, p in preds.items() if "directions" not in p}
    if not curves:
        return None
    rows = []
    for case_id, pred in sorted(curves.items()):
        case = bench["cases"][case_id]
        frames, values = validate_curve(case_id, pred, case)
        r, undefined = pearson_with_time(frames / float(case["fps"]), values)
        rows.append({"case_id": case_id, "task": case["task"], "quality": case["quality"],
                     "voc": r, "voc_undefined": undefined, "last": float(values[-1])})
    out = {}
    for scope, members in (("all_single_views", set(bench["cases"])),
                           ("main_camera", bench["main_camera"])):
        chosen = [r for r in rows if r["case_id"] in members]
        result = ranking_aggregate(chosen)
        result.update(expected=len(members), missing=len(members) - len(chosen))
        result["succ_fail_diff_scaled"] = (None if result["succ_fail_diff"] is None
                                          else result["succ_fail_diff"] * value_scale)
        out[scope] = result
    out["value_scale"] = value_scale
    return out


def evaluate(predictions: dict, *, cohort: str = "single_view", mode: str = "exact",
             value_scale: float = 1.0, root: Path = ROOT, bootstrap: bool = True) -> dict:
    bench = load_benchmark(root, cohort)
    return {"benchmark": "RRM-Bench-1K", "cohort": cohort,
            "direction": score_direction(bench, predictions, mode, bootstrap),
            "ranking": score_ranking(bench, predictions, value_scale)}


# ---------------------------------------------------------------------------- reporting
def _fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def render(name: str, report: dict) -> str:
    d = report["direction"]
    o = d["overall"]
    tag = "†" if d["interpolated"] else ""
    lines = [f"## {name}{tag} ({report['cohort']}, mode={d['mode']})", "",
             f"Scored {d['coverage']['scored']}/{d['coverage']['nonflat_transitions']} non-flat 1 FPS "
             f"transitions (missing {d['coverage']['missing']}); "
             f"{d['coverage']['cases_with_predictions']}/{d['coverage']['cases']} cases with predictions.", "",
             "| Rising P | Rising R | Rising F1 | Falling P | Falling R | Falling F1 | Macro-F1 | Acc | pred falling | ties |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if o.get("n"):
        lines.append("| " + " | ".join(_fmt(o[c][k]) for c in ("rising", "falling")
                                         for k in ("precision", "recall", "f1"))
                     + f" | {_fmt(o['macro_f1'])} | {_fmt(o['accuracy'])} | "
                       f"{_fmt(o['pred_falling_share'])} | {_fmt(o['pred_tie_share'])} |")
    ar = d["always_rising_same_support"]
    if ar.get("n"):
        lines += ["", f"Always-rising on the same support: Macro-F1 {_fmt(ar['macro_f1'])}."]
    lines += ["", "| category | n | Macro-F1 | Falling F1 | always-rising Macro-F1 |",
              "|---|---:|---:|---:|---:|"]
    for k, v in d["by_category"].items():
        lines.append(f"| {k} | {v.get('n', 0)} | {_fmt(v.get('macro_f1'))} | "
                     f"{_fmt((v.get('falling') or {}).get('f1'))} | "
                     f"{_fmt(v.get('always_rising_macro_f1'))} |")
    p = d.get("d_versus_paired_v")
    if p:
        m = p["macro_f1"]
        lines += ["", f"Visibility: P/X views vs their paired V view ({p['episodes']} episodes, "
                      f"{p['transitions']} transitions): Macro-F1 {m['d']:.3f} vs {m['paired_v']:.3f}, "
                      f"delta {m['delta']:+.3f} [95% CI {m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]."]
    rk = report.get("ranking")
    if rk:
        lines += ["", "| scope | cases | VOC r | Kendall tau_a | Succ-Fail Diff (raw) | Diff x scale |",
                  "|---|---:|---:|---:|---:|---:|"]
        for scope in ("all_single_views", "main_camera"):
            v = rk[scope]
            lines.append(f"| {scope} | {v['n_cases']}/{v['expected']} | {_fmt(v['voc_pearson_r'])} | "
                         f"{_fmt(v['kendall_tau_a'])} | {_fmt(v['succ_fail_diff'])} | "
                         f"{_fmt(v['succ_fail_diff_scaled'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True,
                    help="predictions .jsonl or a rewardbox bench shard <model>.json")
    ap.add_argument("--name", default=None, help="display name (default: from the file)")
    ap.add_argument("--cohort", choices=("single_view", "multiview"), default="single_view")
    ap.add_argument("--mode", choices=("exact", "interpolate"), default="exact",
                    help="how curve predictions are read on the 1 FPS grid (interpolate is flagged)")
    ap.add_argument("--value-scale", type=float, default=1.0,
                    help="multiply Succ-Fail Diff for display, e.g. 0.01 for percent outputs")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--benchmark-root", type=Path, default=ROOT)
    ap.add_argument("-o", "--output", type=Path, help="write the full report as JSON")
    args = ap.parse_args(argv)

    default_name, preds = load_predictions(args.predictions)
    name = args.name or default_name
    report = evaluate(preds, cohort=args.cohort, mode=args.mode, value_scale=args.value_scale,
                      root=args.benchmark_root, bootstrap=not args.no_bootstrap)
    report["model"] = name
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(render(name, report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
