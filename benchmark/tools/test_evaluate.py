#!/usr/bin/env python3
"""Unit tests for tools/evaluate.py on the real benchmark labels (no media, no models).

    python tools/test_evaluate.py        # or: python -m pytest tools/test_evaluate.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

BENCH = ev.load_benchmark()


def gt_curve_preds(cases):
    """Predict the GT curve itself on each grid: perfect direction, VOC = GT-vs-time."""
    out = {}
    for cid, c in cases.items():
        t = c["grid"] / c["fps"]
        v = np.interp(t, [p["time"] for p in c["curve"]], [p["value"] for p in c["curve"]])
        out[cid] = {"frame_indices": c["grid"].tolist(), "progress": v.tolist()}
    return out


def test_population_counts():
    cases = BENCH["cases"]
    nonflat = sum(int(np.count_nonzero(c["gt_directions"])) for c in cases.values())
    rising = sum(int((c["gt_directions"] > 0).sum()) for c in cases.values())
    assert len(cases) == 1117 and len(BENCH["main_camera"]) == 201
    assert (nonflat, rising, nonflat - rising) == (17381, 15679, 1702)


def test_oracle_curve_is_perfect():
    d = ev.score_direction(BENCH, gt_curve_preds(BENCH["cases"]), bootstrap=False)
    assert d["overall"]["macro_f1"] == 1.0 and d["coverage"]["missing"] == 0


def test_always_rising_reference():
    preds = {cid: {"directions": [1] * (len(c["grid"]) - 1)} for cid, c in BENCH["cases"].items()}
    o = ev.score_direction(BENCH, preds, bootstrap=False)["overall"]
    assert abs(o["macro_f1"] - 0.47425892316999396) < 1e-12 and o["falling"]["f1"] == 0.0


def test_action_effect_is_scored_on_effect_showing_views():
    cases = BENCH["cases"]
    members = [c for c in cases.values() if "action_effect" in c["categories"]]
    scored = [c for c in members if c["action_effect_scored"]]
    assert (len(members), len({c["clip_id"] for c in members}), len(scored)) == (159, 55, 86)
    assert all(c["view_label"] == "V" for c in scored)
    assert not any(c["action_effect_scored"] for c in cases.values() if c not in members)
    preds = {cid: {"directions": [1] * (len(c["grid"]) - 1)} for cid, c in cases.items()}
    cell = ev.score_direction(BENCH, preds, bootstrap=False)["by_category"]["action_effect"]
    assert (cell["n"], cell["n_falling"]) == (1012, 128)
    assert cell["macro_f1"] == cell["always_rising_macro_f1"]


def test_time_only_reference_ranking():
    preds = {}
    for cid, c in BENCH["cases"].items():
        frames = sorted(set(c["grid"].tolist()) | {c["frames"] - 1})
        preds[cid] = {"frame_indices": frames, "progress": [f / (c["frames"] - 1) for f in frames]}
    r = ev.score_ranking(BENCH, preds)["all_single_views"]
    assert abs(r["voc_pearson_r"] - 1.0) < 1e-12 and r["kendall_tau_a"] == 0.0


def test_ties_count_as_wrong_and_missing_is_reported():
    cid = next(c for c, v in BENCH["cases"].items() if np.count_nonzero(v["gt_directions"]) > 3)
    n = len(BENCH["cases"][cid]["grid"]) - 1
    d = ev.score_direction(BENCH, {cid: {"directions": [0] * n}}, bootstrap=False)
    assert d["overall"]["accuracy"] == 0.0 and d["overall"]["pred_tie_share"] == 1.0
    d = ev.score_direction(BENCH, {cid: {"directions": [None] * n}}, bootstrap=False)
    assert d["overall"] == {"n": 0} and d["coverage"]["missing"] == d["coverage"]["nonflat_transitions"]


def test_exact_mode_needs_grid_frames_interpolate_does_not():
    cid, c = next(iter(BENCH["cases"].items()))
    off_grid = {cid: {"frame_indices": [1, c["frames"] - 2], "progress": [0.0, 1.0]}}
    assert ev.score_direction(BENCH, off_grid, "exact", bootstrap=False)["overall"] == {"n": 0}
    d = ev.score_direction(BENCH, off_grid, "interpolate", bootstrap=False)
    assert d["interpolated"] and d["overall"]["n"] > 0


def test_result_does_not_depend_on_prediction_order():
    preds = gt_curve_preds(BENCH["cases"])
    rng = np.random.default_rng(0)
    for p in preds.values():                       # some noise so the bootstrap is non-trivial
        p["progress"] = (np.asarray(p["progress"]) + rng.normal(0, 0.05, len(p["progress"]))).tolist()
    keys = list(preds)
    random.Random(1).shuffle(keys)
    a = ev.score_direction(BENCH, preds)
    b = ev.score_direction(BENCH, {k: preds[k] for k in keys})
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_rejects_unknown_ids_and_bad_curves():
    for bad in ({"nope": {"directions": []}},
                {next(iter(BENCH["cases"])): {"frame_indices": [5, 3], "progress": [0, 1]}}):
        try:
            ev.score_direction(BENCH, bad, bootstrap=False)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_loads_jsonl_and_bench_shard():
    cid, c = next(iter(BENCH["cases"].items()))
    row = {"frame_indices": c["grid"].tolist(), "progress": list(range(len(c["grid"])))}
    with tempfile.TemporaryDirectory() as tmp:
        jl = Path(tmp) / "m.jsonl"
        jl.write_text(json.dumps({"id": cid, **row}) + "\n")
        shard = Path(tmp) / "m.json"
        shard.write_text(json.dumps({"model": "m", "results": {cid: {**row, "model": "m"}}}))
        assert ev.load_predictions(jl)[1] == ev.load_predictions(shard)[1] == {cid: row}


def test_pair_protocol_shard_is_read_as_directions():
    """vlm-pair / jev-1.14-pair shards carry per-transition judgements (null = unanswered)."""
    cid = next(c for c, v in BENCH["cases"].items() if np.count_nonzero(v["gt_directions"]) > 3)
    grid = BENCH["cases"][cid]["grid"].tolist()
    directions = [1] * (len(grid) - 1)
    directions[0] = None
    result = {"model": "vlm-pair", "frame_indices": grid,
              "progress": [0.0] + [0.0 if d is None else float(d) for d in directions],
              "raw": {"transition_directions": directions},
              "meta": {"output_kind": "transition_directions"}}
    with tempfile.TemporaryDirectory() as tmp:
        shard = Path(tmp) / "vlm-pair.json"
        shard.write_text(json.dumps({"model": "vlm-pair", "results": {cid: result}}))
        preds = ev.load_predictions(shard)[1]
        assert preds[cid]["directions"] == directions
        gt = BENCH["cases"][cid]["gt_directions"]
        d = ev.score_direction(BENCH, preds, bootstrap=False)
        assert d["overall"]["n"] == int(np.count_nonzero(gt[1:]))      # the null is missing, not wrong
        assert d["overall"]["pred_rising"] == d["overall"]["n"]
        assert ev.score_ranking(BENCH, preds) is None                   # no curve, no ranking metrics
        result["frame_indices"] = grid[:-1]                             # not the 1 FPS grid
        result["raw"]["transition_directions"] = directions[:-1]
        shard.write_text(json.dumps({"model": "vlm-pair", "results": {cid: result}}))
        try:
            ev.score_direction(BENCH, ev.load_predictions(shard)[1], bootstrap=False)
        except ValueError as error:
            assert "1 FPS grid" in str(error)
        else:
            raise AssertionError("accepted pair judgements off the 1 FPS grid")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
