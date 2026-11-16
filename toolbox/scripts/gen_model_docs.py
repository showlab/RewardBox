"""Regenerate docs/models.md and docs/models.json from the live registry.

    python scripts/gen_model_docs.py          # write both files
    python scripts/gen_model_docs.py --check  # exit 1 if they are stale (CI)

Everything per model comes from ``rewardbox.get_model_info`` and
``rewardbox.upstream``; only the method family, runtime class and validation
document per adapter module are curated here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rewardbox  # noqa: E402
from rewardbox import upstream as up  # noqa: E402
from rewardbox.adapters.gvl import credential_env_for_profile  # noqa: E402

# adapter module -> (method family, runtime class, environment, validation or guide doc)
MODULES = {
    "topreward": ("TOPReward", "in-process", "shared py3.11 (+ flash-attn wheel)",
                  "docs/validation.md"),
    "gvl": ("GVL", "hosted API", "shared; GEMINI_API_KEY",
            "docs/validation.md"),
    "roboreward": ("RoboReward", "in-process", "shared py3.11",
                   "docs/validation.md"),
    "robodopamine": ("Robo-Dopamine (GRM)", "in-process vLLM", "shared py3.11",
                     "docs/validation.md"),
    "robometer": ("Robometer", "in-process", "shared py3.11",
                  "docs/validation.md"),
    "sole_r1": ("SOLE-R1", "model server (ZMQ)", "server in its own py3.10 env",
                "docs/validation.md"),
    "lrm": ("LRM", "model server (HTTP)", "server in its own py3.10 env",
            "docs/validation.md"),
    "lrm_vllm": ("LRM", "in-process vLLM (RewardBox backend)", "shared py3.11",
                 "docs/validation.md"),
    "rynnvalue": ("RynnValue", "model server (HTTP)", "server in upstream uv env (py3.10)",
                  "docs/validation.md"),
    "robovaluerl": ("Robo-ValueRL", "model server (HTTP)", "server in its own env",
                    "docs/validation.md"),
    "procvlm": ("ProcVLM", "model server (HTTP)", "server in upstream uv env (py3.10)",
                "docs/validation.md"),
    "vlac": ("VLAC", "isolated worker", ".venv-vlac (py3.10, ms-swift 3.3)",
             "docs/validation.md"),
    "vlac_cut": ("VLAC-Cut", "isolated worker", ".venv-vlac-cut (py3.11, ms-swift 3.12)",
                 "docs/validation.md"),
    "vlac_cut_vllm": ("VLAC-Cut", "in-process vLLM (RewardBox backend)", "shared py3.11",
                      "docs/guides/vllm_adapters.md"),
    "primo": ("PRIMO-R1", "isolated worker", ".venv-primo (py3.11, vLLM 0.7.2)",
              "docs/validation.md"),
    "flora": ("Flora-Rewarder", "isolated worker", ".venv-flora (py3.12, vLLM 0.24)",
              "docs/validation.md"),
    "general_vlm": ("General VLM (protocol)", "hosted API",
                    "shared; OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY or api_key_env",
                    "docs/guides/general_vlm_and_jev.md"),
    "jev": ("Jev (protocol)", "hosted API", "shared; base_url and JEV_API_KEY or api_key_env",
            "docs/guides/general_vlm_and_jev.md"),
}
# Families that are RewardBox protocols around a general model, not reward-model methods.
PROTOCOL_FAMILIES = ("General VLM (protocol)", "Jev (protocol)")


def sampling_text(s: dict) -> str:
    st = s["strategy"]
    if st in ("uniform", "uniform_floor"):
        return f"{st} {s['num_frames']}"
    if st == "fps":
        return f"{s['fps']:g} fps"
    if st == "interval":
        return f"every {s['interval']}"
    return "all frames"


def collect() -> list[dict]:
    rows = []
    for name in rewardbox.list_models():
        info = rewardbox.get_model_info(name)
        module = info["module"].rsplit(".", 1)[-1]
        family, runtime, env, evidence = MODULES[module]
        v = info["verified"]
        rows.append({
            "name": name,
            "method": family,
            "runtime": runtime,
            "environment": env,
            "reward_mode": info["reward_mode"],
            "signal_type": info["signal_type"],
            "implementation_kind": info["implementation_kind"],
            "default_sampling": info["sampling"],
            "required_views": info["required_views"],
            "optional_views": info["optional_views"],
            "score_range": info["score_range"],
            "higher_better": info["higher_better"],
            "supports_pairwise": info["supports_pairwise"],
            "outputs_success": info["outputs_success"],
            "weights": info["weights"],
            "paper": info["paper"],
            "upstream_entry": info["upstream"],
            "upstream_checkouts": [u.key for u in up.upstreams_for_model(name)],
            "credentials_env": ([credential_env_for_profile(name)] if module == "gvl"
                                else ["OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY"]
                                if module == "general_vlm"
                                else ["JEV_API_KEY"] if module == "jev" else []),
            "vram_gb": info["requires"]["vram_gb"],
            "python": info["requires"]["python"],
            "calibration_record": ("score+compare" if v["score"] and v["compare"]
                                   else "score" if v["score"] else None),
            "evidence": evidence,
            "description": info["description"],
        })
    return rows


def render_md(rows: list[dict]) -> str:
    families: dict[str, list[dict]] = {}
    for r in rows:
        families.setdefault(r["method"], []).append(r)
    n_methods = len([f for f in families if f not in PROTOCOL_FAMILIES])
    n_method_rows = len([r for r in rows if r["method"] not in PROTOCOL_FAMILIES])
    out = [
        "# Model zoo",
        "",
        "<!-- generated by scripts/gen_model_docs.py; do not edit by hand -->",
        "",
        f"{len(rows)} registered configurations: {n_method_rows} from {n_methods} reward-model "
        f"methods and {len(rows) - n_method_rows} protocol profiles for general VLMs and Jev. "
        "`rewardbox list` / `rewardbox info NAME --json` show the same data live; "
        "[`models.json`](models.json) is the machine-readable copy.",
        "",
        "Columns: **runtime** says where the model executes (see "
        "[installation](installation.md)); **default sampling** is the upstream recipe "
        "the adapter applies when you do not pass a `SamplingSpec`; **score range** is "
        "the raw native scale (RewardBox never rescales outputs); **calibration** marks a "
        "frozen real-weight regression record in `rewardbox/calibration/records/`; "
        "**kind** is `official_adapter` (calls the authors' code), `model_card_reference` "
        "(no official inference code; follows the model card) or `rewardbox_extension` "
        "(an explicit RewardBox protocol: on top of official code, e.g. prefix curves, or "
        "around a general model, e.g. the Sequence / Pair prompts).",
        "",
        "| method | name | kind | runtime | default sampling | views | score range | checkouts | calibration |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for family in sorted(families, key=str.lower):
        for r in families[family]:
            views = "+".join(r["required_views"])
            if r["optional_views"]:
                views += " (opt. " + ", ".join(r["optional_views"]) + ")"
            lo, hi = r["score_range"]
            out.append(
                f"| {family} | `{r['name']}` | {r['implementation_kind']} | {r['runtime']} | "
                f"{sampling_text(r['default_sampling'])} | {views} | [{lo:g}, {hi:g}] | "
                f"{', '.join(r['upstream_checkouts']) or '—'} | {r['calibration_record'] or '—'} |")
    out += ["", "## Per-method notes", ""]
    for family in sorted(families, key=str.lower):
        rs = families[family]
        out.append(f"### {family}")
        out.append("")
        out.append(f"- Environment: {rs[0]['environment']}")
        papers = sorted({r['paper'] for r in rs if r['paper']})
        if papers:
            out.append(f"- Paper / card: {', '.join(papers)}")
        weights = sorted({r['weights'] for r in rs if r['weights']})
        if weights:
            out.append(f"- Weights: {'; '.join(f'`{w}`' for w in weights)}")
        evid = sorted({r['evidence'] for r in rs})
        out.append(f"- Validation: {', '.join(f'[{Path(e).name}](../{e})' for e in evid)}")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    rows = collect()
    md, js = render_md(rows), json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    targets = {ROOT / "docs" / "models.md": md, ROOT / "docs" / "models.json": js}
    if args.check:
        stale = [str(p) for p, text in targets.items()
                 if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            print("stale: " + ", ".join(stale))
            return 1
        return 0
    for p, text in targets.items():
        p.write_text(text, encoding="utf-8")
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
