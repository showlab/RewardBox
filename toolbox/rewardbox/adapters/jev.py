"""Jev 1.14 (TypeSafe ``jev-1.14.0``), a decision model, as a progress evaluator.

Jev returns probabilities over fixed answer options instead of text, so it gets
its own request format, but it follows the same two protocols as
``rewardbox.adapters.general_vlm`` and reuses their instructions:

* ``jev-1.14-sequence``: one request per trajectory with every sampled
  observation and one five-level completion question per observation.
  ``progress`` is the expected level x 25, in [0, 100].
* ``jev-1.14-pair``: one request per adjacent pair (BEFORE, AFTER) with a
  two-option choice, rising or falling; the more probable option is the
  judgement and an exact tie is 0.  ``progress`` uses the Pair layout of
  ``vlm-pair`` (0, then one judgement per transition); read
  ``raw["transition_directions"]``.

Every image is prescaled to 75%, fitted into 280 x 280 on black and sent as
JPEG quality 70, the recipe of the RRM-Bench-1K study.  The request goes to the
``/v1/systemone`` route of the endpoint named by ``base_url``, with a Bearer
key; there is no default endpoint.  A request over the service's input limit
fails; nothing is truncated, split or tiled here.
"""
from __future__ import annotations

import base64
import copy
import math
import uuid
from typing import Any

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory
from .general_vlm import (
    OUTPUT_KIND_DIRECTIONS,
    directions_to_progress,
    load_prompt,
    ordered_views,
    positive_float,
    positive_int,
    post_json,
    redact,
    resolve_api_key,
    run_pairs,
    user_prompt,
    validate_base_url,
)

MODEL = "jev-1.14.0"
_ROUTE = "/v1/systemone"
_HEADERS = {"Content-Type": "application/json", "TypeSafe-Danger-Zone-Images": "true",
            "TypeSafe-Danger-Zone-Long-Context": "true"}
_IMAGE = {"prescale": 0.75, "side": 280, "jpeg_quality": 70}
SEQUENCE_PROTOCOL_ID = "jev114-sequence-fullclip-1fps-letterbox280-jpeg70-v1"
PAIR_PROTOCOL_ID = "jev114-pair-adjacent1fps-letterbox280-jpeg70-v1"
LEVELS = [
    "No achieved physical progress toward the task goal; about 0 percent complete.",
    "A small but meaningful part of the task goal is achieved; about 25 percent complete.",
    "About half of the task goal is physically achieved; about 50 percent complete.",
    "Most of the task goal is achieved but important requirements remain; about 75 percent complete.",
    "All requirements of the task goal are physically satisfied; 100 percent complete.",
]
PAIR_OPTIONS = ("rising", "falling")
PAIR_QUESTION = {
    "type": "choice",
    "instructions": "Compare AFTER against BEFORE. Which direction of physical task progress "
                    "is best supported by the visual observations?",
    "criteria": {"rising": "AFTER is closer to satisfying the task goal than BEFORE.",
                 "falling": "AFTER is further from satisfying the task goal than BEFORE."},
}

_COMMON = {
    "module": __name__,
    "cls": "JevAdapter",
    "signal_type": SignalType.OPTION_PROBABILITY,
    "requires": Requirements(
        pip=("requests>=2.32.4", "opencv-python-headless>=4.7"),
        python=">=3.10",
        note=(
            "Hosted decision model. Set base_url to the API root of your Jev access that serves "
            "/v1/systemone; credential JEV_API_KEY (or the variable named by api_key_env). "
            "`available` only says the HTTP dependencies are installed."
        ),
        install_hint=("pip install -e '.[general-vlm]'",),
    ),
    "sampling": SamplingSpec(strategy="fps", fps=1.0, include_last=False),
    "optional_views": ("left_wrist", "right_wrist"),
    "upstream": "rewardbox.adapters.jev (RewardBox protocol over the jev-1.14.0 API)",
    "implementation_kind": ImplementationKind.REWARDBOX_EXTENSION,
}
SEQUENCE_SPEC = ModelSpec(
    name="jev-1.14-sequence",
    reward_mode=RewardMode.PROGRESS,
    score_range=(0.0, 100.0),
    description=(
        "Jev 1.14, Sequence protocol: all sampled observations (1 FPS by default) in one "
        "request, one five-level completion question per observation; progress is the "
        "expected level x 25."
    ),
    **_COMMON,
)
PAIR_SPEC = ModelSpec(
    name="jev-1.14-pair",
    reward_mode=RewardMode.PREFERENCE,
    score_range=(-1.0, 1.0),
    description=(
        "Jev 1.14, Pair protocol: one request per adjacent pair of sampled observations, "
        "two-option choice (rising / falling). progress is 0 for the first observation, then "
        "the judgement of each transition; it is not a curve (see raw.transition_directions)."
    ),
    **_COMMON,
)
SPECS = [SEQUENCE_SPEC, PAIR_SPEC]


# ---------------------------------------------------------------- request bodies
def _instructions(style: str, instruction: str, n_observations: int) -> tuple[str, str]:
    """The general-VLM system and user prompts without their output-format sentences
    (Jev answers through ``questions``, not through text)."""
    system = load_prompt("system").replace(" Follow the requested output format exactly.", "")
    return system, user_prompt(style, instruction, n_observations).split("\nReturn ")[0]


def _observations(traj: Trajectory, positions: list[int], image) -> tuple[list[dict], dict]:
    observations, assets = [], {}
    for order, position in enumerate(positions):
        views = []
        for view in ordered_views(traj):
            reference = str(uuid.uuid4())
            assets[reference] = {"type": "image", "format": "jpg",
                                 "data_base64": image(position, view)}
            views.append({"view": view, "image": reference})
        observations.append({"observation_index": order, "views": views})
    return observations, assets


def sequence_body(traj: Trajectory, image) -> dict:
    n = len(traj)
    system, scoring = _instructions("sequence", traj.instruction, n)
    observations, assets = _observations(traj, list(range(n)), image)
    questions = {f"progress_{i:04d}": {
        "type": "score",
        "instructions": (
            f"Rate absolute physical task completion at observation index {i} in `observations`. "
            "Use the same completion scale across the sequence. Later observations need not have "
            "higher completion. Do not force the first to zero or the last to full completion."),
        "criteria": LEVELS,
    } for i in range(n)}
    return {"model": MODEL,
            "state": {"evaluation_instructions": system,
                      "task_and_scoring_instructions": scoring, "observations": observations},
            "assets": assets, "questions": questions}


def pair_body(traj: Trajectory, position: int, image) -> dict:
    system, scoring = _instructions("pair", traj.instruction, 2)
    observations, assets = _observations(traj, [position - 1, position], image)
    for observation, label in zip(observations, ("BEFORE", "AFTER"), strict=True):
        observation["label"] = label
    return {"model": MODEL,
            "state": {"evaluation_instructions": system,
                      "task_and_scoring_instructions": scoring, "observations": observations},
            "assets": assets, "questions": {"direction": copy.deepcopy(PAIR_QUESTION)}}


# ---------------------------------------------------------------- response parsing
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _common_checks(payload) -> dict:
    _require(isinstance(payload, dict), "the response is not an object")
    _require(payload.get("model") == MODEL, f"reported model mismatch: {payload.get('model')!r}")
    answers = payload.get("answers")
    _require(isinstance(answers, dict), "the response has no answers")
    return answers


def parse_sequence(payload, n_observations: int) -> dict:
    answers = _common_checks(payload)
    expected = [f"progress_{i:04d}" for i in range(n_observations)]
    _require(set(answers) == set(expected), "answer ids differ from the requested observations")
    for key in expected:
        answer = answers[key]
        _require(isinstance(answer, dict) and answer.get("type") == "score",
                 f"invalid answer type: {key}")
        _require(_number(answer.get("score")) and 0 <= answer["score"] <= 4,
                 f"invalid native score: {key}")
        probabilities = answer.get("probabilities")
        _require(isinstance(probabilities, dict)
                 and set(probabilities) == {str(i) for i in range(5)},
                 f"invalid probability labels: {key}")
        _require(all(_number(v) and 0 <= v <= 1 for v in probabilities.values()),
                 f"invalid probability: {key}")
        _require(abs(sum(probabilities.values()) - 1) <= .0200001,
                 f"probability sum differs from 1: {key}")
        _require(abs(answer["score"] - sum(int(k) * v for k, v in probabilities.items()))
                 <= .0600001, f"score inconsistent with its probabilities: {key}")
    return {"progress": [answers[key]["score"] * 25 for key in expected],
            "native_answers": answers, "usage": payload.get("usage"),
            "reported_model": payload["model"]}


def parse_pair(payload) -> dict:
    answers = _common_checks(payload)
    _require(set(answers) == {"direction"}, "answer ids differ from the request")
    answer = answers["direction"]
    _require(isinstance(answer, dict) and answer.get("type") == "choice", "invalid answer type")
    probabilities = answer.get("probabilities")
    _require(isinstance(probabilities, dict) and set(probabilities) == set(PAIR_OPTIONS),
             "invalid probability labels")
    _require(all(_number(v) and 0 <= v <= 1 for v in probabilities.values()),
             "invalid probability")
    _require(abs(sum(probabilities.values()) - 1) <= .0200001, "probability sum differs from 1")
    rising, falling = probabilities["rising"], probabilities["falling"]
    return {"direction": 1 if rising > falling else -1 if falling > rising else 0,
            "choice": answer.get("choice"), "probabilities": probabilities,
            "native_answers": answers, "usage": payload.get("usage"),
            "reported_model": payload["model"]}


# ---------------------------------------------------------------- adapter
class JevAdapter(BaseAdapter):
    """Sequence or Pair, chosen by the registered name."""

    def __init__(
        self,
        spec: ModelSpec,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "JEV_API_KEY",
        timeout_s: float = 900.0,
        max_retries: int = 3,
        max_concurrency: int = 4,
        **opts: Any,
    ):
        """``base_url`` (required): API root that serves ``/v1/systemone``.  ``api_key_env``: name
        of the environment variable that holds the key.  ``max_concurrency`` applies to the
        Pair protocol only (pair requests of one trajectory in flight at the same time)."""
        super().__init__(spec, **opts)
        if spec.name not in (SEQUENCE_SPEC.name, PAIR_SPEC.name):
            raise ValueError(f"unknown Jev profile: {spec.name!r}")
        self.style = "sequence" if spec.name == SEQUENCE_SPEC.name else "pair"
        if not base_url:
            raise ValueError(
                f"{spec.name} has no default endpoint: pass base_url, the API root of your Jev "
                "access that serves /v1/systemone (CLI: --model-opt base_url=https://...)")
        self.base_url = validate_base_url(base_url).removesuffix(_ROUTE)
        self.api_key_env = api_key_env
        self.timeout_s = positive_float("timeout_s", timeout_s)
        self.max_retries = positive_int("max_retries", max_retries, minimum=0)
        self.max_concurrency = positive_int("max_concurrency", max_concurrency)
        self._api_key = api_key

    def _image_source(self, traj: Trajectory):
        from ..core.video import letterbox_jpeg

        cache: dict[tuple[int, str], str] = {}

        def image(position: int, view: str) -> str:
            key = (position, view)
            if key not in cache:
                blob = letterbox_jpeg(traj.view(view)[position], _IMAGE["side"],
                                      prescale=_IMAGE["prescale"], quality=_IMAGE["jpeg_quality"])
                cache[key] = base64.b64encode(blob).decode("ascii")
            return cache[key]

        return image

    def _post(self, body: dict, key: str) -> tuple[dict, int]:
        return post_json(
            self.base_url + _ROUTE, {**_HEADERS, "Authorization": f"Bearer {key}"}, body,
            secret=key, timeout_s=self.timeout_s, max_retries=self.max_retries, label="Jev API")

    def _meta(self, traj: Trajectory, n_calls: int, **extra) -> dict:
        return {
            "protocol": SEQUENCE_PROTOCOL_ID if self.style == "sequence" else PAIR_PROTOCOL_ID,
            "rewardbox_extension": "jev_progress_protocol",
            "model_name": MODEL,
            "base_url": self.base_url,
            "api_route": _ROUTE,
            "credential_source": ("explicit_constructor" if self._api_key is not None
                                  else f"{self.api_key_env}_environment"),
            "views_in_request_order": ordered_views(traj),
            "image_policy": dict(_IMAGE, letterbox="black", format="jpeg"),
            "normalized": False,
            "n_backend_calls": n_calls,
            **extra,
        }

    def _score(self, traj: Trajectory) -> RewardOutput:
        key = resolve_api_key(self._api_key, self.api_key_env)
        image = self._image_source(traj)
        if self.style == "sequence":
            payload, attempts = self._post(sequence_body(traj, image), key)
            try:
                parsed = parse_sequence(payload, len(traj))
            except ValueError as exc:
                raise RuntimeError(f"Jev answered in an unexpected format: {redact(exc, key)}") from None
            return self._out(
                parsed["progress"], traj,
                raw={"native_answers": parsed["native_answers"], "usage": parsed["usage"]},
                meta=self._meta(traj, 1, progress_conversion="expected level (0-4) x 25",
                                reported_model=parsed["reported_model"], http_attempts=attempts),
            )

        def ask(position: int) -> dict:
            payload, attempts = self._post(pair_body(traj, position, image), key)
            try:
                parsed = parse_pair(payload)
            except ValueError as exc:
                return {"direction": None, "format_error": redact(exc, key),
                        "http_attempts": attempts}
            return {**parsed, "format_error": None, "http_attempts": attempts}

        results = run_pairs(len(traj), ask, self.max_concurrency)
        directions = [r["direction"] for r in results]
        return self._out(
            directions_to_progress(directions), traj,
            raw={OUTPUT_KIND_DIRECTIONS: directions, "pairs": results},
            meta=self._meta(
                traj, len(results),
                output_kind=OUTPUT_KIND_DIRECTIONS,
                progress_layout="0 for the first observation, then the judgement of the "
                                "transition ending at each observation (unanswered or tie = 0)",
                direction_rule="more probable of rising/falling; an exact tie is 0",
                n_format_errors=sum(d is None for d in directions),
                max_concurrency=self.max_concurrency,
            ),
        )
