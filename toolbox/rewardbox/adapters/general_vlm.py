"""General-purpose VLMs as progress evaluators: the Sequence and Pair protocols.

This is a RewardBox protocol, not an official reward model.  A general VLM is
asked about task progress through a chat API, with the prompts shipped in
``rewardbox/prompts/vlm_*_v1.txt`` (written for the RRM-Bench-1K study after the
RoboPulse++ evaluation of general VLMs; they are not a verbatim upstream prompt).

* ``vlm-sequence``: one request per trajectory with every sampled observation in
  chronological order; the model returns ``{"progress": [...]}`` with one value in
  [0, 100] per observation.  ``RewardOutput.progress`` is that array.
* ``vlm-pair``: one request per adjacent pair of sampled observations (BEFORE,
  AFTER); the model returns ``<score>+1</score>`` or ``<score>-1</score>``.
  ``RewardOutput.progress`` is 0 for the first observation and then the judgement
  of the transition that ends at each observation (the layout of
  ``lrm-contrastive``).  It is not a progress curve: read
  ``raw["transition_directions"]`` (``None`` marks an answer in the wrong format).

Request layout (both protocols): the prompt text, then for every observation a
label (``Observation i`` or ``BEFORE``/``AFTER``) followed by ``View: <name>`` and
the image of each view, in the order primary, left_wrist, right_wrist.

Three wire formats are supported through plain HTTP: ``openai`` (chat
completions; also local vLLM / API gateways), ``anthropic`` (Messages) and
``gemini`` (generateContent).  The model id is a required option; sampling
settings, reasoning effort and similar provider parameters stay at the provider
default unless given in ``extra_body``.  Credentials are read from an
environment variable and never written to results.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from ..core.base import BaseAdapter
from ..core.enums import ImplementationKind, RewardMode, SignalType
from ..core.spec import ModelSpec, Requirements
from ..core.types import RewardOutput, SamplingSpec, Trajectory

PROTOCOL_ID = "rewardbox-general-vlm-v1"
PROMPT_VERSION = "robopulsepp-inspired-v1"
VIEW_ORDER = ("primary", "left_wrist", "right_wrist")
OUTPUT_KIND_DIRECTIONS = "transition_directions"
_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PAPER = "https://arxiv.org/abs/2608.14284"

_API_DEFAULTS = {
    #          base URL                                     key variable         auth header
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "bearer"),
    "anthropic": ("https://api.anthropic.com", "ANTHROPIC_API_KEY", "x-api-key"),
    "gemini": ("https://generativelanguage.googleapis.com", "GEMINI_API_KEY", "x-goog-api-key"),
}
_AUTH_HEADERS = ("bearer", "x-api-key", "x-goog-api-key")
_RESERVED_BODY_KEYS = {
    "openai": {"model", "messages"},
    "anthropic": {"model", "messages", "system", "max_tokens"},
    "gemini": {"contents", "systemInstruction"},
}

_REQ = Requirements(
    pip=("requests>=2.32.4", "opencv-python-headless>=4.7"),
    python=">=3.10",
    note=(
        "No local weights: images are sent to a chat API. Pass the model id with model=..., "
        "choose the wire format with api=openai|anthropic|gemini, and export the API key "
        "(OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY, or the variable named by "
        "api_key_env). `available` only says the HTTP dependencies are installed."
    ),
    install_hint=("pip install -e '.[general-vlm]'",),
)
_SAMPLING = SamplingSpec(strategy="fps", fps=1.0, include_last=False)
_COMMON = {
    "module": __name__,
    "signal_type": SignalType.GENERATED_TEXT,
    "requires": _REQ,
    "sampling": _SAMPLING,
    "optional_views": ("left_wrist", "right_wrist"),
    "paper": _PAPER,
    "upstream": "rewardbox.adapters.general_vlm (RewardBox protocol; prompts/vlm_*_v1.txt)",
    "implementation_kind": ImplementationKind.REWARDBOX_EXTENSION,
}
SEQUENCE_SPEC = ModelSpec(
    name="vlm-sequence",
    cls="GeneralVLMSequenceAdapter",
    reward_mode=RewardMode.PROGRESS,
    score_range=(0.0, 100.0),
    description=(
        "General VLM, Sequence protocol: all sampled observations (1 FPS by default) in one "
        "request, one 0-100 completion score per observation. The API model is the required "
        "option model=...; prompts are RewardBox's own (v1)."
    ),
    **_COMMON,
)
PAIR_SPEC = ModelSpec(
    name="vlm-pair",
    cls="GeneralVLMPairAdapter",
    reward_mode=RewardMode.PREFERENCE,
    score_range=(-1.0, 1.0),
    description=(
        "General VLM, Pair protocol: one request per adjacent pair of sampled observations, "
        "forced +1/-1 direction. progress is 0 for the first observation, then the judgement of "
        "each transition; it is not a curve (see raw.transition_directions)."
    ),
    **_COMMON,
)
SPECS = [SEQUENCE_SPEC, PAIR_SPEC]


# ---------------------------------------------------------------- prompts and layout
def load_prompt(name: str) -> str:
    return (_PROMPT_DIR / f"vlm_{name}_v1.txt").read_text(encoding="utf-8")


def ordered_views(traj: Trajectory) -> list[str]:
    """primary, left_wrist, right_wrist, then any other view by name."""
    return sorted(traj.views, key=lambda v: (VIEW_ORDER.index(v) if v in VIEW_ORDER else 99, v))


def user_prompt(style: str, instruction: str, n_observations: int) -> str:
    return Template(load_prompt(f"{style}_user")).substitute(
        task_instruction=instruction, n_observations=n_observations)


def request_parts(traj: Trajectory, style: str, indices: list[int], image) -> list[dict]:
    """Provider-neutral user content: ``{"text": ...}`` and ``{"image": (media_type, b64)}``.

    ``indices`` are positions on the sampled grid; ``image(position, view)`` returns the
    encoded image of one observation and view.
    """
    parts: list[dict] = [{"text": user_prompt(style, traj.instruction, len(indices))}]
    for order, position in enumerate(indices):
        label = ("BEFORE" if order == 0 else "AFTER") if style == "pair" else f"Observation {order}"
        parts.append({"text": label})
        for view in ordered_views(traj):
            parts.append({"text": f"View: {view}"})
            parts.append({"image": image(position, view)})
    return parts


# ---------------------------------------------------------------- answer parsing
def _strip_markdown_fence(text: str) -> tuple[str, bool]:
    """Remove one Markdown code fence that wraps the whole answer; nothing else is repaired."""
    match = re.fullmatch(r"```[A-Za-z]*[ \t]*\n(.*?)\n?```", text.strip(), flags=re.DOTALL)
    return (match.group(1), True) if match else (text, False)


def parse_sequence(answer: str, n_observations: int) -> tuple[list[float], bool]:
    """``{"progress": [n finite numbers in [0, 100]]}`` and nothing else."""
    def reject_constant(value):
        raise ValueError(f"non-finite JSON number: {value}")

    def reject_duplicate_keys(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    text, fenced = _strip_markdown_fence(answer)
    try:
        parsed = json.loads(text, parse_constant=reject_constant,
                            object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the answer is not a JSON object: {exc}") from None
    if not isinstance(parsed, dict) or set(parsed) != {"progress"}:
        raise ValueError('the answer must be a JSON object with exactly the key "progress"')
    values = parsed["progress"]
    if not isinstance(values, list) or len(values) != n_observations:
        got = len(values) if isinstance(values, list) else type(values).__name__
        raise ValueError(f"expected {n_observations} progress values, got {got}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or not 0 <= v <= 100 for v in values):
        raise ValueError("progress values must be finite numbers in [0, 100]")
    return [float(v) for v in values], fenced


def parse_pair(answer: str) -> int:
    match = re.fullmatch(r"<score>([+-]1)</score>", answer.strip())
    if match is None:
        raise ValueError("the answer must be exactly <score>+1</score> or <score>-1</score>")
    return int(match.group(1))


# ---------------------------------------------------------------- HTTP
def redact(text: object, secret: str | None) -> str:
    value = str(text)
    return value.replace(secret, "[REDACTED]") if secret else value


def post_json(url: str, headers: dict, body: dict, *, secret: str | None, timeout_s: float,
              max_retries: int, label: str) -> tuple[dict, int]:
    """POST a JSON body and return (JSON object, HTTP attempts).

    HTTP 408/409/429/5xx and network errors are retried with exponential backoff; any
    other status fails at once.  Redirects are not followed and the credential is removed
    from every error message.
    """
    import requests

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body,
                                     timeout=(30.0, timeout_s), allow_redirects=False)
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise RuntimeError(f"{label} request failed: {redact(exc, secret)}") from None
        else:
            status = response.status_code
            if status == 200:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if not isinstance(payload, dict):
                    raise RuntimeError(f"{label} returned HTTP 200 without a JSON object")
                return payload, attempt + 1
            if status not in (408, 409, 429) and status < 500 or attempt == max_retries:
                detail = redact(getattr(response, "text", "") or "", secret)[:800]
                raise RuntimeError(f"{label} HTTP {status}: {detail}")
        time.sleep(min(2.0 ** attempt, 60.0))
    raise AssertionError("unreachable")


def validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("base_url must be an HTTP(S) address without credentials, query or fragment")
    return base_url.rstrip("/")


def resolve_api_key(explicit: str | None, env_name: str) -> str:
    key = explicit if explicit is not None else os.environ.get(env_name)
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"no API key: export {env_name} (or choose another variable with api_key_env=...)")
    return key.strip()


def _wire_content(api: str, parts: list[dict]) -> list[dict]:
    out = []
    for part in parts:
        if "text" in part:
            out.append({"text": part["text"]} if api == "gemini"
                       else {"type": "text", "text": part["text"]})
            continue
        media_type, data = part["image"]
        if api == "openai":
            out.append({"type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"}})
        elif api == "anthropic":
            out.append({"type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data}})
        else:
            out.append({"inlineData": {"mimeType": media_type, "data": data}})
    return out


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_answer(api: str, payload: dict) -> tuple[str, dict]:
    """Answer text and response metadata; a truncated, refused or empty answer is an error."""
    if api == "openai":
        choices = payload.get("choices")
        _require(isinstance(choices, list) and choices and isinstance(choices[0], dict),
                 "the response has no choices")
        finish = choices[0].get("finish_reason")
        _require(finish in (None, "stop"), f"generation did not finish normally: finish_reason={finish}")
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") in ("text", "output_text"))
        answer, stop = content, finish
    elif api == "anthropic":
        stop = payload.get("stop_reason")
        _require(stop == "end_turn", f"generation did not finish normally: stop_reason={stop}")
        blocks = payload.get("content")
        _require(isinstance(blocks, list), "the response has no content blocks")
        answer = "".join(b["text"] for b in blocks if isinstance(b, dict)
                         and b.get("type") == "text" and isinstance(b.get("text"), str))
    else:
        candidates = payload.get("candidates")
        _require(isinstance(candidates, list) and candidates and isinstance(candidates[0], dict),
                 "the response has no candidates")
        stop = candidates[0].get("finishReason")
        _require(stop in (None, "STOP"), f"generation did not finish normally: finishReason={stop}")
        content = candidates[0].get("content")
        parts = content.get("parts", []) if isinstance(content, dict) else []
        answer = "".join(p["text"] for p in parts if isinstance(p, dict)
                         and isinstance(p.get("text"), str) and p.get("thought") is not True)
    _require(isinstance(answer, str) and bool(answer.strip()), "the response has no answer text")
    usage = payload.get("usageMetadata" if api == "gemini" else "usage")
    return answer, {
        "reported_model": payload.get("modelVersion" if api == "gemini" else "model"),
        "response_id": payload.get("responseId" if api == "gemini" else "id"),
        "stop_reason": stop,
        "usage": usage if isinstance(usage, dict) else None,
    }


# ---------------------------------------------------------------- pair helpers (shared with Jev)
def run_pairs(n_observations: int, ask, max_concurrency: int) -> list:
    """Run ``ask(after_position)`` for every adjacent pair, a few at a time, in order.

    Any raised error (transport failure) fails the whole trajectory, so a benchmark
    re-run with ``--retry-errors`` scores it again as a whole.
    """
    if n_observations < 2:
        raise ValueError("the Pair protocol needs at least two sampled observations")
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(ask, position) for position in range(1, n_observations)]
        results, errors = [], []
        for position, future in enumerate(futures, start=1):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(None)
                errors.append(f"pair {position - 1}->{position}: {exc}")
    if errors:
        raise RuntimeError(
            f"{len(errors)} of {n_observations - 1} pair requests failed; first: {errors[0]}")
    return results


def directions_to_progress(directions: list) -> np.ndarray:
    """[0, d_1, ..., d_{T-1}] with an unanswered transition written as 0."""
    return np.array([0.0] + [0.0 if d is None else float(d) for d in directions])


def positive_int(name: str, value, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def positive_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


# ---------------------------------------------------------------- adapters
class _GeneralVLMAdapter(BaseAdapter):
    style = ""

    def __init__(
        self,
        spec: ModelSpec,
        api: str = "openai",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        auth_header: str | None = None,
        extra_body: dict | None = None,
        max_tokens: int = 16000,
        timeout_s: float = 600.0,
        max_retries: int = 3,
        image_format: str = "png",
        jpeg_quality: int = 90,
        image_max_side: int | None = None,
        **opts: Any,
    ):
        """``api``: wire format (openai | anthropic | gemini).  ``model``: the provider's
        model id (required).  ``base_url``: API root, e.g. a gateway or a local
        OpenAI-compatible server.  ``api_key_env``: name of the environment variable that
        holds the key.  ``auth_header``: bearer | x-api-key | x-goog-api-key (gateways
        usually want bearer).  ``extra_body``: extra top-level request fields, e.g.
        ``{"output_config": {"effort": "medium"}}``.  ``max_tokens`` is sent only by the
        anthropic format, which requires it.  ``image_format`` png | jpeg and
        ``image_max_side`` control the image payload; nothing is shrunk by default.
        """
        super().__init__(spec, **opts)
        if api not in _API_DEFAULTS:
            raise ValueError(f"api must be one of {sorted(_API_DEFAULTS)}, got {api!r}")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"{spec.name} needs the API model id: model=... "
                f"(CLI: --model-opt model=NAME; bench: --model-opt {spec.name}.model=NAME)")
        default_url, default_env, default_auth = _API_DEFAULTS[api]
        root = validate_base_url(base_url or default_url)
        for suffix in {"anthropic": ("/v1",), "gemini": ("/v1beta",)}.get(api, ()):
            root = root.removesuffix(suffix)
        auth_header = auth_header or default_auth
        if auth_header not in _AUTH_HEADERS:
            raise ValueError(f"auth_header must be one of {list(_AUTH_HEADERS)}")
        extra_body = dict(extra_body or {})
        clash = sorted(set(extra_body) & _RESERVED_BODY_KEYS[api])
        if clash:
            raise ValueError(f"extra_body must not set {clash}; they are built by the adapter")
        if image_format not in ("png", "jpeg"):
            raise ValueError("image_format must be 'png' or 'jpeg'")
        self.api = api
        self.model = model.strip()
        self.base_url = root
        self.api_key_env = api_key_env or default_env
        self.auth_header = auth_header
        self.extra_body = extra_body
        self.max_tokens = positive_int("max_tokens", max_tokens)
        self.timeout_s = positive_float("timeout_s", timeout_s)
        self.max_retries = positive_int("max_retries", max_retries, minimum=0)
        self.image_format = image_format
        self.jpeg_quality = positive_int("jpeg_quality", jpeg_quality)
        self.image_max_side = (None if image_max_side is None
                               else positive_int("image_max_side", image_max_side))
        self._api_key = api_key

    # ---- request
    def _image_source(self, traj: Trajectory):
        """Encode each observation/view once, even when two pair requests share it."""
        from ..core.video import encode_image

        cache: dict[tuple[int, str], tuple[str, str]] = {}
        media_type = f"image/{self.image_format}"

        def image(position: int, view: str) -> tuple[str, str]:
            key = (position, view)
            if key not in cache:
                blob = encode_image(traj.view(view)[position], self.image_format,
                                    jpeg_quality=self.jpeg_quality, max_side=self.image_max_side)
                cache[key] = (media_type, base64.b64encode(blob).decode("ascii"))
            return cache[key]

        return image

    def _ask(self, parts: list[dict]) -> tuple[str, dict]:
        key = resolve_api_key(self._api_key, self.api_key_env)
        system = load_prompt("system")
        content = _wire_content(self.api, parts)
        headers = {"Content-Type": "application/json"}
        if self.auth_header == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        else:
            headers[self.auth_header] = key
        if self.api == "openai":
            url = f"{self.base_url}/chat/completions"
            body = {"model": self.model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": content}]}
        elif self.api == "anthropic":
            url = f"{self.base_url}/v1/messages"
            headers["anthropic-version"] = "2023-06-01"
            body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                    "messages": [{"role": "user", "content": content}]}
        else:
            url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": content}]}
        body.update(self.extra_body)
        payload, attempts = post_json(url, headers, body, secret=key, timeout_s=self.timeout_s,
                                      max_retries=self.max_retries, label=f"{self.api} API")
        try:
            answer, info = _read_answer(self.api, payload)
        except RuntimeError as exc:
            raise RuntimeError(f"{self.api} API: {redact(exc, key)}") from None
        return answer, {**info, "http_attempts": attempts}

    def _meta(self, traj: Trajectory, n_calls: int, **extra) -> dict:
        return {
            "protocol": f"{PROTOCOL_ID}:{self.style}",
            "prompt_version": PROMPT_VERSION,
            "official_verbatim_prompt": False,
            "rewardbox_extension": "general_vlm_progress_protocol",
            "api": self.api,
            "model_name": self.model,
            "base_url": self.base_url,
            "extra_body": self.extra_body,
            "generation_settings": "provider defaults, plus extra_body",
            "credential_source": ("explicit_constructor" if self._api_key is not None
                                  else f"{self.api_key_env}_environment"),
            "views_in_request_order": ordered_views(traj),
            "image_policy": {"format": self.image_format,
                             "jpeg_quality": self.jpeg_quality if self.image_format == "jpeg" else None,
                             "max_side": self.image_max_side},
            "normalized": False,
            "n_backend_calls": n_calls,
            **extra,
        }


class GeneralVLMSequenceAdapter(_GeneralVLMAdapter):
    """All sampled observations in one request; one 0-100 score per observation."""

    style = "sequence"

    def _score(self, traj: Trajectory) -> RewardOutput:
        n = len(traj)
        answer, info = self._ask(
            request_parts(traj, "sequence", list(range(n)), self._image_source(traj)))
        try:
            progress, fenced = parse_sequence(answer, n)
        except ValueError as exc:
            raise RuntimeError(
                f"{self.model} answered in the wrong format ({exc}); answer starts: {answer[:200]!r}"
            ) from None
        return self._out(
            progress, traj,
            raw={"answer": answer, "response": info},
            meta=self._meta(traj, 1, markdown_fence_removed=fenced),
        )


class GeneralVLMPairAdapter(_GeneralVLMAdapter):
    """One request per adjacent pair of sampled observations; forced +1/-1."""

    style = "pair"

    # The registry accepts only explicitly named constructor options, so the shared ones
    # are restated here next to ``max_concurrency``.
    def __init__(
        self,
        spec: ModelSpec,
        api: str = "openai",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        auth_header: str | None = None,
        extra_body: dict | None = None,
        max_tokens: int = 16000,
        timeout_s: float = 600.0,
        max_retries: int = 3,
        image_format: str = "png",
        jpeg_quality: int = 90,
        image_max_side: int | None = None,
        max_concurrency: int = 4,
        **opts: Any,
    ):
        """Options of the Sequence adapter, plus ``max_concurrency``: how many pair
        requests of one trajectory are in flight at the same time."""
        super().__init__(
            spec, api=api, model=model, base_url=base_url, api_key=api_key,
            api_key_env=api_key_env, auth_header=auth_header, extra_body=extra_body,
            max_tokens=max_tokens, timeout_s=timeout_s, max_retries=max_retries,
            image_format=image_format, jpeg_quality=jpeg_quality,
            image_max_side=image_max_side, **opts)
        self.max_concurrency = positive_int("max_concurrency", max_concurrency)

    def _score(self, traj: Trajectory) -> RewardOutput:
        resolve_api_key(self._api_key, self.api_key_env)   # fail once, before any request
        image = self._image_source(traj)

        def ask(position: int) -> dict:
            answer, info = self._ask(request_parts(traj, "pair", [position - 1, position], image))
            try:
                direction, error = parse_pair(answer), None
            except ValueError as exc:
                direction, error = None, str(exc)
            return {"direction": direction, "answer": answer, "format_error": error,
                    "response": info}

        results = run_pairs(len(traj), ask, self.max_concurrency)
        directions = [r["direction"] for r in results]
        return self._out(
            directions_to_progress(directions), traj,
            raw={OUTPUT_KIND_DIRECTIONS: directions, "pairs": results},
            meta=self._meta(
                traj, len(results),
                output_kind=OUTPUT_KIND_DIRECTIONS,
                progress_layout="0 for the first observation, then the judgement of the "
                                "transition ending at each observation (unanswered = 0)",
                n_format_errors=sum(d is None for d in directions),
                max_concurrency=self.max_concurrency,
            ),
        )
