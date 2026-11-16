"""Shared pytest behaviour.

Many offline tests run the *pinned official source* of an upstream project
(prompt builders, parsers, samplers) without loading any weights.  A fresh
clone has no such checkouts, so a test that reaches one raises
``UpstreamNotFoundError``; that is reported as a skip with the fix
(``rewardbox fetch-upstream ...``) instead of a failure.

Tests that construct a model whose optional stack is not installed (torch,
vLLM, requests, ...) are skipped the same way, so a light ``pip install -e
.[dev]`` runs the whole suite without false failures.

Set ``REWARDBOX_REQUIRE_UPSTREAM=1`` (CI with checkouts and the full stack) to
turn all of these skips back into failures, so nothing missing can make a run
look green.
"""
from __future__ import annotations

import os

import pytest
from rewardbox.core.spec import DependencyError
from rewardbox.upstream import UpstreamNotFoundError, find_upstream

_REQUIRE = os.environ.get("REWARDBOX_REQUIRE_UPSTREAM") == "1"

# Optional third-party stacks a light ``pip install -e .[dev]`` does not have.
# A missing *rewardbox* module, or anything not listed, still fails loudly.
_OPTIONAL_MODULES = {
    "torch", "torchvision", "transformers", "vllm", "accelerate", "peft", "unsloth",
    "swift", "qwen_vl_utils", "huggingface_hub", "requests", "zmq", "lz4", "agentlace",
    "decord", "av", "flash_attn",
}


def pytest_runtest_setup(item):
    """``@pytest.mark.upstream("robometer", ...)``: skip when a needed checkout is absent."""
    for mark in item.iter_markers("upstream"):
        missing = [key for key in mark.args if find_upstream(key) is None]
        if missing and not _REQUIRE:
            pytest.skip(f"upstream checkout missing: {missing} "
                        f"(rewardbox fetch-upstream {' '.join(missing)})")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if _REQUIRE or call.excinfo is None or report.when not in ("setup", "call"):
        return
    exc = call.excinfo.value
    if call.excinfo.errisinstance(UpstreamNotFoundError):
        reason = f"upstream checkout missing: {exc}"
    elif call.excinfo.errisinstance(DependencyError):
        reason = f"optional dependencies missing: {str(exc).splitlines()[0]}"
    elif (call.excinfo.errisinstance(ModuleNotFoundError)
          and (exc.name or "").split(".")[0] in _OPTIONAL_MODULES):
        reason = f"optional module missing: {exc.name}"
    else:
        return
    report.outcome = "skipped"
    report.longrepr = (str(item.path), item.location[1] or 0, reason)

