"""Opt-in filesystem guard for an unchanged official RoboDopamine pipeline."""
from __future__ import annotations

import re
from types import FunctionType, MethodType
from uuid import uuid4


def bounded_run_pipeline(grm):
    """Bound only the output-directory slug; the full task still reaches prompts.

    Use private function globals rather than changing the upstream module or
    Python's process-global regex module.
    """
    method = grm.run_pipeline
    function = method.__func__
    original_re = function.__globals__.get("re", re)

    class DirectoryRegex:
        def sub(self, pattern, replacement, value, *args, **kwargs):
            result = original_re.sub(pattern, replacement, value, *args, **kwargs)
            if pattern == r"[^A-Za-z0-9_]" and replacement == "_" and len(result) > 96:
                return result[:80] + "_" + uuid4().hex[:12]
            return result

        def __getattr__(self, name):
            return getattr(original_re, name)

    namespace = {**function.__globals__, "re": DirectoryRegex()}
    wrapped = FunctionType(function.__code__, namespace, function.__name__,
                           function.__defaults__, function.__closure__)
    wrapped.__kwdefaults__ = function.__kwdefaults__
    return MethodType(wrapped, grm)
