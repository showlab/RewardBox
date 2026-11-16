"""Small helpers that keep credentials out of CLI artifacts and process argv.

Adapters may accept credentials programmatically, but benchmark identities are
persisted to disk and isolated workers receive their options on the command
line.  Those two boundaries must never receive a secret value.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "access_token",
    "hub_token",
    "password",
    "secret",
    "token",
}
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_access_token",
    "_auth_token",
    "_password",
    "_secret",
    "_token",
)
REDACTED = "<redacted>"


def is_sensitive_key(key: object) -> bool:
    """Return whether a mapping key conventionally carries a credential.

    Delimiter normalisation deliberately does not classify names such as
    ``tokenizer`` or ``max_tokens`` as secrets.
    """
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return normalized in _SECRET_KEY_NAMES or normalized.endswith(_SECRET_KEY_SUFFIXES)


def sensitive_option_paths(value: object, prefix: str = "") -> list[str]:
    """List credential-bearing paths in nested model options."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if is_sensitive_key(key):
                found.append(path)
            else:
                found.extend(sensitive_option_paths(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            found.extend(sensitive_option_paths(item, path))
    return found


def redact_sensitive_options(value: Any) -> Any:
    """Return a JSON-friendly copy with credential values replaced."""
    if isinstance(value, Mapping):
        return {
            key: REDACTED if is_sensitive_key(key) else redact_sensitive_options(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [redact_sensitive_options(item) for item in value]
    if isinstance(value, list):
        return [redact_sensitive_options(item) for item in value]
    return value
