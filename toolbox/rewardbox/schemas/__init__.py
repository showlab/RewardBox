"""Public, independently auditable JSON schema files.  The lightweight runtime
equivalent is ``rewardbox.core.results._validate_payload`` (jsonschema is not a runtime
dependency).
"""
from __future__ import annotations

from pathlib import Path

RESULT_V1_SCHEMA_PATH = Path(__file__).resolve().parent / "result-v1.schema.json"


def load_result_v1_schema() -> dict:
    import json
    return json.loads(RESULT_V1_SCHEMA_PATH.read_text())
