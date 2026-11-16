"""Small helper for atomic JSON writes.  ``ResultSet.save()`` and the benchmark's
``ModelShard.save()`` share one "temp file + fsync + os.replace" implementation."""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(path: Path, payload: dict) -> None:
    """Temp file in the same directory + fsync + os.replace.  The temp file is removed
    on error; the target holds either the old or the new content, never a half write.
    **Concurrent writers to the same path are not supported**: callers serialise writes
    or give each writer its own path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
