"""Local dataset manifests (csv/jsonl).  Larger datasets such as LIBERO/OXE come later;
see the header of rewardbox/data/manifest.py."""
from .manifest import Manifest, ManifestEntry

__all__ = ["Manifest", "ManifestEntry"]
