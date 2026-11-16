"""Public benchmark entry points.

Lazy exports matter: ``python -m rewardbox.bench.worker`` executes this package's
``__init__`` first.  Importing runner here right away (runner imports worker) would
preload that same worker before runpy executes it and trigger the "unpredictable
behaviour" warning.
"""
from __future__ import annotations

__all__ = ["run_benchmark", "aggregate", "ModelShard", "BenchmarkIdentityError"]


def __getattr__(name: str):
    if name in {"run_benchmark", "aggregate"}:
        from .runner import aggregate, run_benchmark
        return {"run_benchmark": run_benchmark, "aggregate": aggregate}[name]
    if name in {"ModelShard", "BenchmarkIdentityError"}:
        from .shard import BenchmarkIdentityError, ModelShard
        return {
            "ModelShard": ModelShard,
            "BenchmarkIdentityError": BenchmarkIdentityError,
        }[name]
    raise AttributeError(name)
