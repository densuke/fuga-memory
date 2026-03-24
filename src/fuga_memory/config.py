"""アプリケーション設定。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_db_path() -> Path:
    return Path.home() / ".local" / "share" / "fuga-memory" / "memories.db"


def _default_thread_workers() -> int:
    import os

    cpu = os.cpu_count() or 2
    return max(1, cpu // 2)


@dataclass
class Config:
    db_path: Path = field(default_factory=_default_db_path)
    model_name: str = "cl-nagoya/ruri-v3-310m"
    embedding_dim: int = 768
    thread_workers: int = field(default_factory=_default_thread_workers)
    rrf_k: int = 60
    decay_halflife_days: int = 30
    default_top_k: int = 5

    @classmethod
    def from_env(cls) -> Config:
        """環境変数からConfigを生成する。"""
        config = cls()

        if db_path_str := os.environ.get("FUGA_MEMORY_DB_PATH"):
            config.db_path = Path(db_path_str)

        if model_name := os.environ.get("FUGA_MEMORY_MODEL_NAME"):
            config.model_name = model_name

        if workers_str := os.environ.get("FUGA_MEMORY_THREAD_WORKERS"):
            config.thread_workers = int(workers_str)

        return config
