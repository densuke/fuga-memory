"""共通テスト fixture。"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from fuga_memory.db.connection import get_connection
from fuga_memory.db.schema import initialize_schema


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """テスト毎の一時DBファイルパスを返す。"""
    return tmp_path / "test_memories.db"


@pytest.fixture
def initialized_db(tmp_db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """スキーマ初期化済みのDB接続を返す。テスト終了時にクローズ。"""
    conn = get_connection(tmp_db_path)
    initialize_schema(conn)
    yield conn
    conn.close()


@pytest.fixture
def mock_encoder() -> MagicMock:
    """固定ランダムベクトルを返すダミーエンコーダ。"""
    rng = np.random.default_rng(seed=42)

    encoder = MagicMock()
    encoder.encode.side_effect = lambda text: rng.random(768).astype(np.float32).tolist()
    return encoder


@pytest.fixture
def sample_memories() -> list[dict[str, Any]]:
    """テスト用の記憶データ（5件）。"""
    return [
        {
            "content": "Rustのlifetimeについて議論した。借用チェッカーの仕組みを理解した。",
            "session_id": "session-001",
            "source": "claude_code",
        },
        {
            "content": "Pythonのasyncioとスレッドプールの使い分けを整理した。",
            "session_id": "session-001",
            "source": "claude_code",
        },
        {
            "content": "SQLiteのWALモードとFTS5の設定方法を調べた。",
            "session_id": "session-002",
            "source": "claude_code",
        },
        {
            "content": "MCPプロトコルの仕様を確認した。stdio transportとHTTP transportの違い。",
            "session_id": "session-002",
            "source": "claude_code",
        },
        {
            "content": "sentence-transformersのONNXバックエンドでモデルを高速化できることを確認。",
            "session_id": "session-003",
            "source": "manual",
        },
    ]
