"""MCPサーバーツール関数のユニットテスト。

DB はインメモリ SQLite、エンコーダは mock を使用。
fastmcp のトランスポート層はテスト対象外。
"""

from __future__ import annotations

import gc
import sqlite3
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import fuga_memory.server as srv
from fuga_memory.db.schema import initialize_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_encoder() -> MagicMock:
    """固定ランダムベクトルを返すダミーエンコーダ。"""
    rng = np.random.default_rng(seed=42)
    encoder = MagicMock()
    encoder.encode.side_effect = lambda text: rng.random(768).astype(np.float32).tolist()
    return encoder


@pytest.fixture
def in_memory_conn() -> Generator[sqlite3.Connection]:
    """インメモリ SQLite 接続（スキーマ初期化済み）。テスト終了時にクローズ。"""
    import sqlite_vec

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    initialize_schema(conn)
    yield conn
    conn.close()
    gc.collect()  # Python 3.13 でのGC遅延による ResourceWarning を防ぐ


@pytest.fixture
def server_deps(
    in_memory_conn: sqlite3.Connection,
    mock_encoder: MagicMock,
) -> Generator[dict[str, Any]]:
    """server モジュールのグローバル依存を差し替え、テスト後にリセットする。

    テスト内では `srv` モジュールを直接参照してツール関数を呼び出す。
    DB への直接アクセスが必要な場合は `server_deps["conn"]` を使用する。
    """
    srv._conn = in_memory_conn  # type: ignore[attr-defined]
    srv._encoder = mock_encoder  # type: ignore[attr-defined]
    yield {"conn": in_memory_conn}
    # テスト後にサーバーグローバルをリセットして参照を解放
    srv._conn = None  # type: ignore[attr-defined]
    srv._encoder = None  # type: ignore[attr-defined]
    srv._config = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# save_memory のテスト
# ---------------------------------------------------------------------------


class TestSaveMemory:
    def test_save_returns_id_and_status(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """save_memory は {"id": int, "status": "saved"} を返す。"""
        result = srv.save_memory("テスト記憶", "session-001", "manual")

        assert isinstance(result["id"], int)
        assert result["id"] > 0
        assert result["status"] == "saved"

    def test_save_persists_to_db(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """save_memory で保存した記憶が DB に存在する。"""
        conn = server_deps["conn"]

        result = srv.save_memory("SQLiteの使い方", "session-002", "claude_code")
        saved_id = result["id"]

        row = conn.execute("SELECT * FROM memories WHERE id = ?", (saved_id,)).fetchone()
        assert row is not None
        assert row["content"] == "SQLiteの使い方"
        assert row["session_id"] == "session-002"
        assert row["source"] == "claude_code"

    def test_save_default_source_is_manual(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """source を省略した場合のデフォルト値は 'manual'。"""
        conn = server_deps["conn"]

        result = srv.save_memory("デフォルトソーステスト", "session-003")
        row = conn.execute("SELECT source FROM memories WHERE id = ?", (result["id"],)).fetchone()
        assert row["source"] == "manual"

    def test_save_multiple_memories(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """複数の記憶を保存でき、それぞれ異なる ID が割り当てられる。"""
        r1 = srv.save_memory("記憶1", "session-001")
        r2 = srv.save_memory("記憶2", "session-001")
        r3 = srv.save_memory("記憶3", "session-002")

        ids = {r1["id"], r2["id"], r3["id"]}
        assert len(ids) == 3  # 全て異なる ID

    def test_save_empty_content_does_not_raise(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """空文字列のコンテンツも保存できる（バリデーションは呼び出し側の責任）。"""
        result = srv.save_memory("", "session-001")
        assert result["status"] == "saved"


# ---------------------------------------------------------------------------
# search_memory のテスト
# ---------------------------------------------------------------------------


class TestSearchMemory:
    def _populate(self, contents: list[tuple[str, str, str]]) -> None:
        """(content, session_id, source) のリストを一括保存するヘルパー。"""
        for content, session_id, source in contents:
            srv.save_memory(content, session_id, source)

    def test_search_returns_list(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """search_memory はリストを返す。"""
        self._populate([("Pythonのasyncioについて調べた", "s1", "manual")])
        results = srv.search_memory("Python")
        assert isinstance(results, list)

    def test_search_result_has_required_fields(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """検索結果の各要素は必須フィールドを持つ。"""
        self._populate([("Rustのlifetimeを理解した", "s1", "manual")])
        results = srv.search_memory("Rust")
        assert len(results) >= 1
        item = results[0]
        for field in ("id", "score", "content", "session_id", "source", "created_at"):
            assert field in item, f"フィールド {field!r} が結果に含まれていない"

    def test_search_top_k_limits_results(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """top_k を指定すると返る件数が top_k 以下になる。"""
        for i in range(10):
            srv.save_memory(f"記憶{i}: Pythonのテスト", "s1", "manual")

        results = srv.search_memory("Python", top_k=3)
        assert len(results) <= 3

    def test_search_no_match_returns_empty_list(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """DB が空のとき、search_memory は空リストを返す。"""
        results = srv.search_memory("全くマッチしないクエリXYZ")
        assert results == []

    def test_search_score_is_float(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """スコアは float 型で返る。"""
        self._populate([("SQLiteのFTS5を使ったフルテキスト検索", "s1", "manual")])
        results = srv.search_memory("SQLite")
        if results:
            assert isinstance(results[0]["score"], float)

    def test_search_default_top_k_is_5(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """top_k のデフォルト値は 5 で、返る件数は 5 以下になる。"""
        for i in range(10):
            srv.save_memory(f"記憶{i}: テストコンテンツ Python", "s1", "manual")

        results = srv.search_memory("テスト")
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# list_sessions のテスト
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_sessions_returns_list(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """list_sessions はリストを返す。"""
        result = srv.list_sessions()
        assert isinstance(result, list)

    def test_list_sessions_empty_when_no_memories(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """記憶が0件のとき、list_sessions は空リストを返す。"""
        result = srv.list_sessions()
        assert result == []

    def test_list_sessions_has_required_fields(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """セッション一覧の各要素は必須フィールドを持つ。"""
        srv.save_memory("記憶", "session-001", "manual")
        sessions = srv.list_sessions()
        assert len(sessions) >= 1
        item = sessions[0]
        for field in ("session_id", "memory_count", "last_updated"):
            assert field in item, f"フィールド {field!r} が結果に含まれていない"

    def test_list_sessions_counts_memories_per_session(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """各セッションの memory_count が正しく集計される。"""
        srv.save_memory("記憶1", "session-A", "manual")
        srv.save_memory("記憶2", "session-A", "manual")
        srv.save_memory("記憶3", "session-B", "manual")

        sessions = srv.list_sessions()
        counts = {s["session_id"]: s["memory_count"] for s in sessions}
        assert counts["session-A"] == 2
        assert counts["session-B"] == 1

    def test_list_sessions_limit(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """limit パラメータで返す件数を制限できる。"""
        for i in range(5):
            srv.save_memory("記憶", f"session-{i:03d}", "manual")

        result = srv.list_sessions(limit=3)
        assert len(result) <= 3

    def test_list_sessions_default_limit_is_20(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """limit のデフォルト値は 20 で、返る件数は 20 以下になる。"""
        for i in range(25):
            srv.save_memory("記憶", f"session-{i:03d}", "manual")

        result = srv.list_sessions()
        assert len(result) <= 20

    def test_list_sessions_ordered_by_last_updated(
        self,
        server_deps: dict[str, Any],
    ) -> None:
        """セッション一覧は last_updated の降順（同値時は id 降順）で返る。"""
        conn = server_deps["conn"]
        srv.save_memory("古い記憶", "session-old", "manual")
        srv.save_memory("新しい記憶", "session-new", "manual")

        # created_at は秒精度のため同一秒になり得る。順序を確定させるため明示的に更新する。
        conn.execute(
            "UPDATE memories SET created_at='2020-01-01T00:00:01Z' WHERE session_id='session-old'"
        )
        conn.execute(
            "UPDATE memories SET created_at='2020-01-01T00:00:02Z' WHERE session_id='session-new'"
        )
        conn.commit()

        sessions = srv.list_sessions()
        assert len(sessions) >= 2
        assert sessions[0]["last_updated"] >= sessions[1]["last_updated"]
        assert sessions[0]["session_id"] == "session-new"
