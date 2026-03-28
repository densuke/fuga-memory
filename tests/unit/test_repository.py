"""repository.py のユニットテスト。"""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from fuga_memory.db.repository import MemoryRepository


class TestSaveMemory:
    def test_save_returns_id(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        memory_id = repo.save("テスト内容", "session-001")
        assert isinstance(memory_id, int)
        assert memory_id > 0

    def test_save_persists_content(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("テスト内容", "session-001")
        cur = initialized_db.execute("SELECT content FROM memories")
        row = cur.fetchone()
        assert row[0] == "テスト内容"

    def test_save_with_source(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("内容", "session-001", source="claude_code")
        cur = initialized_db.execute("SELECT source FROM memories")
        assert cur.fetchone()[0] == "claude_code"

    def test_save_default_source_is_manual(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("内容", "session-001")
        cur = initialized_db.execute("SELECT source FROM memories")
        assert cur.fetchone()[0] == "manual"

    def test_save_stores_embedding_in_vec_table(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        memory_id = repo.save("テスト内容", "session-001")
        cur = initialized_db.execute("SELECT id FROM memories_vec WHERE id = ?", (memory_id,))
        assert cur.fetchone() is not None

    def test_save_indexes_fts(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("Rustのlifetime", "session-001")
        cur = initialized_db.execute(
            "SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'lifetime'"
        )
        assert cur.fetchone() is not None


class TestSearchFts:
    def test_fts_returns_matching_results(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        for m in sample_memories:
            repo.save(m["content"], m["session_id"], m["source"])

        results = repo.search_fts("Rust", top_k=5)
        assert len(results) >= 1
        assert any("Rust" in r["content"] for r in results)

    def test_fts_returns_empty_for_no_match(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("Python の話", "session-001")
        results = repo.search_fts("存在しないキーワードxyz123", top_k=5)
        assert results == []

    def test_fts_respects_top_k(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        for m in sample_memories:
            repo.save(m["content"], m["session_id"], m["source"])

        results = repo.search_fts("の", top_k=2)
        assert len(results) <= 2

    def test_fts_result_has_required_fields(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("SQLiteのFTS5を使う", "session-001")
        results = repo.search_fts("FTS5", top_k=5)
        assert len(results) >= 1
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "session_id" in r
        assert "created_at" in r
        assert "rank" in r


class TestSearchVector:
    def test_vector_search_returns_results(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        for m in sample_memories:
            repo.save(m["content"], m["session_id"], m["source"])

        query_vec = np.random.default_rng(0).random(768).astype(np.float32).tolist()
        results = repo.search_vector(query_vec, top_k=3)
        assert len(results) <= 3

    def test_vector_search_result_has_required_fields(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("テスト", "session-001")
        query_vec = np.random.default_rng(0).random(768).astype(np.float32).tolist()
        results = repo.search_vector(query_vec, top_k=5)
        if results:
            r = results[0]
            assert "id" in r
            assert "content" in r
            assert "distance" in r


class TestListSessions:
    def test_list_sessions_returns_unique_sessions(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        for m in sample_memories:
            repo.save(m["content"], m["session_id"], m["source"])

        sessions = repo.list_sessions(limit=10)
        session_ids = [s["session_id"] for s in sessions]
        assert len(session_ids) == len(set(session_ids))  # ユニーク

    def test_list_sessions_respects_limit(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        for m in sample_memories:
            repo.save(m["content"], m["session_id"], m["source"])

        sessions = repo.list_sessions(limit=1)
        assert len(sessions) <= 1

    def test_list_sessions_empty_db(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        sessions = repo.list_sessions()
        assert sessions == []


class TestDeleteMemory:
    def test_delete_removes_from_all_tables(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        memory_id = repo.save("削除予定の内容", "session-001")

        # 削除実行
        success = repo.delete_memory(memory_id)
        assert success is True

        # 各テーブルから消えているか確認
        assert (
            initialized_db.execute(
                "SELECT COUNT(*) FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            initialized_db.execute(
                "SELECT COUNT(*) FROM memories_fts WHERE rowid = ?", (memory_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            initialized_db.execute(
                "SELECT COUNT(*) FROM memories_vec WHERE id = ?", (memory_id,)
            ).fetchone()[0]
            == 0
        )

    def test_delete_returns_false_if_not_existed(
        self, initialized_db: sqlite3.Connection, mock_encoder: MagicMock
    ) -> None:
        repo = MemoryRepository(initialized_db, mock_encoder)
        # 存在しないID(999)を削除
        success = repo.delete_memory(999)
        assert success is False
