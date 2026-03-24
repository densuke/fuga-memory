"""search/fts.py のユニットテスト。"""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import MagicMock

import pytest

from fuga_memory.db.repository import MemoryRepository
from fuga_memory.search.fts import search_fts


class TestSearchFts:
    """search_fts 関数のテスト。"""

    def _populate(
        self,
        conn: sqlite3.Connection,
        encoder: MagicMock,
        memories: list[dict[str, Any]],
    ) -> None:
        """テスト用データをDBに投入する。"""
        repo = MemoryRepository(conn, encoder)
        for m in memories:
            repo.save(m["content"], m["session_id"], m["source"])

    def test_returns_matching_results(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        """マッチするキーワードで結果が返ること。"""

        self._populate(initialized_db, mock_encoder, sample_memories)
        results = search_fts(initialized_db, "Rust", top_k=5)
        assert len(results) >= 1
        assert any("Rust" in r["content"] for r in results)

    def test_returns_empty_for_no_match(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """マッチしないキーワードでは空リストを返すこと。"""

        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("Python の話", "session-001")
        results = search_fts(initialized_db, "存在しないキーワードxyz123", top_k=5)
        assert results == []

    def test_respects_top_k(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        """top_k が結果数の上限として機能すること。"""

        self._populate(initialized_db, mock_encoder, sample_memories)
        results = search_fts(initialized_db, "の", top_k=2)
        assert len(results) <= 2

    def test_result_has_required_fields(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """結果の辞書に必須フィールドが含まれること。"""

        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("SQLiteのFTS5を使う", "session-001")
        results = search_fts(initialized_db, "FTS5", top_k=5)
        assert len(results) >= 1
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "session_id" in r
        assert "source" in r
        assert "created_at" in r
        assert "rank" in r

    def test_results_are_sorted_by_rank_ascending(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        """結果が rank の昇順でソートされていること（FTS5のrank は負値で小さい方が高関連度）。"""

        self._populate(initialized_db, mock_encoder, sample_memories)
        results = search_fts(initialized_db, "の", top_k=5)
        if len(results) >= 2:
            ranks = [r["rank"] for r in results]
            assert ranks == sorted(ranks)

    def test_returns_list_type(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """戻り値が list であること。"""

        results = search_fts(initialized_db, "テスト", top_k=5)
        assert isinstance(results, list)

    def test_default_top_k_is_five(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        """top_k のデフォルト値が 5 であること（5件以下の結果）。"""

        # 6件以上のデータを投入
        repo = MemoryRepository(initialized_db, mock_encoder)
        for i in range(10):
            repo.save(f"テストコンテンツ {i}", f"session-{i:03d}")

        results = search_fts(initialized_db, "テストコンテンツ")
        assert len(results) <= 5

    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_top_k_various_values(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
        top_k: int,
    ) -> None:
        """様々な top_k の値で結果数が制限されること。"""

        self._populate(initialized_db, mock_encoder, sample_memories)
        results = search_fts(initialized_db, "の", top_k=top_k)
        assert len(results) <= top_k
