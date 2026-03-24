"""search/vector.py のユニットテスト。"""

from __future__ import annotations

import sqlite3
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from fuga_memory.db.repository import MemoryRepository


class TestSearchVector:
    """search_vector 関数のテスト。"""

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

    def _make_query_vec(self, seed: int = 0, dim: int = 768) -> list[float]:
        """テスト用クエリベクトルを生成する。"""
        return np.random.default_rng(seed).random(dim).astype(np.float32).tolist()

    def test_returns_results(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        """記憶が存在する場合に結果が返ること。"""
        from fuga_memory.search.vector import search_vector

        self._populate(initialized_db, mock_encoder, sample_memories)
        query_vec = self._make_query_vec()
        results = search_vector(initialized_db, query_vec, top_k=3)
        assert len(results) <= 3

    def test_result_has_required_fields(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """結果の辞書に必須フィールドが含まれること。"""
        from fuga_memory.search.vector import search_vector

        repo = MemoryRepository(initialized_db, mock_encoder)
        repo.save("テスト", "session-001")
        query_vec = self._make_query_vec()
        results = search_vector(initialized_db, query_vec, top_k=5)
        if results:
            r = results[0]
            assert "id" in r
            assert "content" in r
            assert "session_id" in r
            assert "source" in r
            assert "created_at" in r
            assert "distance" in r

    def test_raises_value_error_for_wrong_dim(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """embedding_dim と query_vec の長さが異なる場合に ValueError が発生すること。"""
        from fuga_memory.search.vector import search_vector

        wrong_dim_vec = [0.1] * 100  # 768 ではなく 100 次元
        with pytest.raises(ValueError):
            search_vector(initialized_db, wrong_dim_vec, embedding_dim=768)

    def test_respects_top_k(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
    ) -> None:
        """top_k が結果数の上限として機能すること。"""
        from fuga_memory.search.vector import search_vector

        self._populate(initialized_db, mock_encoder, sample_memories)
        query_vec = self._make_query_vec()
        results = search_vector(initialized_db, query_vec, top_k=2)
        assert len(results) <= 2

    def test_returns_list_type(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """戻り値が list であること。"""
        from fuga_memory.search.vector import search_vector

        query_vec = self._make_query_vec()
        results = search_vector(initialized_db, query_vec)
        assert isinstance(results, list)

    def test_empty_db_returns_empty_list(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """DBが空の場合は空リストを返すこと。"""
        from fuga_memory.search.vector import search_vector

        query_vec = self._make_query_vec()
        results = search_vector(initialized_db, query_vec, top_k=5)
        assert results == []

    @pytest.mark.parametrize("top_k", [1, 3, 5])
    def test_top_k_various_values(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
        sample_memories: list[dict[str, Any]],
        top_k: int,
    ) -> None:
        """様々な top_k の値で結果数が制限されること。"""
        from fuga_memory.search.vector import search_vector

        self._populate(initialized_db, mock_encoder, sample_memories)
        query_vec = self._make_query_vec()
        results = search_vector(initialized_db, query_vec, top_k=top_k)
        assert len(results) <= top_k

    def test_default_embedding_dim_is_768(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """embedding_dim のデフォルトは 768 で、768次元ベクトルが受け入れられること。"""
        from fuga_memory.search.vector import search_vector

        vec_768 = self._make_query_vec(dim=768)
        # デフォルト embedding_dim=768 で例外が発生しないこと
        results = search_vector(initialized_db, vec_768)
        assert isinstance(results, list)

    def test_wrong_dim_raises_with_custom_embedding_dim(
        self,
        initialized_db: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """カスタム embedding_dim と長さが合わない場合も ValueError が発生すること。"""
        from fuga_memory.search.vector import search_vector

        # embedding_dim=256 なのに 512 次元を渡す
        wrong_vec = self._make_query_vec(dim=512)
        with pytest.raises(ValueError):
            search_vector(initialized_db, wrong_vec, embedding_dim=256)
