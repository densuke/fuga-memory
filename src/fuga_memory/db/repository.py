"""記憶のCRUD操作。"""

from __future__ import annotations

import sqlite3
import struct
from typing import Any

from fuga_memory.embedding.encoder import Encoder
from fuga_memory.search.fts import search_fts as _search_fts_safe

EMBEDDING_DIM = 768

__all__ = ["Encoder", "MemoryRepository"]


class MemoryRepository:
    """memoriesテーブルへのCRUD操作を提供する。"""

    def __init__(
        self,
        conn: sqlite3.Connection,
        encoder: Encoder,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        self._conn = conn
        self._encoder = encoder
        self._embedding_dim = embedding_dim

    def save(
        self,
        content: str,
        session_id: str,
        source: str = "manual",
    ) -> int:
        """記憶を保存し、生成されたIDを返す。3つの書き込みをトランザクションで原子的に実行する。"""
        embedding = self._encoder.encode(content)
        self._validate_dim(embedding, "encoder output")

        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO memories (content, session_id, source) VALUES (?, ?, ?)",
                (content, session_id, source),
            )
            memory_id = cur.lastrowid
            assert memory_id is not None

            # FTS5 インデックス更新
            self._conn.execute(
                "INSERT INTO memories_fts (rowid, content) VALUES (?, ?)",
                (memory_id, content),
            )

            # ベクトル保存
            packed = struct.pack(f"{len(embedding)}f", *embedding)
            self._conn.execute(
                "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
                (memory_id, packed),
            )

        return memory_id

    def search_fts(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """FTS5全文検索で記憶を検索する。クエリは自動でサニタイズされる。"""
        return _search_fts_safe(self._conn, query, top_k=top_k)

    def search_vector(self, query_vec: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """ベクトル類似度で記憶を検索する。"""
        self._validate_dim(query_vec, "query vector")
        packed = struct.pack(f"{len(query_vec)}f", *query_vec)
        cur = self._conn.execute(
            """
            SELECT m.id, m.content, m.session_id, m.source, m.created_at,
                   v.distance
            FROM memories_vec v
            JOIN memories m ON v.id = m.id
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (packed, top_k),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """セッション一覧を返す（直近更新順）。"""
        cur = self._conn.execute(
            """
            SELECT session_id,
                   COUNT(*)        AS memory_count,
                   MAX(created_at) AS last_updated
            FROM memories
            GROUP BY session_id
            ORDER BY last_updated DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    @classmethod
    def delete_memory(cls, conn: sqlite3.Connection, memory_id: int) -> bool:
        """指定されたIDの記憶をすべてのテーブルから削除する。削除された場合はTrueを返す。"""
        with conn:
            # memories_vec から削除
            conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
            # memories_fts から削除
            conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))
            # memories から削除し、削除された行数を確認
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def _validate_dim(self, vec: list[float], name: str) -> None:
        if len(vec) != self._embedding_dim:
            raise ValueError(
                f"{name} の次元数が不正です: expected {self._embedding_dim}, got {len(vec)}"
            )
