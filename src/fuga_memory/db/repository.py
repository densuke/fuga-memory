"""記憶のCRUD操作。"""

from __future__ import annotations

import sqlite3
import struct
from typing import Any, Protocol


class Encoder(Protocol):
    """テキストをベクトルに変換するプロトコル。"""

    def encode(self, text: str) -> list[float]: ...


class MemoryRepository:
    """memoriesテーブルへのCRUD操作を提供する。"""

    def __init__(self, conn: sqlite3.Connection, encoder: Encoder) -> None:
        self._conn = conn
        self._encoder = encoder

    def save(
        self,
        content: str,
        session_id: str,
        source: str = "manual",
    ) -> int:
        """記憶を保存し、生成されたIDを返す。"""
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
        embedding = self._encoder.encode(content)
        packed = struct.pack(f"{len(embedding)}f", *embedding)
        self._conn.execute(
            "INSERT INTO memories_vec (id, embedding) VALUES (?, ?)",
            (memory_id, packed),
        )

        self._conn.commit()
        return memory_id

    def search_fts(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """FTS5全文検索で記憶を検索する。"""
        cur = self._conn.execute(
            """
            SELECT m.id, m.content, m.session_id, m.source, m.created_at,
                   rank AS rank
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, top_k),
        )
        return [dict(row) for row in cur.fetchall()]

    def search_vector(self, query_vec: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """ベクトル類似度で記憶を検索する。"""
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
