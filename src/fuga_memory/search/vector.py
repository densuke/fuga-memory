"""sqlite-vec を使ったベクトル類似度検索。"""

from __future__ import annotations

import sqlite3
import struct
from typing import Any


def search_vector(
    conn: sqlite3.Connection,
    query_vec: list[float],
    top_k: int = 5,
    embedding_dim: int = 768,
) -> list[dict[str, Any]]:
    """sqlite-vec で KNN 検索し、マッチした記憶のリストを返す。

    Args:
        conn: SQLite接続
        query_vec: クエリベクトル（embedding_dim 次元の float リスト）
        top_k: 返す最大件数（デフォルト: 5）
        embedding_dim: 埋め込みの次元数（デフォルト: 768）

    Returns:
        id/content/session_id/source/created_at/distance を含む辞書のリスト。

    Raises:
        ValueError: embedding_dim と query_vec の長さが異なる場合。
    """
    if len(query_vec) != embedding_dim:
        raise ValueError(
            f"query_vec の次元数が不正です: expected {embedding_dim}, got {len(query_vec)}"
        )

    packed = struct.pack(f"{len(query_vec)}f", *query_vec)
    cur = conn.execute(
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
