"""FTS5 全文検索。"""

from __future__ import annotations

import sqlite3
from typing import Any


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """FTS5 で全文検索し、マッチした記憶のリストを返す。

    Args:
        conn: SQLite接続
        query: 検索クエリ文字列
        top_k: 返す最大件数（デフォルト: 5）

    Returns:
        id/content/session_id/source/created_at/rank を含む辞書のリスト。
        結果は rank（昇順）でソート済み。
    """
    cur = conn.execute(
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
