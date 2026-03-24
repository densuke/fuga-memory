"""FTS5 全文検索。"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# FTS5 の特殊構文文字（フレーズ引用符・グルーピング・カラム指定等）
_FTS5_SPECIAL = re.compile(r'["\(\)\*\^\:\.]')


def _sanitize_fts_query(query: str) -> str:
    """FTS5 クエリの特殊構文を除去し、安全な語句検索に変換する。

    FTS5 は "AND"/"OR"/"NOT"（大文字）を論理演算子として解釈する。
    特殊文字と大文字演算子を除去することで構文エラーを防ぐ。

    Args:
        query: ユーザー入力のクエリ文字列

    Returns:
        サニタイズ済みクエリ文字列（トークン間はスペース区切り）
    """
    sanitized = _FTS5_SPECIAL.sub(" ", query)
    # 大文字の論理演算子キーワードを小文字化（FTS5 は大文字限定で解釈）
    sanitized = re.sub(r"\b(AND|OR|NOT)\b", lambda m: m.group().lower(), sanitized)
    return " ".join(sanitized.split())


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """FTS5 で全文検索し、マッチした記憶のリストを返す。

    クエリは自動でサニタイズされる。FTS5 エラー時は空リストを返す（例外を伝播しない）。

    Args:
        conn: SQLite接続
        query: 検索クエリ文字列
        top_k: 返す最大件数（デフォルト: 5）

    Returns:
        id/content/session_id/source/created_at/rank を含む辞書のリスト。
        結果は rank（昇順）でソート済み。
    """
    sanitized = _sanitize_fts_query(query)
    if not sanitized:
        return []

    try:
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
            (sanitized, top_k),
        )
        return [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        logger.warning("FTS5 検索でエラーが発生しました（サニタイズ後クエリ: %r）", sanitized)
        return []
