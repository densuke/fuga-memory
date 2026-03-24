"""fastmcp を使った MCP サーバー定義。

グローバルな接続・エンコーダを遅延初期化し、MCP ツール関数から参照する。
テストでは _conn / _encoder を直接差し替えてインメモリ DB・モックを注入できる。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastmcp import FastMCP

from fuga_memory.config import Config
from fuga_memory.db.connection import get_connection
from fuga_memory.db.repository import MemoryRepository
from fuga_memory.db.schema import initialize_schema
from fuga_memory.embedding.encoder import Encoder
from fuga_memory.embedding.loader import ModelLoader
from fuga_memory.search.fts import search_fts
from fuga_memory.search.fusion import reciprocal_rank_fusion
from fuga_memory.search.vector import search_vector

mcp: FastMCP = FastMCP("fuga-memory")

# ---------------------------------------------------------------------------
# グローバル依存（テストで差し替え可能）
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection | None = None
_encoder: Encoder | None = None


def _get_conn() -> sqlite3.Connection:
    """グローバル DB 接続を返す。未初期化なら Config.from_env() で初期化する。"""
    global _conn
    if _conn is None:
        config = Config.from_env()
        _conn = get_connection(config.db_path)
        initialize_schema(_conn, config.embedding_dim)
    return _conn


def _get_encoder() -> Encoder:
    """グローバルエンコーダを返す。未初期化なら ModelLoader で初期化する。"""
    global _encoder
    if _encoder is None:
        config = Config.from_env()
        loader = ModelLoader(config.model_name, config.thread_workers)
        _encoder = loader.get_encoder()
    return _encoder


# ---------------------------------------------------------------------------
# MCP ツール
# ---------------------------------------------------------------------------


@mcp.tool()
def save_memory(content: str, session_id: str, source: str = "manual") -> dict[str, Any]:
    """記憶を保存する。

    Args:
        content: 保存するテキスト。
        session_id: セッション識別子。
        source: 記憶のソース（デフォルト: "manual"）。

    Returns:
        {"id": int, "status": "saved"}
    """
    conn = _get_conn()
    encoder = _get_encoder()
    config = Config.from_env()
    repo = MemoryRepository(conn, encoder, config.embedding_dim)
    memory_id = repo.save(content, session_id, source)
    return {"id": memory_id, "status": "saved"}


@mcp.tool()
def search_memory(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """ハイブリッド検索（FTS + ベクトル + RRF + 時間減衰）。

    Args:
        query: 検索クエリ文字列。
        top_k: 返す最大件数（デフォルト: 5）。

    Returns:
        [{"id", "score", "content", "session_id", "source", "created_at"}, ...]
        score の降順でソート済み。
    """
    conn = _get_conn()
    encoder = _get_encoder()
    config = Config.from_env()

    query_vec = encoder.encode(query)
    fts_results = search_fts(conn, query, top_k=top_k)
    vec_results = search_vector(conn, query_vec, top_k=top_k, embedding_dim=config.embedding_dim)
    fused = reciprocal_rank_fusion(
        fts_results,
        vec_results,
        k=config.rrf_k,
        halflife_days=config.decay_halflife_days,
    )
    return fused[:top_k]


@mcp.tool()
def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """セッション一覧を返す（直近更新順）。

    Args:
        limit: 返す最大件数（デフォルト: 20）。

    Returns:
        [{"session_id", "memory_count", "last_updated"}, ...]
    """
    conn = _get_conn()
    encoder = _get_encoder()
    config = Config.from_env()
    repo = MemoryRepository(conn, encoder, config.embedding_dim)
    return repo.list_sessions(limit=limit)
