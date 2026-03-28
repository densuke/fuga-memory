"""fastmcp を使った MCP サーバー定義。

グローバルな接続・エンコーダ・設定を遅延初期化し、MCP ツール関数から参照する。
テストでは _conn / _encoder を直接差し替えてインメモリ DB・モックを注入できる。
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# 入力値の上限定数
_MAX_CONTENT_LENGTH = 100_000  # 100,000文字（マルチバイト含む）
_MAX_QUERY_LENGTH = 4_096  # 検索クエリの最大文字数
_MAX_TOP_K = 100
_MAX_LIMIT = 200

mcp: FastMCP = FastMCP("fuga-memory")

# ---------------------------------------------------------------------------
# グローバル依存（テストで差し替え可能）
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection | None = None
_encoder: Encoder | None = None
_config: Config | None = None


def _get_config() -> Config:
    """グローバル Config を返す。未初期化なら Config.load() で初期化する。"""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def _get_conn() -> sqlite3.Connection:
    """グローバル DB 接続を返す。未初期化なら Config で初期化する。"""
    global _conn
    if _conn is None:
        config = _get_config()
        _conn = get_connection(config.db_path)
        initialize_schema(_conn, config.embedding_dim)
    return _conn


def _get_encoder() -> Encoder:
    """グローバルエンコーダを返す。未初期化なら ModelLoader で初期化する。"""
    global _encoder
    if _encoder is None:
        config = _get_config()
        from fuga_memory.embedding.onnx_cache import get_onnx_cache_dir

        cache_dir = get_onnx_cache_dir(config.model_name, config.onnx_cache_dir)
        loader = ModelLoader(config.model_name, config.thread_workers, cache_dir=cache_dir)
        _encoder = loader.get_encoder()
    return _encoder


# ---------------------------------------------------------------------------
# MCP ツール
# ---------------------------------------------------------------------------


@mcp.tool()
def save_memory(content: str, session_id: str, source: str = "manual") -> dict[str, Any]:
    """記憶を保存する。

    Args:
        content: 保存するテキスト（1文字以上、100,000文字以下）。
        session_id: セッション識別子。
        source: 記憶のソース（デフォルト: "manual"）。

    Returns:
        {"id": int, "status": "saved"}

    Raises:
        ValueError: content が空または上限を超えた場合。
    """
    if not content:
        logger.warning("save_memory: 空の content が拒否されました")
        raise ValueError("content は空文字列にできません")
    if len(content) > _MAX_CONTENT_LENGTH:
        logger.warning("save_memory: content サイズ超過 (%d文字)", len(content))
        raise ValueError(
            f"content が最大サイズを超えています: {len(content)} > {_MAX_CONTENT_LENGTH}"
        )
    conn = _get_conn()
    encoder = _get_encoder()
    config = _get_config()
    repo = MemoryRepository(conn, encoder, config.embedding_dim)
    memory_id = repo.save(content, session_id, source)
    return {"id": memory_id, "status": "saved"}


@mcp.tool()
def search_memory(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """ハイブリッド検索（FTS + ベクトル + RRF + 時間減衰）。

    Args:
        query: 検索クエリ文字列。
        top_k: 返す最大件数（デフォルト: 5、1 以上）。

    Returns:
        [{"id", "score", "content", "session_id", "source", "created_at"}, ...]
        score の降順でソート済み。

    Raises:
        ValueError: query が 4,096 文字を超える場合、top_k が 1 未満または 100 超の場合。
    """
    if len(query) > _MAX_QUERY_LENGTH:
        logger.warning("search_memory: query サイズ超過 (%d文字)", len(query))
        raise ValueError(f"query が最大長を超えています: {len(query)} > {_MAX_QUERY_LENGTH}")
    if top_k < 1:
        raise ValueError(f"top_k は 1 以上である必要があります: {top_k}")
    if top_k > _MAX_TOP_K:
        raise ValueError(f"top_k は {_MAX_TOP_K} 以下である必要があります: {top_k}")
    conn = _get_conn()
    encoder = _get_encoder()
    config = _get_config()

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

    エンコーダ不要の読み取り専用操作のため、モデルの初期化を発生させない。

    Args:
        limit: 返す最大件数（デフォルト: 20、1 以上）。

    Returns:
        [{"session_id", "memory_count", "last_updated"}, ...]

    Raises:
        ValueError: limit が 1 未満または 200 超の場合。
    """
    if limit < 1:
        raise ValueError(f"limit は 1 以上である必要があります: {limit}")
    if limit > _MAX_LIMIT:
        raise ValueError(f"limit は {_MAX_LIMIT} 以下である必要があります: {limit}")
    conn = _get_conn()
    cur = conn.execute(
        """
        SELECT session_id,
               COUNT(*)        AS memory_count,
               MAX(created_at) AS last_updated
        FROM memories
        GROUP BY session_id
        ORDER BY last_updated DESC, MAX(id) DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [dict(row) for row in cur.fetchall()]
