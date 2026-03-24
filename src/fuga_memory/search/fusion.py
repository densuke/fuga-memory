"""RRF（Reciprocal Rank Fusion）と時間減衰を組み合わせたスコア計算。"""

from __future__ import annotations

from typing import Any

from fuga_memory.search.decay import time_decay


def reciprocal_rank_fusion(
    fts_results: list[dict[str, Any]],
    vec_results: list[dict[str, Any]],
    k: int = 60,
    halflife_days: int = 30,
) -> list[dict[str, Any]]:
    """FTS 結果とベクトル検索結果を RRF で統合し、時間減衰を掛けて返す。

    スコア計算:
        rrf_score = sum(1/(k+rank_i) for each list where item appears)
        final_score = rrf_score * time_decay(created_at)

    同じ ID が両リストに現れる場合はスコアを合算する。

    Args:
        fts_results: FTS5 検索結果のリスト
            （id/content/session_id/source/created_at/rank を含む辞書のリスト）
        vec_results: ベクトル検索結果のリスト
            （id/content/session_id/source/created_at/distance を含む辞書のリスト）
        k: RRF のパラメータ（デフォルト: 60）
        halflife_days: 時間減衰の半減期（日数）（デフォルト: 30）

    Returns:
        id/score/content/session_id/source/created_at を含む辞書のリスト。
        score の降順でソート済み。
    """
    # id -> メタデータ と id -> RRF スコア を管理する辞書
    meta: dict[int, dict[str, Any]] = {}
    rrf_scores: dict[int, float] = {}

    # FTS・ベクトル結果をまとめて処理（0-based rank）
    for results_list in (fts_results, vec_results):
        for rank, item in enumerate(results_list):
            item_id = int(item["id"])
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            if item_id not in meta:
                meta[item_id] = {
                    "id": item_id,
                    "content": item["content"],
                    "session_id": item["session_id"],
                    "source": item["source"],
                    "created_at": item["created_at"],
                }

    # 時間減衰を適用してスコアを確定
    results: list[dict[str, Any]] = []
    for item_id, rrf_score in rrf_scores.items():
        item_meta = meta[item_id]
        decay = time_decay(str(item_meta["created_at"]), halflife_days=halflife_days)
        final_score = rrf_score * decay
        results.append(
            {
                "id": item_id,
                "score": final_score,
                "content": item_meta["content"],
                "session_id": item_meta["session_id"],
                "source": item_meta["source"],
                "created_at": item_meta["created_at"],
            }
        )

    # score の降順でソート
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
