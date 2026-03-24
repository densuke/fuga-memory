"""search/fusion.py のユニットテスト。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def _make_created_at(days_ago: float = 0) -> str:
    """指定日数前の ISO8601 文字列を生成する。"""
    dt = datetime.now(tz=UTC) - timedelta(days=days_ago)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class TestReciprocalRankFusion:
    """reciprocal_rank_fusion 関数のテスト。"""

    def _make_fts_result(
        self, id_: int, content: str = "コンテンツ", days_ago: float = 1
    ) -> dict[str, object]:
        """FTS 検索結果形式の辞書を生成する。"""
        return {
            "id": id_,
            "content": content,
            "session_id": f"session-{id_:03d}",
            "source": "claude_code",
            "created_at": _make_created_at(days_ago),
            "rank": -1.0 * id_,  # FTS5 の rank は負値
        }

    def _make_vec_result(
        self, id_: int, content: str = "コンテンツ", days_ago: float = 1
    ) -> dict[str, object]:
        """ベクトル検索結果形式の辞書を生成する。"""
        return {
            "id": id_,
            "content": content,
            "session_id": f"session-{id_:03d}",
            "source": "claude_code",
            "created_at": _make_created_at(days_ago),
            "distance": 0.1 * id_,
        }

    def test_returns_list(self) -> None:
        """戻り値が list であること。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([], [])
        assert isinstance(result, list)

    def test_empty_inputs_return_empty(self) -> None:
        """両方が空リストの場合は空リストを返すこと。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_fts_only_returns_results(self) -> None:
        """FTS 結果のみで fusion が動作すること。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        fts = [self._make_fts_result(1), self._make_fts_result(2)]
        result = reciprocal_rank_fusion(fts, [])
        assert len(result) == 2

    def test_vec_only_returns_results(self) -> None:
        """ベクトル結果のみで fusion が動作すること。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        vec = [self._make_vec_result(1), self._make_vec_result(2)]
        result = reciprocal_rank_fusion([], vec)
        assert len(result) == 2

    def test_result_has_required_fields(self) -> None:
        """結果の辞書に必須フィールドが含まれること。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        fts = [self._make_fts_result(1, "テスト内容")]
        result = reciprocal_rank_fusion(fts, [])
        assert len(result) == 1
        r = result[0]
        assert "id" in r
        assert "score" in r
        assert "content" in r
        assert "session_id" in r
        assert "source" in r
        assert "created_at" in r

    def test_sorted_by_score_descending(self) -> None:
        """結果が score の降順でソートされていること。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        fts = [
            self._make_fts_result(1),
            self._make_fts_result(2),
            self._make_fts_result(3),
        ]
        result = reciprocal_rank_fusion(fts, [])
        if len(result) >= 2:
            scores = [r["score"] for r in result]
            assert scores == sorted(scores, reverse=True)

    def test_same_id_scores_are_summed(self) -> None:
        """同じ ID が両リストに現れる場合はスコアが合算されること。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        shared_id = 42
        fts = [self._make_fts_result(shared_id, "共通コンテンツ", days_ago=0.1)]
        vec = [self._make_vec_result(shared_id, "共通コンテンツ", days_ago=0.1)]

        # 単独リストでのスコア
        result_fts_only = reciprocal_rank_fusion(fts, [])
        result_vec_only = reciprocal_rank_fusion([], vec)
        # 両方含む場合のスコア
        result_both = reciprocal_rank_fusion(fts, vec)

        score_fts = result_fts_only[0]["score"]
        score_vec = result_vec_only[0]["score"]
        score_both = result_both[0]["score"]

        # 両方含む場合のスコアは各単独スコアより大きいこと
        assert score_both > score_fts
        assert score_both > score_vec

    def test_time_decay_reduces_old_content_score(self) -> None:
        """古いコンテンツは新しいコンテンツよりスコアが低いこと。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        # 同じ順位（1位）で、一方は1日前、もう一方は300日前
        fts_new = [self._make_fts_result(1, "新しい内容", days_ago=1)]
        fts_old = [self._make_fts_result(1, "古い内容", days_ago=300)]

        result_new = reciprocal_rank_fusion(fts_new, [])
        result_old = reciprocal_rank_fusion(fts_old, [])

        assert result_new[0]["score"] > result_old[0]["score"]

    def test_k_parameter_affects_score(self) -> None:
        """k パラメータを変えるとスコアが変わること。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        fts = [self._make_fts_result(1)]
        result_k60 = reciprocal_rank_fusion(fts, [], k=60)
        result_k10 = reciprocal_rank_fusion(fts, [], k=10)

        # k=10 のほうが k=60 よりスコアが大きい（1/(10+1) > 1/(60+1)）
        # ただし decay は同じなので比率で判断
        assert result_k10[0]["score"] > result_k60[0]["score"]

    def test_score_formula_rrf(self) -> None:
        """RRF スコア計算が正しいこと: score = 1/(k+rank) * decay。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        # 既知の値で計算を検証
        # 1件のみ、FTS の 1位（0-based index 0）、k=60
        # FTS での RRF スコア = 1/(60+1) = 1/61
        # 時間減衰は直近なので ≒ 1.0
        fts = [self._make_fts_result(99, days_ago=0.01)]  # ほぼ現在
        result = reciprocal_rank_fusion(fts, [], k=60, halflife_days=30)
        expected_rrf = 1.0 / (60 + 1)
        # decay ≒ 1.0 なので score ≒ expected_rrf
        assert abs(result[0]["score"] - expected_rrf) < 0.001

    def test_multiple_ids_no_duplicate(self) -> None:
        """複数のユニークな ID がある場合、結果に重複がないこと。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        fts = [self._make_fts_result(i) for i in range(1, 4)]
        vec = [self._make_vec_result(i) for i in range(3, 6)]
        result = reciprocal_rank_fusion(fts, vec)

        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids))  # 重複なし

    def test_halflife_days_parameter(self) -> None:
        """halflife_days パラメータが decay に影響すること。"""
        from fuga_memory.search.fusion import reciprocal_rank_fusion

        # 10日前のコンテンツ
        fts = [self._make_fts_result(1, days_ago=10)]

        result_long = reciprocal_rank_fusion(fts, [], halflife_days=30)
        result_short = reciprocal_rank_fusion(fts, [], halflife_days=5)

        # halflife_days が短いほど decay が大きく効いてスコアが低い
        assert result_short[0]["score"] < result_long[0]["score"]
