"""search/decay.py のユニットテスト。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fuga_memory.search.decay import time_decay


class TestTimeDecay:
    """time_decay 関数のテスト。"""

    def test_very_old_content_has_low_score(self) -> None:
        """古いコンテンツは低スコアになること。"""
        # 1000日前のコンテンツ
        old_date = datetime.now(tz=UTC) - timedelta(days=1000)
        created_at = old_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        score = time_decay(created_at)
        assert score < 0.01

    def test_recent_content_has_high_score(self) -> None:
        """直近のコンテンツは高スコアになること。"""
        # 1日前のコンテンツ
        recent_date = datetime.now(tz=UTC) - timedelta(days=1)
        created_at = recent_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        score = time_decay(created_at)
        assert score > 0.97

    def test_halflife_at_halflife_days(self) -> None:
        """半減期の日数経過後はスコアが約0.5になること。"""
        halflife_days = 30
        target_date = datetime.now(tz=UTC) - timedelta(days=halflife_days)
        created_at = target_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        score = time_decay(created_at, halflife_days=halflife_days)
        # 0.5 に近いこと（秒単位の誤差を許容）
        assert abs(score - 0.5) < 0.01

    def test_custom_halflife_affects_score(self) -> None:
        """halflife_days を短くすると同じ経過日数でもスコアが下がること。"""
        target_date = datetime.now(tz=UTC) - timedelta(days=10)
        created_at = target_date.strftime("%Y-%m-%dT%H:%M:%SZ")

        score_long_halflife = time_decay(created_at, halflife_days=30)
        score_short_halflife = time_decay(created_at, halflife_days=5)
        assert score_short_halflife < score_long_halflife

    def test_future_date_returns_one(self) -> None:
        """未来の created_at は 1.0 を返すこと。"""
        future_date = datetime.now(tz=UTC) + timedelta(days=10)
        created_at = future_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        score = time_decay(created_at)
        assert score == 1.0

    def test_return_type_is_float(self) -> None:
        """戻り値が float 型であること。"""
        now_str = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        score = time_decay(now_str)
        assert isinstance(score, float)

    def test_score_is_between_zero_and_one(self) -> None:
        """スコアが 0.0 以上 1.0 以下であること。"""
        # 過去
        old = (datetime.now(tz=UTC) - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")
        # 現在に近い
        near = (datetime.now(tz=UTC) - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

        for created_at in [old, near]:
            score = time_decay(created_at)
            assert 0.0 <= score <= 1.0

    def test_zero_days_age_returns_one(self) -> None:
        """作成直後（経過0秒）は 1.0 に近いスコアを返すこと。"""
        # 現在時刻（わずかに過去）
        now_str = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        score = time_decay(now_str)
        # 経過時間が1秒未満なので 1.0 に極めて近い
        assert score > 0.999

    def test_zero_halflife_raises_value_error(self) -> None:
        """halflife_days=0 は ValueError を発生させること。"""
        import pytest

        now_str = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with pytest.raises(ValueError, match="halflife_days"):
            time_decay(now_str, halflife_days=0)

    def test_negative_halflife_raises_value_error(self) -> None:
        """halflife_days が負の場合も ValueError を発生させること。"""
        import pytest

        now_str = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with pytest.raises(ValueError, match="halflife_days"):
            time_decay(now_str, halflife_days=-1)
