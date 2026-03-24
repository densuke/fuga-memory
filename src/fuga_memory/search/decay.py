"""時間減衰スコアの計算。"""

from __future__ import annotations

from datetime import UTC, datetime

_SECONDS_PER_DAY = 86400.0


def time_decay(created_at: str, halflife_days: int = 30) -> float:
    """ISO8601形式の created_at から時間減衰係数を計算する。

    Args:
        created_at: ISO8601形式の日時文字列（例: "2024-01-01T00:00:00Z"）
        halflife_days: 半減期の日数（デフォルト: 30日）

    Returns:
        decay = 0.5 ** (age_days / halflife_days)
        created_at が未来の場合は 1.0 を返す。
    """
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    now = datetime.now(tz=UTC)
    age_seconds = (now - dt).total_seconds()

    if halflife_days <= 0:
        raise ValueError(f"halflife_days は 1 以上である必要があります: {halflife_days}")

    if age_seconds <= 0:
        return 1.0

    age_days = age_seconds / _SECONDS_PER_DAY
    return float(0.5 ** (age_days / halflife_days))
