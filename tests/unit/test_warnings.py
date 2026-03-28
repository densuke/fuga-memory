"""fuga_memory.warnings モジュールのユニットテスト。"""

from __future__ import annotations

import warnings

import pytest


class TestIsDebugMode:
    def test_returns_false_when_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FUGA_MEMORY_DEBUG", raising=False)
        from fuga_memory.warnings import is_debug_mode

        assert is_debug_mode() is False

    def test_returns_true_for_value_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DEBUG", "1")
        from fuga_memory.warnings import is_debug_mode

        assert is_debug_mode() is True

    def test_returns_true_for_value_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DEBUG", "true")
        from fuga_memory.warnings import is_debug_mode

        assert is_debug_mode() is True

    def test_returns_true_for_value_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DEBUG", "yes")
        from fuga_memory.warnings import is_debug_mode

        assert is_debug_mode() is True

    def test_returns_true_for_uppercase_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DEBUG", "TRUE")
        from fuga_memory.warnings import is_debug_mode

        assert is_debug_mode() is True

    def test_returns_false_for_value_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DEBUG", "0")
        from fuga_memory.warnings import is_debug_mode

        assert is_debug_mode() is False

    def test_returns_false_for_value_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DEBUG", "false")
        from fuga_memory.warnings import is_debug_mode

        assert is_debug_mode() is False

    def test_returns_false_for_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FUGA_MEMORY_DEBUG", "")
        from fuga_memory.warnings import is_debug_mode

        assert is_debug_mode() is False


class TestSuppressWarnings:
    def test_suppress_warnings_does_nothing_in_debug_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """デバッグモードでは suppress_warnings() は警告フィルタを変更しない。"""
        monkeypatch.setenv("FUGA_MEMORY_DEBUG", "1")
        from fuga_memory.warnings import suppress_warnings

        before = warnings.filters[:]
        suppress_warnings()
        assert warnings.filters == before

    def test_suppress_warnings_adds_filters_in_normal_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """通常モードでは suppress_warnings() が警告フィルタを追加する。"""
        monkeypatch.delenv("FUGA_MEMORY_DEBUG", raising=False)
        from fuga_memory.warnings import suppress_warnings

        before_count = len(warnings.filters)
        suppress_warnings()
        assert len(warnings.filters) > before_count

    def test_suppress_warnings_hides_deprecation_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """suppress_warnings() 呼び出し後は DeprecationWarning が出ない。"""
        monkeypatch.delenv("FUGA_MEMORY_DEBUG", raising=False)
        from fuga_memory.warnings import suppress_warnings

        suppress_warnings()
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("test deprecation", DeprecationWarning, stacklevel=1)
        # suppress_warnings が ignore フィルタを追加しているので caught は空
        assert len(caught) == 0

    def test_suppress_warnings_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """複数回呼び出しても問題が起きない（エラーにならない）。"""
        monkeypatch.delenv("FUGA_MEMORY_DEBUG", raising=False)
        from fuga_memory.warnings import suppress_warnings

        suppress_warnings()
        suppress_warnings()  # 2回目も例外が出ないこと

    def test_suppress_warnings_raises_transformers_log_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """suppress_warnings() は各ライブラリの logging を ERROR 以上に引き上げる。"""
        import logging

        monkeypatch.delenv("FUGA_MEMORY_DEBUG", raising=False)
        from fuga_memory.warnings import suppress_warnings

        suppress_warnings()
        assert logging.getLogger("transformers").level >= logging.ERROR
        assert logging.getLogger("optimum").level >= logging.ERROR
        assert logging.getLogger("sentence_transformers").level >= logging.ERROR

    def test_suppress_warnings_sets_tokenizers_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """suppress_warnings() は TOKENIZERS_PARALLELISM を false に設定する。"""
        monkeypatch.delenv("FUGA_MEMORY_DEBUG", raising=False)
        monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
        from fuga_memory.warnings import suppress_warnings

        suppress_warnings()
        import os

        assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
