"""fuga_memory/exceptions.py のユニットテスト。"""

from __future__ import annotations

import pytest

from fuga_memory.exceptions import FugaMemoryError, ModelLoadError


class TestFugaMemoryError:
    def test_is_exception(self) -> None:
        """FugaMemoryError は Exception のサブクラスである。"""
        assert issubclass(FugaMemoryError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """FugaMemoryError を raise して except で補足できる。"""
        with pytest.raises(FugaMemoryError):
            raise FugaMemoryError("テストエラー")

    def test_message_is_preserved(self) -> None:
        """エラーメッセージが保持される。"""
        msg = "something went wrong"
        exc = FugaMemoryError(msg)
        assert str(exc) == msg

    def test_can_be_caught_as_exception(self) -> None:
        """Exception としても補足できる。"""
        assert issubclass(FugaMemoryError, Exception)
        exc = FugaMemoryError("base catch")
        assert isinstance(exc, Exception)


class TestModelLoadError:
    def test_is_fuga_memory_error(self) -> None:
        """ModelLoadError は FugaMemoryError のサブクラスである。"""
        assert issubclass(ModelLoadError, FugaMemoryError)

    def test_is_exception(self) -> None:
        """ModelLoadError は Exception のサブクラスである。"""
        assert issubclass(ModelLoadError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """ModelLoadError を raise して except で補足できる。"""
        with pytest.raises(ModelLoadError):
            raise ModelLoadError("モデルロードに失敗しました")

    def test_message_is_preserved(self) -> None:
        """エラーメッセージが保持される。"""
        msg = "model not found"
        exc = ModelLoadError(msg)
        assert str(exc) == msg

    def test_can_be_caught_as_fuga_memory_error(self) -> None:
        """FugaMemoryError としても補足できる。"""
        with pytest.raises(FugaMemoryError):
            raise ModelLoadError("caught as parent")

    def test_can_be_caught_as_exception(self) -> None:
        """Exception としても補足できる。"""
        assert issubclass(ModelLoadError, Exception)
        exc = ModelLoadError("caught as base")
        assert isinstance(exc, Exception)
