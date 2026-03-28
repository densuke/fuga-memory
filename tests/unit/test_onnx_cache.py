"""embedding/onnx_cache.py のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetOnnxCacheDir:
    def test_returns_path_under_fuga_memory(self) -> None:
        """キャッシュパスは .local/share/fuga-memory 配下になる。"""
        from fuga_memory.embedding.onnx_cache import get_onnx_cache_dir

        path = get_onnx_cache_dir("cl-nagoya/ruri-v3-310m")
        assert "fuga-memory" in path.as_posix()

    def test_slash_in_model_name_is_sanitized(self) -> None:
        """モデル名の '/' はパスに使えない文字に変換される。"""
        from fuga_memory.embedding.onnx_cache import get_onnx_cache_dir

        path = get_onnx_cache_dir("cl-nagoya/ruri-v3-310m")
        # '/' がそのまま含まれていないこと（ディレクトリが分割されていること）
        assert "/" not in path.name

    def test_different_models_have_different_paths(self) -> None:
        """モデルが異なれば異なるキャッシュパスになる。"""
        from fuga_memory.embedding.onnx_cache import get_onnx_cache_dir

        path_a = get_onnx_cache_dir("org/model-a")
        path_b = get_onnx_cache_dir("org/model-b")
        assert path_a != path_b

    def test_same_model_returns_same_path(self) -> None:
        """同じモデル名は常に同じパスを返す（冪等性）。"""
        from fuga_memory.embedding.onnx_cache import get_onnx_cache_dir

        assert get_onnx_cache_dir("org/model") == get_onnx_cache_dir("org/model")

    def test_returns_path_object(self) -> None:
        """戻り値は Path オブジェクト。"""
        from fuga_memory.embedding.onnx_cache import get_onnx_cache_dir

        assert isinstance(get_onnx_cache_dir("org/model"), Path)


class TestIsCached:
    def test_returns_false_when_dir_does_not_exist(self, tmp_path: Path) -> None:
        """キャッシュディレクトリが存在しない場合は False。"""
        from fuga_memory.embedding.onnx_cache import is_cached

        assert is_cached(tmp_path / "nonexistent") is False

    def test_returns_false_when_onnx_file_missing(self, tmp_path: Path) -> None:
        """ディレクトリはあるが onnx/model.onnx がない場合は False。"""
        from fuga_memory.embedding.onnx_cache import is_cached

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        assert is_cached(cache_dir) is False

    def test_returns_true_when_model_onnx_exists(self, tmp_path: Path) -> None:
        """onnx/model.onnx が存在する場合は True。"""
        from fuga_memory.embedding.onnx_cache import is_cached

        cache_dir = tmp_path / "cache"
        onnx_dir = cache_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model.onnx").touch()
        assert is_cached(cache_dir) is True

    def test_returns_false_when_onnx_dir_exists_but_no_model(self, tmp_path: Path) -> None:
        """onnx/ ディレクトリはあるが model.onnx がない場合は False。"""
        from fuga_memory.embedding.onnx_cache import is_cached

        cache_dir = tmp_path / "cache"
        (cache_dir / "onnx").mkdir(parents=True)
        assert is_cached(cache_dir) is False


class TestExportAndCache:
    def test_calls_sentence_transformer_save(self, tmp_path: Path) -> None:
        """export_and_cache() は SentenceTransformer をロードして save() を呼ぶ。"""
        from fuga_memory.embedding.onnx_cache import export_and_cache

        mock_model = MagicMock()

        with patch(
            "fuga_memory.embedding.onnx_cache.SentenceTransformer",
            return_value=mock_model,
        ) as mock_cls:
            result = export_and_cache("cl-nagoya/ruri-v3-310m", tmp_path)

        mock_cls.assert_called_once_with("cl-nagoya/ruri-v3-310m", backend="onnx")
        mock_model.save.assert_called_once_with(str(tmp_path))
        assert result == tmp_path

    def test_creates_cache_dir_if_not_exists(self, tmp_path: Path) -> None:
        """キャッシュディレクトリが存在しない場合は作成する。"""
        from fuga_memory.embedding.onnx_cache import export_and_cache

        cache_dir = tmp_path / "new_cache" / "nested"
        mock_model = MagicMock()

        with patch("fuga_memory.embedding.onnx_cache.SentenceTransformer", return_value=mock_model):
            export_and_cache("some/model", cache_dir)

        assert cache_dir.exists()

    def test_propagates_exception_on_failure(self, tmp_path: Path) -> None:
        """エクスポート失敗時は例外を伝播する。"""
        from fuga_memory.embedding.onnx_cache import export_and_cache

        with (
            patch(
                "fuga_memory.embedding.onnx_cache.SentenceTransformer",
                side_effect=RuntimeError("download failed"),
            ),
            pytest.raises(RuntimeError, match="download failed"),
        ):
            export_and_cache("some/model", tmp_path)


class TestLoadCachedModel:
    def test_loads_from_cache_dir(self, tmp_path: Path) -> None:
        """load_cached_model() はキャッシュディレクトリから SentenceTransformer をロードする。"""
        from fuga_memory.embedding.onnx_cache import load_cached_model

        mock_model = MagicMock()

        with patch(
            "fuga_memory.embedding.onnx_cache.SentenceTransformer",
            return_value=mock_model,
        ) as mock_cls:
            result = load_cached_model(tmp_path)

        mock_cls.assert_called_once_with(str(tmp_path), backend="onnx")
        assert result is mock_model
