"""embedding/loader.py のユニットテスト。"""

from __future__ import annotations

import threading
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from fuga_memory.exceptions import ModelLoadError


class TestModelLoaderInit:
    def test_init_with_model_name(self) -> None:
        """model_name と thread_workers を受け取って初期化できる。"""
        from fuga_memory.embedding.loader import ModelLoader

        loader = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=1)
        assert loader is not None

    def test_init_default_thread_workers(self) -> None:
        """thread_workers のデフォルト値は 1 である。"""
        from fuga_memory.embedding.loader import ModelLoader

        loader = ModelLoader(model_name="cl-nagoya/ruri-v3-310m")
        # thread_workers デフォルト=1 でインスタンス生成できることを確認
        assert loader is not None

    def test_init_stores_model_name(self) -> None:
        """model_name が保持される。"""
        from fuga_memory.embedding.loader import ModelLoader

        loader = ModelLoader(model_name="test-model", thread_workers=2)
        assert loader.model_name == "test-model"

    def test_init_stores_thread_workers(self) -> None:
        """thread_workers が保持される。"""
        from fuga_memory.embedding.loader import ModelLoader

        loader = ModelLoader(model_name="test-model", thread_workers=4)
        assert loader.thread_workers == 4

    def test_init_raises_on_zero_thread_workers(self) -> None:
        """thread_workers=0 は ValueError を発生させる。"""
        from fuga_memory.embedding.loader import ModelLoader

        with pytest.raises(ValueError, match="thread_workers"):
            ModelLoader(model_name="test-model", thread_workers=0)

    def test_init_raises_on_negative_thread_workers(self) -> None:
        """thread_workers が負の値の場合は ValueError を発生させる。"""
        from fuga_memory.embedding.loader import ModelLoader

        with pytest.raises(ValueError, match="thread_workers"):
            ModelLoader(model_name="test-model", thread_workers=-1)


class TestModelLoaderGetEncoder:
    def test_get_encoder_returns_encoder(self) -> None:
        """get_encoder() はエンコーダオブジェクトを返す。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768

        with patch(
            "fuga_memory.embedding.loader.RuriEncoder",
            return_value=mock_encoder,
        ):
            loader = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=1)
            encoder = loader.get_encoder()

        assert encoder is not None

    def test_get_encoder_caches_result(self) -> None:
        """get_encoder() を2回呼び出すと同一オブジェクトが返る（キャッシュ）。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768

        with patch(
            "fuga_memory.embedding.loader.RuriEncoder",
            return_value=mock_encoder,
        ) as mock_cls:
            loader = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=1)
            enc1 = loader.get_encoder()
            enc2 = loader.get_encoder()

        # RuriEncoder のコンストラクタは一度しか呼ばれない
        assert mock_cls.call_count == 1
        assert enc1 is enc2

    def test_get_encoder_returns_encode_capable_object(self) -> None:
        """get_encoder() の戻り値は encode() メソッドを持つ。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768

        with patch(
            "fuga_memory.embedding.loader.RuriEncoder",
            return_value=mock_encoder,
        ):
            loader = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=1)
            encoder = loader.get_encoder()

        assert hasattr(encoder, "encode")
        assert callable(encoder.encode)

    def test_get_encoder_raises_model_load_error_on_failure(self) -> None:
        """モデルロードに失敗した場合 ModelLoadError が発生する。"""
        from fuga_memory.embedding.loader import ModelLoader

        with patch(
            "fuga_memory.embedding.loader.RuriEncoder",
            side_effect=RuntimeError("ダウンロード失敗"),
        ):
            loader = ModelLoader(model_name="nonexistent-model", thread_workers=1)
            with pytest.raises(ModelLoadError):
                loader.get_encoder()

    def test_get_encoder_retries_after_failure(self) -> None:
        """ロード失敗後に再度 get_encoder() を呼ぶと再試行できる。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768

        # 1回目は失敗、2回目は成功
        with patch(
            "fuga_memory.embedding.loader.RuriEncoder",
            side_effect=[RuntimeError("1回目失敗"), mock_encoder],
        ):
            loader = ModelLoader(model_name="test-model", thread_workers=1)
            with pytest.raises(ModelLoadError):
                loader.get_encoder()
            # 失敗後に再試行でき、成功する
            encoder = loader.get_encoder()
            assert encoder is mock_encoder

    def test_get_encoder_uses_thread_pool_executor(self) -> None:
        """get_encoder() は ThreadPoolExecutor.submit() を使ってモデルをロードする。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768

        with patch("fuga_memory.embedding.loader.ThreadPoolExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor_cls.return_value = mock_executor

            future: Future[MagicMock] = Future()
            future.set_result(mock_encoder)
            mock_executor.submit.return_value = future

            loader = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=2)
            loader.get_encoder()

        # ThreadPoolExecutor が指定のワーカー数で生成され、submit が呼ばれたことを確認
        mock_executor_cls.assert_called_once_with(max_workers=2)
        mock_executor.submit.assert_called_once()
        # shutdown(wait=False) で executor が解放されたことを確認
        mock_executor.shutdown.assert_called_once_with(wait=False)

    def test_get_encoder_future_result_is_awaited(self) -> None:
        """get_encoder() は ThreadPoolExecutor.submit() の Future 経由でロード結果を取得する。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768

        with patch("fuga_memory.embedding.loader.ThreadPoolExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor_cls.return_value = mock_executor

            mock_future: Future[MagicMock] = Future()
            mock_future.set_result(mock_encoder)
            mock_executor.submit.return_value = mock_future

            loader2 = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=1)
            encoder = loader2.get_encoder()

        assert encoder is mock_encoder


class TestModelLoaderThreadSafety:
    def test_concurrent_get_encoder_calls_return_same_object(self) -> None:
        """並行して get_encoder() を呼んでも同一オブジェクトが返る。

        RuriEncoder の呼び出し回数が 1 であることで重複ロードが起きていないことを保証する。
        """
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder_a = MagicMock()
        mock_encoder_b = MagicMock()
        # side_effect で呼ばれるたびに異なるオブジェクトを返す（重複生成の検出に使う）
        mock_ruri_cls = MagicMock(side_effect=[mock_encoder_a, mock_encoder_b])

        with patch("fuga_memory.embedding.loader.RuriEncoder", mock_ruri_cls):
            loader = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=2)
            results: list[object] = []

            def call_get_encoder() -> None:
                enc = loader.get_encoder()
                results.append(enc)

            threads = [threading.Thread(target=call_get_encoder) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(results) == 5
        # RuriEncoder は1度しか生成されない（スレッドセーフの証明）
        assert mock_ruri_cls.call_count == 1
        # 全スレッドが同一オブジェクトを受け取る
        first = results[0]
        assert all(r is first for r in results)
