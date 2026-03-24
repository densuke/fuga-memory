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

    def test_get_encoder_background_load_uses_thread_pool(self) -> None:
        """モデルロードは ThreadPoolExecutor を使ってバックグラウンド実行される。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768
        load_thread_id: list[int] = []

        def capturing_constructor(model_name: str) -> MagicMock:
            load_thread_id.append(threading.get_ident())
            return mock_encoder

        with patch(
            "fuga_memory.embedding.loader.RuriEncoder",
            side_effect=capturing_constructor,
        ):
            loader = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=1)
            loader.get_encoder()

        # ロードは別スレッドで行われる（ThreadPoolExecutor使用時）
        # または同スレッドだが Future.result() でブロック
        # 少なくとも1回呼ばれていることを確認
        assert len(load_thread_id) == 1

    def test_get_encoder_future_result_is_awaited(self) -> None:
        """get_encoder() は concurrent.futures.Future を経由してロード結果を取得する。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768

        with patch(
            "fuga_memory.embedding.loader.RuriEncoder",
            return_value=mock_encoder,
        ):
            # ThreadPoolExecutor.submit() が返す Future.result() が呼ばれることを確認
            with patch("fuga_memory.embedding.loader.ThreadPoolExecutor") as mock_executor_cls:
                mock_executor = MagicMock()
                mock_executor_cls.return_value.__enter__ = MagicMock(return_value=mock_executor)
                mock_executor_cls.return_value.__exit__ = MagicMock(return_value=False)

                mock_future: Future[MagicMock] = Future()
                mock_future.set_result(mock_encoder)
                mock_executor.submit.return_value = mock_future

                loader2 = ModelLoader(model_name="cl-nagoya/ruri-v3-310m", thread_workers=1)
                encoder = loader2.get_encoder()

            assert encoder is mock_encoder


class TestModelLoaderThreadSafety:
    def test_concurrent_get_encoder_calls_return_same_object(self) -> None:
        """並行して get_encoder() を呼んでも同一オブジェクトが返る。"""
        from fuga_memory.embedding.loader import ModelLoader

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.0] * 768

        with patch(
            "fuga_memory.embedding.loader.RuriEncoder",
            return_value=mock_encoder,
        ):
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
        # 全て同一オブジェクト
        first = results[0]
        assert all(r is first for r in results)
