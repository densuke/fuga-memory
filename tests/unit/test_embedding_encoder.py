"""embedding/encoder.py のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


class TestEncoderProtocol:
    def test_encoder_protocol_is_importable(self) -> None:
        """Encoder Protocol が embedding/encoder.py からインポートできる。"""
        from fuga_memory.embedding.encoder import Encoder

        assert Encoder is not None

    def test_encoder_protocol_has_encode_method(self) -> None:
        """Encoder Protocol は encode(text: str) -> list[float] を持つ。"""
        from fuga_memory.embedding.encoder import Encoder

        # Protocol のメソッド確認
        hints = Encoder.__protocol_attrs__ if hasattr(Encoder, "__protocol_attrs__") else None
        # 少なくとも encode が定義されていることを確認
        assert hasattr(Encoder, "encode") or (hints is not None and "encode" in hints)

    def test_custom_class_satisfies_encoder_protocol(self) -> None:
        """encode(text: str) -> list[float] を実装したクラスは Encoder Protocol を満たす。"""

        class MyEncoder:
            def encode(self, text: str) -> list[float]:
                return [0.0] * 768

        # Protocol の structural subtyping チェック
        enc = MyEncoder()
        assert callable(enc.encode)


class TestRuriEncoderInit:
    def test_init_with_default_model_name(self) -> None:
        """デフォルトモデル名 'cl-nagoya/ruri-v3-310m' で初期化できる。"""
        mock_st = MagicMock()

        with patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st):
            from fuga_memory.embedding.encoder import RuriEncoder

            encoder = RuriEncoder()
            assert encoder is not None

    def test_init_with_custom_model_name(self) -> None:
        """カスタムモデル名で初期化できる。"""
        mock_st = MagicMock()

        with patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st):
            from fuga_memory.embedding.encoder import RuriEncoder

            encoder = RuriEncoder(model_name="some/other-model")
            assert encoder is not None

    def test_init_uses_onnx_backend(self) -> None:
        """SentenceTransformer は ONNX バックエンドを使って初期化される。"""
        mock_st = MagicMock()

        with patch(
            "fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st
        ) as mock_cls:
            from fuga_memory.embedding.encoder import RuriEncoder

            RuriEncoder(model_name="cl-nagoya/ruri-v3-310m")

        # ONNX バックエンドが指定されることを確認
        call_kwargs = mock_cls.call_args
        assert call_kwargs is not None
        # backend="onnx" が渡されているか確認
        args, kwargs = call_kwargs
        backend_value = kwargs.get("backend") or (args[1] if len(args) > 1 else None)
        assert backend_value == "onnx"

    def test_init_stores_model_name(self) -> None:
        """model_name が保持される。"""
        mock_st = MagicMock()

        with patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st):
            from fuga_memory.embedding.encoder import RuriEncoder

            encoder = RuriEncoder(model_name="cl-nagoya/ruri-v3-310m")
            assert encoder.model_name == "cl-nagoya/ruri-v3-310m"


class TestRuriEncoderEncode:
    def _make_encoder(self, mock_st: Any) -> Any:
        """モック済みのRuriEncoderを生成するヘルパー。"""
        with patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st):
            from fuga_memory.embedding.encoder import RuriEncoder

            return RuriEncoder()

    def test_encode_returns_list_of_float(self) -> None:
        """encode() は list[float] を返す。"""
        import numpy as np

        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768, dtype=np.float32)

        encoder = self._make_encoder(mock_st)
        result = encoder.encode("テスト文字列")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_encode_returns_768_dimensions(self) -> None:
        """encode() の戻り値は768次元である。"""
        import numpy as np

        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768, dtype=np.float32)

        encoder = self._make_encoder(mock_st)
        result = encoder.encode("テスト文字列")

        assert len(result) == 768

    def test_encode_adds_document_prefix(self) -> None:
        """encode() はテキストに '検索文書: ' プレフィックスを付与してモデルに渡す。"""
        import numpy as np

        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768, dtype=np.float32)

        encoder = self._make_encoder(mock_st)
        encoder.encode("Pythonの学習について")

        # SentenceTransformer.encode に渡されたテキストを確認
        call_args = mock_st.encode.call_args
        assert call_args is not None
        args, kwargs = call_args
        passed_text = args[0] if args else kwargs.get("sentences")
        assert passed_text == "検索文書: Pythonの学習について"

    def test_encode_empty_string(self) -> None:
        """空文字列でも encode() は動作する。"""
        import numpy as np

        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768, dtype=np.float32)

        encoder = self._make_encoder(mock_st)
        result = encoder.encode("")

        assert isinstance(result, list)
        assert len(result) == 768

    def test_encode_unicode_text(self) -> None:
        """日本語・絵文字を含むテキストでも動作する。"""
        import numpy as np

        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768, dtype=np.float32)

        encoder = self._make_encoder(mock_st)
        result = encoder.encode("日本語テキスト：情報を記憶する🧠")

        assert isinstance(result, list)
        assert len(result) == 768

    def test_encode_values_are_floats(self) -> None:
        """戻り値の各要素が Python float 型である。"""
        import numpy as np

        rng = np.random.default_rng(seed=0)
        mock_st = MagicMock()
        mock_st.encode.return_value = rng.random(768).astype(np.float32)

        encoder = self._make_encoder(mock_st)
        result = encoder.encode("test")

        assert all(isinstance(v, float) for v in result)

    def test_encode_prefix_plus_text_passed_to_model(self) -> None:
        """プレフィックス付きテキストが正確に '検索文書: ' + 元テキスト である。"""
        import numpy as np

        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768, dtype=np.float32)

        encoder = self._make_encoder(mock_st)
        original_text = "some content to embed"
        encoder.encode(original_text)

        call_args = mock_st.encode.call_args
        assert call_args is not None
        args, kwargs = call_args
        passed_text = args[0] if args else kwargs.get("sentences")
        assert passed_text == f"検索文書: {original_text}"


class TestRuriEncoderWithCache:
    def test_init_with_cache_dir_loads_from_cache_when_cached(self, tmp_path: Path) -> None:
        """cache_dir が指定されキャッシュが存在する場合はキャッシュからロードする。"""

        # キャッシュが存在する状態を作る
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").touch()

        mock_st = MagicMock()
        with (
            patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st),
            patch(
                "fuga_memory.embedding.encoder.load_cached_model", return_value=mock_st
            ) as mock_load,
            patch("fuga_memory.embedding.encoder.is_cached", return_value=True),
        ):
            from fuga_memory.embedding.encoder import RuriEncoder

            RuriEncoder(model_name="cl-nagoya/ruri-v3-310m", cache_dir=tmp_path)

        mock_load.assert_called_once_with(tmp_path)

    def test_init_with_cache_dir_exports_when_not_cached(self, tmp_path: Path) -> None:
        """cache_dir が指定されキャッシュがない場合はエクスポートして保存する。"""
        mock_st = MagicMock()
        with (
            patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st),
            patch(
                "fuga_memory.embedding.encoder.export_and_cache", return_value=tmp_path
            ) as mock_export,
            patch("fuga_memory.embedding.encoder.load_cached_model", return_value=mock_st),
            patch("fuga_memory.embedding.encoder.is_cached", return_value=False),
        ):
            from fuga_memory.embedding.encoder import RuriEncoder

            RuriEncoder(model_name="cl-nagoya/ruri-v3-310m", cache_dir=tmp_path)

        mock_export.assert_called_once_with("cl-nagoya/ruri-v3-310m", tmp_path)

    def test_init_without_cache_dir_uses_direct_load(self) -> None:
        """cache_dir=None の場合は従来どおり SentenceTransformer を直接使う。"""
        mock_st = MagicMock()
        with (
            patch(
                "fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st
            ) as mock_cls,
            patch("fuga_memory.embedding.encoder.is_cached") as mock_is_cached,
        ):
            from fuga_memory.embedding.encoder import RuriEncoder

            RuriEncoder(model_name="cl-nagoya/ruri-v3-310m", cache_dir=None)

        # is_cached は呼ばれない（キャッシュパスがないので）
        mock_is_cached.assert_not_called()
        mock_cls.assert_called_once()

    def test_init_with_cache_dir_falls_back_on_export_failure(self, tmp_path: Path) -> None:
        """エクスポート失敗時はフォールバックして直接ロードする。"""
        mock_st = MagicMock()
        with (
            patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st),
            patch(
                "fuga_memory.embedding.encoder.export_and_cache",
                side_effect=RuntimeError("network error"),
            ),
            patch("fuga_memory.embedding.encoder.is_cached", return_value=False),
        ):
            from fuga_memory.embedding.encoder import RuriEncoder

            # フォールバックするため例外は出ない
            encoder = RuriEncoder(model_name="cl-nagoya/ruri-v3-310m", cache_dir=tmp_path)
            assert encoder is not None


class TestRuriEncoderSatisfiesProtocol:
    def test_ruri_encoder_has_encode_method(self) -> None:
        """RuriEncoder は encode() メソッドを持つ。"""
        mock_st = MagicMock()
        with patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st):
            from fuga_memory.embedding.encoder import RuriEncoder

            encoder = RuriEncoder()
            assert hasattr(encoder, "encode")
            assert callable(encoder.encode)

    def test_ruri_encoder_satisfies_encoder_protocol(self) -> None:
        """RuriEncoder は Encoder Protocol を満たす（structural subtyping）。"""
        import numpy as np

        mock_st = MagicMock()
        mock_st.encode.return_value = np.zeros(768, dtype=np.float32)

        with patch("fuga_memory.embedding.encoder.SentenceTransformer", return_value=mock_st):
            from fuga_memory.embedding.encoder import RuriEncoder

            encoder = RuriEncoder()

        # Encoder Protocol の encode シグネチャを満たす
        result = encoder.encode("test")
        assert isinstance(result, list)
