"""テキストをベクトルに変換するエンコーダ。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from sentence_transformers import SentenceTransformer

from fuga_memory.embedding.onnx_cache import export_and_cache, is_cached, load_cached_model

_DOCUMENT_PREFIX = "検索文書: "
_DEFAULT_MODEL = "cl-nagoya/ruri-v3-310m"

logger = logging.getLogger(__name__)


class Encoder(Protocol):
    """テキストをベクトルに変換するプロトコル。"""

    def encode(self, text: str) -> list[float]: ...


class RuriEncoder:
    """ruri-v3 モデルを使って日本語テキストを 768 次元ベクトルに変換するエンコーダ。

    保存時プレフィックス「検索文書: 」を自動付与する。
    sentence-transformers の ONNX バックエンドを使用する。

    cache_dir が指定された場合:
      - キャッシュが存在すれば高速にローカルロード
      - キャッシュが存在しなければエクスポートして保存してからロード
      - エクスポート失敗時は直接ロードにフォールバック
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        self._model_name = model_name

        if cache_dir is not None:
            self._model = self._load_with_cache(model_name, cache_dir)
        else:
            self._model = SentenceTransformer(model_name, backend="onnx")

    def _load_with_cache(self, model_name: str, cache_dir: Path) -> SentenceTransformer:
        """キャッシュを利用してモデルをロードする。"""
        if is_cached(cache_dir):
            return load_cached_model(cache_dir)

        try:
            # export_and_cache がエクスポート済みモデルを返すので再ロード不要
            return export_and_cache(model_name, cache_dir)
        except Exception as exc:
            logger.warning("ONNXキャッシュ作成失敗。直接ロードにフォールバック: %s", exc)
            return SentenceTransformer(model_name, backend="onnx")

    @property
    def model_name(self) -> str:
        """使用するモデル名。"""
        return self._model_name

    def encode(self, text: str) -> list[float]:
        """テキストを 768 次元の float リストに変換する。

        「検索文書: 」プレフィックスを自動付与してからモデルに渡す。

        Args:
            text: 変換対象のテキスト。

        Returns:
            768 次元の float リスト。
        """
        prefixed = f"{_DOCUMENT_PREFIX}{text}"
        embedding = self._model.encode(prefixed)
        return [float(v) for v in embedding]
