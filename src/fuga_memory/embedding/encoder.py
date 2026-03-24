"""テキストをベクトルに変換するエンコーダ。"""

from __future__ import annotations

from typing import Protocol

from sentence_transformers import SentenceTransformer

_DOCUMENT_PREFIX = "検索文書: "
_DEFAULT_MODEL = "cl-nagoya/ruri-v3-310m"


class Encoder(Protocol):
    """テキストをベクトルに変換するプロトコル。"""

    def encode(self, text: str) -> list[float]: ...


class RuriEncoder:
    """ruri-v3 モデルを使って日本語テキストを 768 次元ベクトルに変換するエンコーダ。

    保存時プレフィックス「検索文書: 」を自動付与する。
    sentence-transformers の ONNX バックエンドを使用する。
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(model_name, backend="onnx")

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
