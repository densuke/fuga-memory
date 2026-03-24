"""fuga-memory カスタム例外クラス。"""

from __future__ import annotations


class FugaMemoryError(Exception):
    """fuga-memory の基底例外クラス。"""


class ModelLoadError(FugaMemoryError):
    """埋め込みモデルのロードに失敗した際に発生する例外。"""
