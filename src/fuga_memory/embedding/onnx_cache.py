"""ONNXモデルのローカルキャッシュ管理。

初回起動時に SentenceTransformer でエクスポートしたONNXモデルをディスクに保存し、
2回目以降はキャッシュから高速にロードする。
"""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer

_ONNX_MODEL_FILE = "onnx/model.onnx"
_DEFAULT_CACHE_BASE = Path.home() / ".local" / "share" / "fuga-memory" / "onnx_cache"


def get_onnx_cache_dir(model_name: str, base_dir: Path = _DEFAULT_CACHE_BASE) -> Path:
    """モデル名に対応するONNXキャッシュディレクトリパスを返す。

    モデル名中の '/' を '--' に変換してディレクトリ名として使用する。
    ディレクトリの作成は行わない（呼び出し元が判断する）。

    Args:
        model_name: HuggingFace モデルID（例: "cl-nagoya/ruri-v3-310m"）
        base_dir: キャッシュのベースディレクトリ

    Returns:
        キャッシュディレクトリパス。
    """
    sanitized = model_name.replace("/", "--")
    return base_dir / sanitized


def is_cached(cache_dir: Path) -> bool:
    """キャッシュ済みONNXモデルが存在するか判定する。

    Args:
        cache_dir: キャッシュディレクトリパス

    Returns:
        onnx/model.onnx が存在すれば True。
    """
    return (cache_dir / _ONNX_MODEL_FILE).is_file()


def export_and_cache(model_name: str, cache_dir: Path) -> Path:
    """SentenceTransformerモデルをONNX形式でエクスポートしてキャッシュに保存する。

    Args:
        model_name: HuggingFace モデルID
        cache_dir: 保存先ディレクトリ（存在しない場合は作成する）

    Returns:
        保存先ディレクトリパス（= cache_dir）

    Raises:
        Exception: モデルのダウンロードやエクスポートに失敗した場合
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, backend="onnx")
    model.save(str(cache_dir))
    return cache_dir


def load_cached_model(cache_dir: Path) -> SentenceTransformer:
    """キャッシュ済みONNXモデルから SentenceTransformer をロードする。

    Args:
        cache_dir: ONNXキャッシュディレクトリ（export_and_cache で作成されたもの）

    Returns:
        ONNX バックエンドで初期化済みの SentenceTransformer。
    """
    return SentenceTransformer(str(cache_dir), backend="onnx")
