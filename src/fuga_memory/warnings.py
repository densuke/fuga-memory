"""ライブラリ由来の警告を抑制するユーティリティ。

通常実行時に PyTorch / ONNX / transformers 等が出す DeprecationWarning 等を
まとめて抑制する。FUGA_MEMORY_DEBUG=1 のときは何もしない。
"""

from __future__ import annotations

import logging
import os
import warnings

_TRUE_VALUES = frozenset({"1", "true", "yes"})


def is_debug_mode() -> bool:
    """デバッグモードかどうかを判定する。

    環境変数 FUGA_MEMORY_DEBUG が "1" / "true" / "yes"（大小文字不問）の場合に True を返す。

    Returns:
        デバッグモードであれば True。
    """
    return os.environ.get("FUGA_MEMORY_DEBUG", "").lower() in _TRUE_VALUES


def suppress_warnings() -> None:
    """ライブラリ由来の警告を一括抑制する。

    PyTorch / ONNX エクスポート / transformers / tokenizers 等が出す
    DeprecationWarning, FutureWarning, UserWarning, TracerWarning を無視する。

    FUGA_MEMORY_DEBUG=1 のときは何もしない（警告はそのまま出る）。
    """
    if is_debug_mode():
        return

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # torch.jit.TracerWarning は torch がインストールされている場合のみ
    try:
        import torch.jit

        tracer_warning = getattr(torch.jit, "TracerWarning", None)
        if tracer_warning is not None:
            warnings.filterwarnings("ignore", category=tracer_warning)
    except ImportError:
        pass

    # tokenizers の並列処理警告を環境変数で抑制
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ONNXRuntime ネイティブログを ERROR 以上に抑制（3=Error, 4=Fatal）
    os.environ.setdefault("ONNXRUNTIME_LOG_SEVERITY_LEVEL", "3")
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        ort.set_default_logger_severity(3)
    except (ImportError, Exception):
        pass

    # transformers / optimum / sentence_transformers が出す WARNING ログを ERROR 以上に抑制
    for logger_name in ("transformers", "optimum", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
