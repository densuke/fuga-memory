"""アプリケーション設定。

読み込み優先順位（低 → 高）:
  1. デフォルト値
  2. 設定ファイル（config_file_paths() の先頭に存在するもの1つ）
  3. 環境変数（FUGA_MEMORY_* プレフィックス）

設定ファイルの探索順（先に見つかったものを使用）:
  macOS: ~/Library/Application Support/fuga-memory/config.toml
  XDG:   $XDG_CONFIG_HOME/fuga-memory/config.toml
         （未設定時は ~/.config/fuga-memory/config.toml）
  共通:  ~/.fuga-memory.toml
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def _default_db_path() -> Path:
    return Path.home() / ".local" / "share" / "fuga-memory" / "memories.db"


def _default_thread_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, cpu // 2)


def config_file_paths() -> list[Path]:
    """設定ファイルの候補パスを優先順に返す。"""
    paths: list[Path] = []

    if sys.platform == "darwin":
        paths.append(
            Path.home() / "Library" / "Application Support" / "fuga-memory" / "config.toml"
        )

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    xdg_base = Path(xdg_config_home) if xdg_config_home else (Path.home() / ".config")
    paths.append(xdg_base / "fuga-memory" / "config.toml")

    paths.append(Path.home() / ".fuga-memory.toml")
    return paths


def _parse_toml_file(path: Path) -> dict[str, object]:
    """TOML ファイルを読み込み、[fuga-memory] セクション優先でキーを返す。"""
    with path.open("rb") as f:
        data: dict[str, object] = tomllib.load(f)

    section = data.get("fuga-memory")
    if isinstance(section, dict):
        return section

    return {k: v for k, v in data.items() if not isinstance(v, dict)}


def _parse_int(value: object, key: str, path: Path) -> int:
    """TOML 値を int に変換する。失敗時はキー名とファイルパスを含むエラーを出す。"""
    try:
        return int(str(value))
    except ValueError as exc:
        raise ValueError(
            f"設定ファイル '{path}' の '{key}' に無効な整数値が設定されています: {value!r}"
        ) from exc


def _apply_toml(config: Config, path: Path) -> Config:
    """TOML ファイルの値を Config に適用して新しい Config を返す。"""
    values = _parse_toml_file(path)
    updates: dict[str, Any] = {}

    if "db_path" in values:
        updates["db_path"] = Path(str(values["db_path"])).expanduser()
    if "model_name" in values:
        updates["model_name"] = str(values["model_name"])
    if "thread_workers" in values:
        updates["thread_workers"] = _parse_int(values["thread_workers"], "thread_workers", path)
    if "rrf_k" in values:
        updates["rrf_k"] = _parse_int(values["rrf_k"], "rrf_k", path)
    if "decay_halflife_days" in values:
        updates["decay_halflife_days"] = _parse_int(
            values["decay_halflife_days"], "decay_halflife_days", path
        )
    if "default_top_k" in values:
        updates["default_top_k"] = _parse_int(values["default_top_k"], "default_top_k", path)

    return replace(config, **updates)


def _apply_env(config: Config) -> Config:
    """環境変数の値を Config に適用して新しい Config を返す。"""
    updates: dict[str, Any] = {}

    if db_path_str := os.environ.get("FUGA_MEMORY_DB_PATH"):
        updates["db_path"] = Path(db_path_str).expanduser()

    if model_name_env := os.environ.get("FUGA_MEMORY_MODEL_NAME"):
        updates["model_name"] = model_name_env

    if workers_str := os.environ.get("FUGA_MEMORY_THREAD_WORKERS"):
        try:
            updates["thread_workers"] = int(workers_str)
        except ValueError as exc:
            raise ValueError(
                f"FUGA_MEMORY_THREAD_WORKERS に無効な値が設定されています: {workers_str!r}"
                " (整数を指定してください)"
            ) from exc

    if rrf_k_str := os.environ.get("FUGA_MEMORY_RRF_K"):
        try:
            updates["rrf_k"] = int(rrf_k_str)
        except ValueError as exc:
            raise ValueError(
                f"FUGA_MEMORY_RRF_K に無効な値が設定されています: {rrf_k_str!r}"
                " (整数を指定してください)"
            ) from exc

    if halflife_str := os.environ.get("FUGA_MEMORY_DECAY_HALFLIFE_DAYS"):
        try:
            updates["decay_halflife_days"] = int(halflife_str)
        except ValueError as exc:
            raise ValueError(
                f"FUGA_MEMORY_DECAY_HALFLIFE_DAYS に無効な値が設定されています: {halflife_str!r}"
                " (整数を指定してください)"
            ) from exc

    if top_k_str := os.environ.get("FUGA_MEMORY_DEFAULT_TOP_K"):
        try:
            updates["default_top_k"] = int(top_k_str)
        except ValueError as exc:
            raise ValueError(
                f"FUGA_MEMORY_DEFAULT_TOP_K に無効な値が設定されています: {top_k_str!r}"
                " (整数を指定してください)"
            ) from exc

    return replace(config, **updates)


@dataclass
class Config:
    db_path: Path = field(default_factory=_default_db_path)
    model_name: str = "cl-nagoya/ruri-v3-310m"
    embedding_dim: int = 768
    thread_workers: int = field(default_factory=_default_thread_workers)
    rrf_k: int = 60
    decay_halflife_days: int = 30
    default_top_k: int = 5

    @classmethod
    def load(cls, config_path: Path | None = None) -> Config:
        """設定ファイル→環境変数の優先順で設定を読み込む。

        Args:
            config_path: 明示的に指定する設定ファイルのパス。
                         None の場合は config_file_paths() を順番に探索する。

        Returns:
            デフォルト → 設定ファイル → 環境変数の順で上書きされた Config。
        """
        config = cls()

        search_paths = [config_path] if config_path is not None else config_file_paths()

        for path in search_paths:
            if path.is_file():
                config = _apply_toml(config, path)
                break
            elif path.exists():
                raise IsADirectoryError(f"設定ファイルのパスがディレクトリです: {path}")

        return _apply_env(config)

    @classmethod
    def from_env(cls) -> Config:
        """環境変数のみを適用した Config を返す（設定ファイルは読まない）。

        設定ファイルも含めて読み込みたい場合は load() を使用すること。
        """
        return _apply_env(cls())
