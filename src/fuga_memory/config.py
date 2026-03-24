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
from dataclasses import dataclass, field
from pathlib import Path


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


def _apply_toml(config: Config, path: Path) -> Config:
    """TOML ファイルの値を Config に適用して新しい Config を返す。"""
    values = _parse_toml_file(path)

    db_path = config.db_path
    model_name = config.model_name
    thread_workers = config.thread_workers
    rrf_k = config.rrf_k
    decay_halflife_days = config.decay_halflife_days
    default_top_k = config.default_top_k

    if "db_path" in values:
        db_path = Path(str(values["db_path"])).expanduser()
    if "model_name" in values:
        model_name = str(values["model_name"])
    if "thread_workers" in values:
        thread_workers = int(str(values["thread_workers"]))
    if "rrf_k" in values:
        rrf_k = int(str(values["rrf_k"]))
    if "decay_halflife_days" in values:
        decay_halflife_days = int(str(values["decay_halflife_days"]))
    if "default_top_k" in values:
        default_top_k = int(str(values["default_top_k"]))

    return Config(
        db_path=db_path,
        model_name=model_name,
        embedding_dim=config.embedding_dim,
        thread_workers=thread_workers,
        rrf_k=rrf_k,
        decay_halflife_days=decay_halflife_days,
        default_top_k=default_top_k,
    )


def _apply_env(config: Config) -> Config:
    """環境変数の値を Config に適用して新しい Config を返す。"""
    db_path = config.db_path
    model_name = config.model_name
    thread_workers = config.thread_workers
    rrf_k = config.rrf_k
    decay_halflife_days = config.decay_halflife_days
    default_top_k = config.default_top_k

    if db_path_str := os.environ.get("FUGA_MEMORY_DB_PATH"):
        db_path = Path(db_path_str)

    if model_name_env := os.environ.get("FUGA_MEMORY_MODEL_NAME"):
        model_name = model_name_env

    if workers_str := os.environ.get("FUGA_MEMORY_THREAD_WORKERS"):
        try:
            thread_workers = int(workers_str)
        except ValueError as exc:
            raise ValueError(
                f"FUGA_MEMORY_THREAD_WORKERS に無効な値が設定されています: {workers_str!r}"
                " (整数を指定してください)"
            ) from exc

    if rrf_k_str := os.environ.get("FUGA_MEMORY_RRF_K"):
        try:
            rrf_k = int(rrf_k_str)
        except ValueError as exc:
            raise ValueError(
                f"FUGA_MEMORY_RRF_K に無効な値が設定されています: {rrf_k_str!r}"
                " (整数を指定してください)"
            ) from exc

    if halflife_str := os.environ.get("FUGA_MEMORY_DECAY_HALFLIFE_DAYS"):
        try:
            decay_halflife_days = int(halflife_str)
        except ValueError as exc:
            raise ValueError(
                f"FUGA_MEMORY_DECAY_HALFLIFE_DAYS に無効な値が設定されています: {halflife_str!r}"
                " (整数を指定してください)"
            ) from exc

    if top_k_str := os.environ.get("FUGA_MEMORY_DEFAULT_TOP_K"):
        try:
            default_top_k = int(top_k_str)
        except ValueError as exc:
            raise ValueError(
                f"FUGA_MEMORY_DEFAULT_TOP_K に無効な値が設定されています: {top_k_str!r}"
                " (整数を指定してください)"
            ) from exc

    return Config(
        db_path=db_path,
        model_name=model_name,
        embedding_dim=config.embedding_dim,
        thread_workers=thread_workers,
        rrf_k=rrf_k,
        decay_halflife_days=decay_halflife_days,
        default_top_k=default_top_k,
    )


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
            if path.exists():
                config = _apply_toml(config, path)
                break

        return _apply_env(config)

    @classmethod
    def from_env(cls) -> Config:
        """後方互換エイリアス。load() を呼ぶ。"""
        return cls.load()
