"""SQLite接続管理（WALモード、sqlite-vec拡張ロード）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec


def get_connection(db_path: Path) -> sqlite3.Connection:
    """WALモードとsqlite-vec拡張を有効化したSQLite接続を返す。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _configure(conn)
    return conn


def _configure(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
