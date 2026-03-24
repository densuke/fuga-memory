"""DBスキーマ定義と初期化。"""

from __future__ import annotations

import sqlite3


def initialize_schema(conn: sqlite3.Connection) -> None:
    """memoriesテーブル、FTS5仮想テーブル、vec仮想テーブルを作成する。"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            content    TEXT    NOT NULL,
            session_id TEXT    NOT NULL,
            source     TEXT    NOT NULL DEFAULT 'manual',
            created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            content='memories',
            content_rowid='id',
            tokenize='trigram'
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
            id        INTEGER PRIMARY KEY,
            embedding float[768]
        );
    """)
    conn.commit()
