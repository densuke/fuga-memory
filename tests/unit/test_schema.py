"""schema.py のユニットテスト。"""

from __future__ import annotations

import sqlite3

from fuga_memory.db.schema import initialize_schema


class TestInitializeSchema:
    def test_memories_table_created(self, initialized_db: sqlite3.Connection) -> None:
        cur = initialized_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        assert cur.fetchone() is not None

    def test_memories_fts_table_created(self, initialized_db: sqlite3.Connection) -> None:
        cur = initialized_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
        )
        assert cur.fetchone() is not None

    def test_memories_vec_table_created(self, initialized_db: sqlite3.Connection) -> None:
        cur = initialized_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_vec'"
        )
        assert cur.fetchone() is not None

    def test_memories_columns(self, initialized_db: sqlite3.Connection) -> None:
        cur = initialized_db.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cur.fetchall()}
        assert columns >= {"id", "content", "session_id", "source", "created_at"}

    def test_initialize_is_idempotent(self, initialized_db: sqlite3.Connection) -> None:
        # 2回呼んでもエラーにならない
        initialize_schema(initialized_db)
        cur = initialized_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        assert cur.fetchone() is not None

    def test_wal_mode_enabled(self, initialized_db: sqlite3.Connection) -> None:
        cur = initialized_db.execute("PRAGMA journal_mode")
        mode = cur.fetchone()[0]
        assert mode == "wal"
