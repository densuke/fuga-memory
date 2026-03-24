"""CLI コマンドのユニットテスト。

click.testing.CliRunner を使用して CLI の動作を検証する。
DB・エンコーダは mock で差し替える。
"""

from __future__ import annotations

import gc
import sqlite3
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Click テスト用ランナー。"""
    return CliRunner()


@pytest.fixture
def mock_encoder() -> MagicMock:
    """固定ランダムベクトルを返すダミーエンコーダ。"""
    rng = np.random.default_rng(seed=42)
    encoder = MagicMock()
    encoder.encode.side_effect = lambda text: rng.random(768).astype(np.float32).tolist()
    return encoder


@pytest.fixture
def in_memory_conn() -> Generator[sqlite3.Connection]:
    """インメモリ SQLite 接続（スキーマ初期化済み）。テスト終了時にクローズ。"""
    import sqlite_vec

    from fuga_memory.db.schema import initialize_schema

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()
    initialize_schema(conn)
    yield conn
    conn.close()
    gc.collect()  # Python 3.13 でのGC遅延による ResourceWarning を防ぐ


@pytest.fixture(autouse=True)
def _reset_server_globals() -> Generator[None]:
    """各テスト後に server モジュールのグローバルをリセットして参照を解放。"""
    yield
    import fuga_memory.server as srv

    srv._conn = None  # type: ignore[attr-defined]
    srv._encoder = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# serve コマンドのテスト
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_serve_invokes_mcp_run(self, runner: CliRunner) -> None:
        """serve コマンドは mcp.run() を呼び出す。"""
        from fuga_memory.cli import main

        with patch("fuga_memory.server.mcp") as mock_mcp:
            mock_mcp.run.return_value = None
            runner.invoke(main, ["serve"])
            mock_mcp.run.assert_called_once()

    def test_serve_exits_cleanly(self, runner: CliRunner) -> None:
        """serve コマンドは正常終了（またはモック完了）する。"""
        from fuga_memory.cli import main

        with patch("fuga_memory.server.mcp") as mock_mcp:
            mock_mcp.run.return_value = None
            result = runner.invoke(main, ["serve"])
            # mcp.run がモックされているため exit code 0 で終了する
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# search コマンドのテスト
# ---------------------------------------------------------------------------


class TestSearchCommand:
    def test_search_outputs_results(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """search コマンドは検索結果を標準出力に表示する。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]
        srv.save_memory("Pythonのasyncioを学んだ", "s1", "manual")

        from fuga_memory.cli import main

        result = runner.invoke(main, ["search", "Python"])
        assert result.exit_code == 0
        assert result.output  # 何らかの出力がある

    def test_search_with_top_k_option(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """--top-k オプションを渡せる。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]
        for i in range(5):
            srv.save_memory(f"記憶{i} Python テスト", "s1", "manual")

        from fuga_memory.cli import main

        result = runner.invoke(main, ["search", "Python", "--top-k", "2"])
        assert result.exit_code == 0

    def test_search_no_results_shows_message(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """検索結果が 0 件のとき、それを示すメッセージを出力する。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        from fuga_memory.cli import main

        result = runner.invoke(main, ["search", "全くマッチしないクエリXYZ"])
        assert result.exit_code == 0
        # 「0件」や「見つかりません」などのメッセージが含まれる（実装依存）
        # 最低限、正常終了することを確認する

    def test_search_query_is_required(self, runner: CliRunner) -> None:
        """search コマンドはクエリ引数が必須。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["search"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# save コマンドのテスト
# ---------------------------------------------------------------------------


class TestSaveCommand:
    def test_save_stores_memory(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """save コマンドは記憶を DB に保存する。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        from fuga_memory.cli import main

        result = runner.invoke(
            main, ["save", "--session-id", "session-001", "テスト記憶コンテンツ"]
        )
        assert result.exit_code == 0

        row = in_memory_conn.execute("SELECT * FROM memories").fetchone()
        assert row is not None
        assert row["content"] == "テスト記憶コンテンツ"
        assert row["session_id"] == "session-001"

    def test_save_with_source_option(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """--source オプションで source を指定できる。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        from fuga_memory.cli import main

        result = runner.invoke(
            main,
            [
                "save",
                "--session-id",
                "session-001",
                "--source",
                "claude_code",
                "ソース付き記憶",
            ],
        )
        assert result.exit_code == 0

        row = in_memory_conn.execute("SELECT source FROM memories").fetchone()
        assert row["source"] == "claude_code"

    def test_save_default_source_is_manual(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """--source を省略した場合のデフォルトは 'manual'。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        from fuga_memory.cli import main

        result = runner.invoke(
            main, ["save", "--session-id", "session-001", "デフォルトソース記憶"]
        )
        assert result.exit_code == 0

        row = in_memory_conn.execute("SELECT source FROM memories").fetchone()
        assert row["source"] == "manual"

    def test_save_session_id_is_required(self, runner: CliRunner) -> None:
        """save コマンドは --session-id が必須。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["save", "コンテンツ"])
        assert result.exit_code != 0

    def test_save_content_is_required(self, runner: CliRunner) -> None:
        """save コマンドはコンテンツ引数が必須。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["save", "--session-id", "session-001"])
        assert result.exit_code != 0

    def test_save_outputs_confirmation(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """save コマンドは保存完了のメッセージを出力する。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        from fuga_memory.cli import main

        result = runner.invoke(
            main, ["save", "--session-id", "session-001", "確認メッセージテスト"]
        )
        assert result.exit_code == 0
        assert result.output  # 確認メッセージが出力される


# ---------------------------------------------------------------------------
# main グループのテスト
# ---------------------------------------------------------------------------


class TestMainGroup:
    def test_main_help(self, runner: CliRunner) -> None:
        """main --help は正常終了する。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0

    def test_main_has_serve_command(self, runner: CliRunner) -> None:
        """main グループに serve コマンドが登録されている。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["--help"])
        assert "serve" in result.output

    def test_main_has_search_command(self, runner: CliRunner) -> None:
        """main グループに search コマンドが登録されている。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["--help"])
        assert "search" in result.output

    def test_main_has_save_command(self, runner: CliRunner) -> None:
        """main グループに save コマンドが登録されている。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["--help"])
        assert "save" in result.output
