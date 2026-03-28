"""CLI コマンドのユニットテスト。

click.testing.CliRunner を使用して CLI の動作を検証する。
DB・エンコーダは mock で差し替える。
"""

from __future__ import annotations

import gc
import sqlite3
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
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
    srv._config = None  # type: ignore[attr-defined]


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
        assert "記憶が見つかりませんでした。" in result.output

    def test_search_query_is_required(self, runner: CliRunner) -> None:
        """search コマンドはクエリ引数が必須。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["search"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# save コマンドのテスト
# ---------------------------------------------------------------------------


class TestSaveCommand:
    @pytest.fixture(autouse=True)
    def _daemon_unavailable(self) -> Generator[None]:
        """デーモンが利用不可の状態をシミュレート（フォールバックパスをテスト）。"""
        with patch(
            "fuga_memory.cli.send_save_request",
            side_effect=TimeoutError("daemon unavailable in test"),
        ):
            yield

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
        """save コマンドはコンテンツ引数が必須（--stdin / --file なしの場合）。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["save", "--session-id", "session-001"])
        assert result.exit_code != 0

    def test_save_from_stdin(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """--stdin フラグで標準入力からコンテンツを読み込める。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        from fuga_memory.cli import main

        result = runner.invoke(
            main,
            ["save", "--session-id", "session-stdin", "--stdin"],
            input="標準入力からのテスト記憶",
        )
        assert result.exit_code == 0
        row = in_memory_conn.execute("SELECT content FROM memories").fetchone()
        assert row["content"] == "標準入力からのテスト記憶"

    def test_save_from_file(
        self,
        tmp_path: Path,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """--file オプションでファイルからコンテンツを読み込める。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        content_file = tmp_path / "memory.txt"
        content_file.write_text("ファイルからのテスト記憶", encoding="utf-8")

        from fuga_memory.cli import main

        result = runner.invoke(
            main,
            ["save", "--session-id", "session-file", "--file", str(content_file)],
        )
        assert result.exit_code == 0
        row = in_memory_conn.execute("SELECT content FROM memories").fetchone()
        assert row["content"] == "ファイルからのテスト記憶"

    def test_save_stdin_and_arg_conflict(self, runner: CliRunner) -> None:
        """content 引数と --stdin を同時に指定するとエラー。"""
        from fuga_memory.cli import main

        result = runner.invoke(
            main,
            ["save", "--session-id", "s1", "--stdin", "コンテンツ"],
            input="stdin入力",
        )
        assert result.exit_code != 0

    def test_save_stdin_and_file_conflict(
        self,
        tmp_path: Path,
        runner: CliRunner,
    ) -> None:
        """--stdin と --file を同時に指定するとエラー。"""
        from fuga_memory.cli import main

        content_file = tmp_path / "f.txt"
        content_file.write_text("test")
        result = runner.invoke(
            main,
            ["save", "--session-id", "s1", "--stdin", "--file", str(content_file)],
            input="stdin",
        )
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
# save コマンド デーモン連携テスト
# ---------------------------------------------------------------------------


class TestSaveCommandDaemon:
    """save コマンドがデーモン経由で動作することを確認するテスト。"""

    def test_save_calls_send_save_request_via_daemon(self, runner: CliRunner) -> None:
        """save コマンドが send_save_request を呼ぶ（デーモン経由）。"""
        from fuga_memory.cli import main

        with patch("fuga_memory.cli.send_save_request") as mock_send:
            result = runner.invoke(main, ["save", "--session-id", "session-001", "デーモンテスト"])
        assert result.exit_code == 0
        mock_send.assert_called_once()
        call_kwargs = mock_send.call_args
        assert call_kwargs.args[0] == "デーモンテスト"
        assert call_kwargs.args[1] == "session-001"

    def test_save_daemon_outputs_queued_message(self, runner: CliRunner) -> None:
        """デーモン経由の save は「キューに追加」メッセージを出力する。"""
        from fuga_memory.cli import main

        with patch("fuga_memory.cli.send_save_request"):
            result = runner.invoke(
                main, ["save", "--session-id", "session-001", "メッセージテスト"]
            )
        assert result.exit_code == 0
        assert "キュー" in result.output or "バックグラウンド" in result.output

    def test_save_falls_back_to_direct_when_daemon_fails(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """デーモン経由が失敗した場合は直接実行にフォールバックする。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        from fuga_memory.cli import main

        with patch(
            "fuga_memory.cli.send_save_request", side_effect=TimeoutError("daemon unavailable")
        ):
            result = runner.invoke(
                main, ["save", "--session-id", "session-001", "フォールバックテスト"]
            )
        assert result.exit_code == 0
        assert result.output  # フォールバックでも何らかの出力がある

    def test_save_uses_config_daemon_port(self, runner: CliRunner) -> None:
        """save コマンドが Config を渡して send_save_request を呼ぶ。"""
        from fuga_memory.cli import main

        with patch("fuga_memory.cli.send_save_request") as mock_send:
            result = runner.invoke(main, ["save", "--session-id", "session-001", "ポートテスト"])
        assert result.exit_code == 0
        # config 引数が渡されていることを確認
        from fuga_memory.config import Config

        call_args = mock_send.call_args
        assert isinstance(call_args.kwargs.get("config") or call_args.args[-1], Config)


# ---------------------------------------------------------------------------
# delete コマンドのテスト
# ---------------------------------------------------------------------------


class TestDeleteCommand:
    def test_delete_command_calls_repository_delete(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """delete コマンドは指定された ID の記憶を削除する。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]
        result_save = srv.save_memory("削除テスト", "s1", "manual")
        memory_id = result_save["id"]

        from fuga_memory.cli import main

        result = runner.invoke(main, ["delete", str(memory_id)])
        assert result.exit_code == 0
        assert "削除しました" in result.output

        # DBからも消えているか確認
        cur = in_memory_conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (memory_id,))
        assert cur.fetchone()[0] == 0

    def test_delete_command_not_found(
        self,
        runner: CliRunner,
        in_memory_conn: sqlite3.Connection,
        mock_encoder: MagicMock,
    ) -> None:
        """存在しない ID の場合はエラーメッセージを表示し、非ゼロで終了する。"""
        import fuga_memory.server as srv

        srv._conn = in_memory_conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        from fuga_memory.cli import main

        result = runner.invoke(main, ["delete", "999"])
        assert result.exit_code != 0
        assert "見つかりませんでした" in result.output

    def test_delete_id_is_required(self, runner: CliRunner) -> None:
        """delete コマンドは ID 引数が必須。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["delete"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# main グループのテスト
# ---------------------------------------------------------------------------


class TestDebugOption:
    def test_debug_flag_appears_in_help(self, runner: CliRunner) -> None:
        """--debug オプションがヘルプに表示される。"""
        from fuga_memory.cli import main

        result = runner.invoke(main, ["--help"])
        assert "--debug" in result.output

    def test_suppress_warnings_called_without_debug(self, runner: CliRunner) -> None:
        """--debug なしかつ config.debug=False では suppress_warnings() が呼ばれる。"""
        from fuga_memory.cli import main
        from fuga_memory.config import Config

        mock_config = MagicMock(spec=Config)
        mock_config.debug = False

        with (
            patch("fuga_memory.cli.suppress_warnings") as mock_suppress,
            patch("fuga_memory.cli.Config.load", return_value=mock_config),
            patch("fuga_memory.server.mcp") as mock_mcp,
        ):
            mock_mcp.run.return_value = None
            runner.invoke(main, ["serve"])
        mock_suppress.assert_called_once()

    def test_suppress_warnings_not_called_with_debug(self, runner: CliRunner) -> None:
        """--debug ありでは suppress_warnings() が呼ばれない。"""
        from fuga_memory.cli import main
        from fuga_memory.config import Config

        mock_config = MagicMock(spec=Config)
        mock_config.debug = False

        with (
            patch("fuga_memory.cli.suppress_warnings") as mock_suppress,
            patch("fuga_memory.cli.Config.load", return_value=mock_config),
            patch("fuga_memory.server.mcp") as mock_mcp,
        ):
            mock_mcp.run.return_value = None
            runner.invoke(main, ["--debug", "serve"])
        mock_suppress.assert_not_called()

    def test_suppress_warnings_not_called_when_config_debug_true(self, runner: CliRunner) -> None:
        """config.debug=True のとき suppress_warnings() が呼ばれない。"""
        from fuga_memory.cli import main
        from fuga_memory.config import Config

        mock_config = MagicMock(spec=Config)
        mock_config.debug = True

        with (
            patch("fuga_memory.cli.suppress_warnings") as mock_suppress,
            patch("fuga_memory.cli.Config.load", return_value=mock_config),
            patch("fuga_memory.server.mcp") as mock_mcp,
        ):
            mock_mcp.run.return_value = None
            runner.invoke(main, ["serve"])
        mock_suppress.assert_not_called()


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


# ---------------------------------------------------------------------------
# _to_localtime のテスト
# ---------------------------------------------------------------------------


class TestToLocaltime:
    def test_utc_string_is_converted(self) -> None:
        """UTC 文字列がローカルタイムに変換される（タイムゾーン表記が含まれる）。"""
        from fuga_memory.cli import _to_localtime

        result = _to_localtime("2026-03-25T00:12:33Z")
        # ローカル変換後は年月日・時刻・タイムゾーン略称が含まれる
        assert "2026-" in result
        assert len(result) > 19  # UTC 文字列より長い（タイムゾーン略称が付く）

    def test_returns_string(self) -> None:
        """戻り値は文字列。"""
        from fuga_memory.cli import _to_localtime

        assert isinstance(_to_localtime("2026-01-01T00:00:00Z"), str)

    def test_utc_midnight_converts_to_local(self) -> None:
        """UTC 00:00:00 はローカルタイムで 00:00:00 にはならない（UTC+0 以外の環境）。"""

        from fuga_memory.cli import _to_localtime

        result = _to_localtime("2026-03-25T00:00:00Z")
        # UTC+0 でない環境では時刻が変わっている
        local_offset = datetime.now(UTC).astimezone().utcoffset()
        if local_offset is not None and local_offset.total_seconds() != 0:
            assert "00:00:00" not in result
