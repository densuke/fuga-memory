"""デーモン統合テスト。

実際のデーモンサーバーを起動し、cli.py の save コマンドから
send_save_request 経由でデータが SQLite に書き込まれることを
エンドツーエンドで検証する。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from fuga_memory.config import Config


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def daemon_config(tmp_path: Path) -> Config:
    """テスト用デーモン設定。空きポートと一時 DB を使う。"""
    port = _find_free_port()
    db_path = tmp_path / "integration.db"
    return Config(
        daemon_port=port,
        daemon_idle_timeout=600,
        db_path=db_path,
    )


@pytest.fixture
def running_daemon(daemon_config: Config) -> Config:
    """デーモンサーバーを別スレッドで起動し、テスト終了後に停止する。"""
    from fuga_memory.daemon.server import DaemonServer

    server = DaemonServer(daemon_config, watchdog_interval=10.0)
    t = threading.Thread(target=server.start, daemon=True)
    t.start()

    # 起動待ち（最大5秒）
    port = daemon_config.daemon_port
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.05)
    else:
        pytest.fail("デーモンが起動しませんでした")

    yield daemon_config

    # シャットダウン
    import contextlib

    with contextlib.suppress(Exception):
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/shutdown",
                method="POST",
            ),
            timeout=2,
        )
    t.join(timeout=3)


class TestDaemonHealthIntegration:
    def test_health_endpoint_responds(self, running_daemon: Config) -> None:
        """起動中のデーモンは /health に 200 を返す。"""
        port = running_daemon.daemon_port
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            assert resp.status == 200
            body: dict[str, object] = json.loads(resp.read())
            assert body["app"] == "fuga-memory"
            assert isinstance(body["pending"], int)


class TestDaemonSaveIntegration:
    def test_save_request_accepted_with_202(self, running_daemon: Config) -> None:
        """POST /save は即座に 202 を返す。"""
        port = running_daemon.daemon_port
        payload = json.dumps(
            {"content": "統合テスト記憶", "session_id": "it-s1", "source": "integration"}
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/save",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with (
            patch("fuga_memory.daemon.server._do_save_task"),
            urllib.request.urlopen(req, timeout=2) as resp,
        ):
            assert resp.status == 202

    def test_save_actually_persists_to_db(self, running_daemon: Config) -> None:
        """POST /save 後にバックグラウンド処理が完了し、DB にレコードが書き込まれる。

        _do_save_task をモックすることで ModelLoader のロードを排除し、
        HTTP→バックグラウンドタスク→DB 書き込みのフローを検証する。
        モック内で実際の DB 書き込みを行うため、ストレージ層の動作も確認できる。
        """
        import numpy as np

        from fuga_memory.daemon.client import send_save_request

        rng = np.random.default_rng(seed=0)

        class _MockEncoder:
            def encode(self, text: str) -> list[float]:
                return rng.random(768).astype(np.float32).tolist()

        def _fake_save_task(content: str, session_id: str, source: str, config: Config) -> None:
            """ModelLoader を使わずに直接 DB に書き込む代替実装。"""
            from fuga_memory.db.connection import get_connection
            from fuga_memory.db.repository import MemoryRepository
            from fuga_memory.db.schema import initialize_schema

            conn = get_connection(config.db_path)
            initialize_schema(conn)
            repo = MemoryRepository(conn, _MockEncoder())
            repo.save(content, session_id, source)

        db_path = running_daemon.db_path
        row = None

        with patch("fuga_memory.daemon.server._do_save_task", side_effect=_fake_save_task):
            send_save_request(
                content="DB書き込み統合テスト",
                session_id="it-session-db",
                source="integration",
                config=running_daemon,
            )

            # バックグラウンド処理の完了を待つ（最大10秒）
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    conn = sqlite3.connect(str(db_path))
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT content, session_id, source FROM memories WHERE session_id = ?",
                        ("it-session-db",),
                    ).fetchone()
                    conn.close()
                    if row is not None:
                        break
                except Exception:
                    pass
                time.sleep(0.2)

        assert row is not None, "DB にレコードが書き込まれませんでした"
        assert row["content"] == "DB書き込み統合テスト"
        assert row["session_id"] == "it-session-db"
        assert row["source"] == "integration"

    def test_send_save_request_raises_on_daemon_failure(self, daemon_config: Config) -> None:
        """デーモンが起動していない場合、send_save_request はフォールバックを示す例外を送出する。"""
        from fuga_memory.daemon.client import send_save_request

        # spawn を無効化してデーモンが起動しないようにする
        with (
            patch("fuga_memory.daemon.client.spawn_daemon_process"),
            pytest.raises((TimeoutError, RuntimeError)),
        ):
            send_save_request(
                content="未起動テスト",
                session_id="it-no-daemon",
                source="integration",
                config=daemon_config,
            )


class TestDaemonEnsureRunning:
    def test_ensure_daemon_running_starts_daemon(
        self, tmp_path: Path, daemon_config: Config
    ) -> None:
        """ensure_daemon_running は未起動のデーモンを spawn して待機する。"""
        from fuga_memory.daemon.client import _is_daemon_healthy, ensure_daemon_running
        from fuga_memory.daemon.server import DaemonServer

        port = daemon_config.daemon_port

        # spawn_daemon_process をモックして直接スレッドでサーバーを起動する
        def _fake_spawn(p: int) -> None:
            srv = DaemonServer(daemon_config, watchdog_interval=10.0)
            t = threading.Thread(target=srv.start, daemon=True)
            t.start()

        with patch("fuga_memory.daemon.client.spawn_daemon_process", side_effect=_fake_spawn):
            ensure_daemon_running(daemon_config)

        assert _is_daemon_healthy(port), "デーモンが起動しませんでした"

        # クリーンアップ
        import contextlib

        with contextlib.suppress(Exception):
            urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:{port}/shutdown", method="POST"),
                timeout=2,
            )

    def test_ensure_daemon_running_noop_if_already_healthy(self, running_daemon: Config) -> None:
        """すでに起動中のデーモンがある場合、spawn_daemon_process を呼ばない。"""
        from fuga_memory.daemon.client import ensure_daemon_running

        with patch("fuga_memory.daemon.client.spawn_daemon_process") as mock_spawn:
            ensure_daemon_running(running_daemon)

        mock_spawn.assert_not_called()


class TestCliDaemonFallbackIntegration:
    def test_cli_save_uses_daemon_when_available(
        self, running_daemon: Config, tmp_path: Path
    ) -> None:
        """CLI save コマンドがデーモン経由で保存リクエストを送信する。"""
        from click.testing import CliRunner

        from fuga_memory.cli import main

        runner = CliRunner()

        # Config.load() が running_daemon の設定を返すようにパッチ
        with patch("fuga_memory.cli.Config") as mock_cfg_cls:
            mock_cfg_cls.load.return_value = running_daemon
            with patch("fuga_memory.cli.send_save_request") as mock_send:
                result = runner.invoke(
                    main,
                    ["save", "--session-id", "cli-it-session", "CLI統合テスト記憶"],
                )

        assert result.exit_code == 0
        mock_send.assert_called_once()
        args = mock_send.call_args.args
        assert args[0] == "CLI統合テスト記憶"
        assert args[1] == "cli-it-session"

    def test_cli_save_falls_back_when_daemon_unavailable(
        self,
        running_daemon: Config,
        tmp_path: Path,
    ) -> None:
        """デーモン経由が失敗した場合、CLI save は直接保存にフォールバックする。"""
        import sqlite3 as _sqlite3

        import numpy as np

        # インメモリ DB でフォールバックを受け取る
        import sqlite_vec
        from click.testing import CliRunner

        import fuga_memory.server as srv
        from fuga_memory.cli import main
        from fuga_memory.db.schema import initialize_schema

        conn = _sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = _sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        initialize_schema(conn)

        rng = np.random.default_rng(seed=0)
        mock_encoder = type(
            "Enc", (), {"encode": lambda self, t: rng.random(768).astype(np.float32).tolist()}
        )()

        srv._conn = conn  # type: ignore[attr-defined]
        srv._encoder = mock_encoder  # type: ignore[attr-defined]

        runner = CliRunner()

        with patch(
            "fuga_memory.cli.send_save_request",
            side_effect=TimeoutError("daemon down"),
        ):
            result = runner.invoke(
                main,
                ["save", "--session-id", "cli-fallback", "フォールバック統合テスト"],
            )

        assert result.exit_code == 0
        assert "保存しました" in result.output

        row = conn.execute(
            "SELECT content FROM memories WHERE session_id = ?", ("cli-fallback",)
        ).fetchone()
        assert row is not None
        assert row["content"] == "フォールバック統合テスト"

        # クリーンアップ
        srv._conn = None  # type: ignore[attr-defined]
        srv._encoder = None  # type: ignore[attr-defined]
        conn.close()
