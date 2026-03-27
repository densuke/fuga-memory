"""daemon/server.py のユニットテスト。"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

from fuga_memory.config import Config


def _find_free_port() -> int:
    """OS に空きポートを割り当ててもらう。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_config(port: int, idle_timeout: int = 600, tmp_db: Path | None = None) -> Config:
    db = tmp_db or Path("/tmp/test_daemon.db")
    return Config(daemon_port=port, daemon_idle_timeout=idle_timeout, db_path=db)


def _make_server(
    port: int,
    idle_timeout: int = 600,
    tmp_db: Path | None = None,
    watchdog_interval: float = 10.0,
) -> object:
    from fuga_memory.daemon.server import DaemonServer

    config = _make_config(port, idle_timeout=idle_timeout, tmp_db=tmp_db)
    return DaemonServer(config, watchdog_interval=watchdog_interval)


@pytest.fixture
def free_port() -> int:
    return _find_free_port()


@pytest.fixture
def running_server(free_port: int, tmp_path: Path) -> None:
    """デーモンサーバーを別スレッドで起動し、テスト終了後に停止する。"""
    server = _make_server(free_port, tmp_db=tmp_path / "test.db")

    t = threading.Thread(target=server.start, daemon=True)
    t.start()

    # 起動待ち（最大5秒）
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{free_port}/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.05)
    else:
        pytest.fail("サーバーが起動しませんでした")

    yield free_port

    # シャットダウン
    import contextlib

    with contextlib.suppress(Exception):
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{free_port}/shutdown",
                method="POST",
            ),
            timeout=2,
        )
    t.join(timeout=3)


class TestHealthEndpoint:
    def test_health_returns_200(self, running_server: int) -> None:
        port = running_server
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            assert resp.status == 200

    def test_health_returns_app_name(self, running_server: int) -> None:
        port = running_server
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            body: dict[str, object] = json.loads(resp.read())
            assert body.get("app") == "fuga-memory"

    def test_health_returns_pending_count(self, running_server: int) -> None:
        port = running_server
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            body: dict[str, object] = json.loads(resp.read())
            assert "pending" in body
            assert isinstance(body["pending"], int)


class TestSaveEndpoint:
    def _post_save(self, port: int, payload: dict[str, str]) -> int:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/save",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status

    def test_save_returns_202(self, running_server: int, tmp_path: Path) -> None:
        port = running_server
        with patch("fuga_memory.daemon.server._do_save_task"):
            status = self._post_save(
                port, {"content": "テスト", "session_id": "s1", "source": "manual"}
            )
        assert status == 202

    def test_save_missing_field_returns_400(self, running_server: int) -> None:
        port = running_server
        import urllib.error

        data = json.dumps({"content": "テスト"}).encode()  # session_id, source が欠落
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/save",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=2)
        assert exc_info.value.code == 400

    def test_save_invalid_json_returns_400(self, running_server: int) -> None:
        port = running_server
        import urllib.error

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/save",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=2)
        assert exc_info.value.code == 400

    def test_concurrent_saves_all_accepted(self, running_server: int) -> None:
        """複数スレッドから同時に POST しても全て 202 を返す。"""
        port = running_server
        results: list[int] = []
        errors: list[Exception] = []

        def post_save() -> None:
            try:
                with patch("fuga_memory.daemon.server._do_save_task"):
                    status = self._post_save(
                        port,
                        {"content": "並行テスト", "session_id": "s1", "source": "manual"},
                    )
                results.append(status)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=post_save) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"エラーが発生: {errors}"
        assert all(s == 202 for s in results)
        assert len(results) == 5


class TestShutdownEndpoint:
    def test_shutdown_returns_200(self, free_port: int, tmp_path: Path) -> None:
        server = _make_server(free_port, tmp_db=tmp_path / "test.db")

        t = threading.Thread(target=server.start, daemon=True)
        t.start()

        # 起動待ち
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{free_port}/health", timeout=1):
                    break
            except Exception:
                time.sleep(0.05)

        req = urllib.request.Request(
            f"http://127.0.0.1:{free_port}/shutdown",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert resp.status == 200

        t.join(timeout=5)
        assert not t.is_alive()


class TestWatchdog:
    def test_watchdog_shuts_down_when_idle(self, tmp_path: Path) -> None:
        """pending=0 かつアイドル超過でサーバーが自動終了する。"""
        port = _find_free_port()
        # idle_timeout=1秒、watchdog_interval=0.1秒でテスト高速化
        server = _make_server(
            port, idle_timeout=1, tmp_db=tmp_path / "test.db", watchdog_interval=0.1
        )

        t = threading.Thread(target=server.start, daemon=True)
        t.start()

        # 起動待ち
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except Exception:
                time.sleep(0.05)

        # 1秒以上待てばwatchdogが終了させるはず（テスト用 watchdog_interval=0.1秒）
        t.join(timeout=10)
        assert not t.is_alive(), "サーバーがアイドルタイムアウト後も動作中"

    def test_watchdog_does_not_shutdown_while_pending(self, tmp_path: Path) -> None:
        """pending > 0 の間はアイドルタイムアウトが来ても終了しない。"""
        port = _find_free_port()
        server = _make_server(
            port, idle_timeout=1, tmp_db=tmp_path / "test.db", watchdog_interval=0.1
        )

        t = threading.Thread(target=server.start, daemon=True)
        t.start()

        # 起動待ち
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                    break
            except Exception:
                time.sleep(0.05)

        # pending を人工的に増やす
        with server._lock:
            server._pending += 1

        time.sleep(2)  # アイドルタイムアウト + マージン

        # まだ動いているはず
        assert t.is_alive(), "pending > 0 なのにサーバーが終了した"

        # クリーンアップ
        with server._lock:
            server._pending -= 1
        urllib.request.urlopen(
            urllib.request.Request(f"http://127.0.0.1:{port}/shutdown", method="POST"),
            timeout=2,
        )
        t.join(timeout=5)
