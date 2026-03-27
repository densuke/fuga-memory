"""daemon/_process.py のユニットテスト。"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from fuga_memory.daemon._process import spawn_daemon_process


class TestSpawnDaemonProcess:
    def test_posix_uses_start_new_session(self) -> None:
        """POSIX 環境では start_new_session=True でプロセスを起動する。"""
        with (
            patch("sys.platform", "linux"),
            patch("subprocess.Popen") as mock_popen,
        ):
            spawn_daemon_process(18520)
            mock_popen.assert_called_once()
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is True

    def test_darwin_uses_start_new_session(self) -> None:
        """macOS 環境でも start_new_session=True を使う。"""
        with (
            patch("sys.platform", "darwin"),
            patch("subprocess.Popen") as mock_popen,
        ):
            spawn_daemon_process(18520)
            mock_popen.assert_called_once()
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is True

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows専用テスト")
    def test_windows_uses_detached_process_flag(self) -> None:
        """Windows では DETACHED_PROCESS フラグを使う。（Windows環境でのみ実行）"""
        with patch("subprocess.Popen") as mock_popen:
            spawn_daemon_process(18520)
            mock_popen.assert_called_once()
            _, kwargs = mock_popen.call_args
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            assert kwargs.get("creationflags") == DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    def test_command_includes_server_module(self) -> None:
        """起動コマンドに daemon.server モジュールが含まれる。"""
        with (
            patch("sys.platform", "linux"),
            patch("subprocess.Popen") as mock_popen,
        ):
            spawn_daemon_process(18520)
            args, _ = mock_popen.call_args
            cmd: list[str] = args[0]
            assert "-m" in cmd
            assert "fuga_memory.daemon.server" in cmd

    def test_command_includes_port(self) -> None:
        """起動コマンドに --port と指定ポートが含まれる。"""
        with (
            patch("sys.platform", "linux"),
            patch("subprocess.Popen") as mock_popen,
        ):
            spawn_daemon_process(19000)
            args, _ = mock_popen.call_args
            cmd: list[str] = args[0]
            assert "--port" in cmd
            assert "19000" in cmd

    def test_stdin_stdout_stderr_are_devnull(self) -> None:
        """stdin/stdout/stderr はすべて DEVNULL にリダイレクトされる。"""
        import subprocess

        with (
            patch("sys.platform", "linux"),
            patch("subprocess.Popen") as mock_popen,
        ):
            spawn_daemon_process(18520)
            _, kwargs = mock_popen.call_args
            assert kwargs.get("stdin") == subprocess.DEVNULL
            assert kwargs.get("stdout") == subprocess.DEVNULL
            assert kwargs.get("stderr") == subprocess.DEVNULL

    def test_uses_sys_executable(self) -> None:
        """sys.executable を使って起動する（uv 非依存）。"""
        with (
            patch("sys.platform", "linux"),
            patch("subprocess.Popen") as mock_popen,
        ):
            spawn_daemon_process(18520)
            args, _ = mock_popen.call_args
            cmd: list[str] = args[0]
            assert cmd[0] == sys.executable


class TestWindowsSpawnSimulated:
    """Windows の分岐ロジックを非Windows環境でシミュレートするテスト。"""

    def test_windows_platform_uses_creationflags(self) -> None:
        """sys.platform を win32 にパッチして Windows 分岐をテストする。

        NOTE: Windows暫定実装・未検証 - GitHub Actions Windows VMで実際の動作を確認すること。
        """
        with (
            patch("sys.platform", "win32"),
            patch("subprocess.Popen") as mock_popen,
        ):
            spawn_daemon_process(18520)
            mock_popen.assert_called_once()
            _, kwargs = mock_popen.call_args
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            assert kwargs.get("creationflags") == DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            assert "start_new_session" not in kwargs

    def test_windows_platform_no_start_new_session(self) -> None:
        """Windows では start_new_session を使わない。

        NOTE: Windows暫定実装・未検証
        """
        with (
            patch("sys.platform", "win32"),
            patch("subprocess.Popen") as mock_popen,
        ):
            spawn_daemon_process(18520)
            _, kwargs = mock_popen.call_args
            assert kwargs.get("start_new_session") is None
