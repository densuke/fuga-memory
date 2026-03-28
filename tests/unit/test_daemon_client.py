"""daemon/client.py のユニットテスト。"""

from __future__ import annotations

import json
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from fuga_memory.config import Config
from fuga_memory.daemon.client import (
    _is_daemon_healthy,
    _wait_for_health,
    ensure_daemon_running,
    send_save_request,
)


def _make_mock_response(status: int, body: dict | None = None) -> MagicMock:
    """urllib.request.urlopen の戻り値モックを作成する。"""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body or {}).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestIsDaemonHealthy:
    def test_returns_true_when_app_is_fuga_memory(self) -> None:
        resp = _make_mock_response(200, {"app": "fuga-memory", "pending": 0})
        with patch("urllib.request.urlopen", return_value=resp):
            assert _is_daemon_healthy(18520) is True

    def test_returns_false_when_status_not_200(self) -> None:
        resp = _make_mock_response(503, {})
        with patch("urllib.request.urlopen", return_value=resp):
            assert _is_daemon_healthy(18520) is False

    def test_returns_false_when_app_key_missing(self) -> None:
        resp = _make_mock_response(200, {"status": "ok"})
        with patch("urllib.request.urlopen", return_value=resp):
            assert _is_daemon_healthy(18520) is False

    def test_returns_false_on_connection_error(self) -> None:
        import urllib.error

        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")
        ):
            assert _is_daemon_healthy(18520) is False

    def test_returns_false_on_timeout(self) -> None:
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            assert _is_daemon_healthy(18520) is False


class TestWaitForHealth:
    def test_returns_when_daemon_is_healthy(self) -> None:
        resp = _make_mock_response(200, {"app": "fuga-memory"})
        with patch("urllib.request.urlopen", return_value=resp):
            _wait_for_health(18520, timeout=5.0)  # 例外なし

    def test_raises_timeout_error_when_daemon_not_responding(self) -> None:
        import urllib.error

        with (
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")),
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 0.1, 0.2, 11.0]),
            pytest.raises(TimeoutError, match="デーモンが"),
        ):
            _wait_for_health(18520, timeout=10.0)

    def test_retries_until_healthy(self) -> None:
        """最初は失敗し、2回目に成功するケース。"""
        import urllib.error

        healthy_resp = _make_mock_response(200, {"app": "fuga-memory"})
        call_count = 0

        def side_effect(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise urllib.error.URLError("not ready")
            return healthy_resp

        with (
            patch("urllib.request.urlopen", side_effect=side_effect),
            patch("time.sleep"),
        ):
            _wait_for_health(18520, timeout=10.0)
        assert call_count == 2


class TestEnsureDaemonRunning:
    def test_does_nothing_when_already_healthy(self) -> None:
        """すでに起動中なら spawn を呼ばない。"""
        resp = _make_mock_response(200, {"app": "fuga-memory"})
        config = Config(daemon_port=18520)

        with (
            patch("urllib.request.urlopen", return_value=resp),
            patch("fuga_memory.daemon.client.spawn_daemon_process") as mock_spawn,
        ):
            ensure_daemon_running(config)
            mock_spawn.assert_not_called()

    def test_spawns_when_not_running(self) -> None:
        """未起動なら spawn を呼ぶ。"""
        import urllib.error

        healthy_resp = _make_mock_response(200, {"app": "fuga-memory"})
        call_count = 0

        def side_effect(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # 最初の _is_daemon_healthy で失敗、spawn 後の _wait_for_health で成功
            if call_count <= 1:
                raise urllib.error.URLError("not running")
            return healthy_resp

        config = Config(daemon_port=18520)

        with (
            patch("urllib.request.urlopen", side_effect=side_effect),
            patch("fuga_memory.daemon.client.spawn_daemon_process") as mock_spawn,
            patch("time.sleep"),
        ):
            ensure_daemon_running(config)
            mock_spawn.assert_called_once_with(18520)

    def test_uses_config_port(self) -> None:
        """config.daemon_port のポートを使ってヘルスチェックする。"""
        import urllib.error

        healthy_resp = _make_mock_response(200, {"app": "fuga-memory"})
        call_count = 0

        def side_effect(*args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise urllib.error.URLError("not running")
            return healthy_resp

        config = Config(daemon_port=19999)

        with (
            patch("urllib.request.urlopen", side_effect=side_effect) as mock_open,
            patch("fuga_memory.daemon.client.spawn_daemon_process"),
            patch("time.sleep"),
        ):
            ensure_daemon_running(config)
            # ヘルスチェックURLに 19999 が含まれる
            first_url = mock_open.call_args_list[0][0][0]
            assert "19999" in str(first_url)


class TestSendSaveRequest:
    def _healthy_config(self, port: int = 18520) -> Config:
        return Config(daemon_port=port)

    def test_posts_to_save_endpoint(self) -> None:
        """/save エンドポイントに POST する。"""
        healthy_resp = _make_mock_response(200, {"app": "fuga-memory"})
        save_resp = _make_mock_response(202)
        responses = [healthy_resp, save_resp]

        with patch("urllib.request.urlopen", side_effect=responses):
            send_save_request("内容", "session-1", "claude_code", self._healthy_config())

    def test_raises_on_non_202(self) -> None:
        """202 以外のレスポンスで RuntimeError を送出する。"""
        healthy_resp = _make_mock_response(200, {"app": "fuga-memory"})
        error_resp = _make_mock_response(500)
        responses = [healthy_resp, error_resp]

        with (
            patch("urllib.request.urlopen", side_effect=responses),
            pytest.raises(RuntimeError, match="予期しないステータス"),
        ):
            send_save_request("内容", "session-1", "manual", self._healthy_config())

    def test_payload_contains_all_fields(self) -> None:
        """リクエストボディに content/session_id/source が含まれる。"""
        healthy_resp = _make_mock_response(200, {"app": "fuga-memory"})
        save_resp = _make_mock_response(202)

        captured_request: list[urllib.request.Request] = []  # type: ignore[type-arg]

        def capture(*args: object, **kwargs: object) -> MagicMock:
            if args:
                captured_request.append(args[0])  # type: ignore[arg-type]
            # 1回目はhealthcheck、2回目はsave
            return healthy_resp if len(captured_request) <= 1 else save_resp

        with patch("urllib.request.urlopen", side_effect=capture):
            send_save_request("テスト記憶", "sess-001", "manual", self._healthy_config())

        assert len(captured_request) == 2
        save_req = captured_request[1]
        payload = json.loads(save_req.data)  # type: ignore[arg-type]
        assert payload["content"] == "テスト記憶"
        assert payload["session_id"] == "sess-001"
        assert payload["source"] == "manual"

    def test_raises_runtime_error_on_http_error(self) -> None:
        """HTTP エラー時に RuntimeError を送出する。"""
        import urllib.error

        healthy_resp = _make_mock_response(200, {"app": "fuga-memory"})
        call_count = 0

        def side_effect(*_args: object, **_kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return healthy_resp
            raise urllib.error.HTTPError(
                url="http://127.0.0.1:18520/save",
                code=500,
                msg="Internal Server Error",
                hdrs=MagicMock(),  # type: ignore[arg-type]
                fp=None,
            )

        with (
            patch("urllib.request.urlopen", side_effect=side_effect),
            pytest.raises(RuntimeError, match="save リクエストが失敗"),
        ):
            send_save_request("内容", "sess", "manual", self._healthy_config())
