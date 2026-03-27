"""デーモン HTTP クライアント + 自動起動ロジック。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from fuga_memory.config import Config
from fuga_memory.daemon._process import spawn_daemon_process

_POLL_INTERVAL = 0.1  # 秒
_POLL_TIMEOUT = 10.0  # 秒: 起動確認タイムアウト


def _health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def _save_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/save"


def _is_daemon_healthy(port: int) -> bool:
    """デーモンが /health に応答し、fuga-memory であるか確認する。"""
    try:
        with urllib.request.urlopen(_health_url(port), timeout=1) as resp:
            if resp.status != 200:
                return False
            body: dict[str, object] = json.loads(resp.read())
            return body.get("app") == "fuga-memory"
    except Exception:
        return False


def _wait_for_health(port: int, timeout: float = _POLL_TIMEOUT) -> None:
    """GET /health が fuga-memory デーモンから 200 を返すまでポーリングする。

    Args:
        port: デーモンのポート番号。
        timeout: タイムアウト秒数。

    Raises:
        TimeoutError: timeout 秒以内にデーモンが応答しなかった場合。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_daemon_healthy(port):
            return
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(f"デーモンが {timeout}s 以内に起動しませんでした (port={port})")


def ensure_daemon_running(config: Config) -> None:
    """デーモンが起動していなければ自動起動する。

    すでに起動中なら何もしない。未起動なら spawn してヘルスチェックが通るまで待機する。

    Args:
        config: 設定オブジェクト（daemon_port を参照する）。

    Raises:
        TimeoutError: デーモンが _POLL_TIMEOUT 秒以内に起動しなかった場合。
    """
    port = config.daemon_port
    if _is_daemon_healthy(port):
        return

    spawn_daemon_process(port)
    _wait_for_health(port)


def send_save_request(
    content: str,
    session_id: str,
    source: str,
    config: Config,
) -> None:
    """デーモンに POST /save を送り、即座に返る。

    デーモンが未起動であれば ensure_daemon_running() を呼んで起動する。

    Args:
        content: 保存する記憶の内容。
        session_id: セッション識別子。
        source: 記憶のソース（例: "claude_code", "manual"）。
        config: 設定オブジェクト（daemon_port を参照する）。

    Raises:
        TimeoutError: デーモンの起動待ちがタイムアウトした場合。
        RuntimeError: デーモンが 202 以外のステータスを返した場合。
    """
    ensure_daemon_running(config)

    payload = json.dumps({"content": content, "session_id": session_id, "source": source}).encode()

    req = urllib.request.Request(
        _save_url(config.daemon_port),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 202:
                raise RuntimeError(f"デーモンが予期しないステータスを返しました: {resp.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"デーモンへの save リクエストが失敗しました: HTTP {exc.code}") from exc
