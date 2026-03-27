"""fuga-memory デーモン HTTP サーバー。

ThreadingHTTPServer をベースにした軽量ローカルサーバー。
save リクエストを受け取ったら即座に 202 を返し、
バックグラウンドスレッドで埋め込みモデル + SQLite 保存を処理する。

起動方法:
    python -m fuga_memory.daemon.server --port <PORT>
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fuga_memory.config import Config

logger = logging.getLogger(__name__)

# watchdog が確認する間隔（秒）。テストでは短くするために公開している。
_WATCHDOG_INTERVAL = 10.0


def _do_save_task(content: str, session_id: str, source: str, config: Config) -> None:
    """バックグラウンドで save_memory を呼び出す。

    この関数はスレッドプールから実行される。例外はログに記録してサイレント処理する。
    """
    try:
        from fuga_memory.db.connection import get_connection
        from fuga_memory.db.repository import MemoryRepository
        from fuga_memory.db.schema import initialize_schema
        from fuga_memory.embedding.loader import ModelLoader

        conn = get_connection(config.db_path)
        initialize_schema(conn)
        loader = ModelLoader(config.model_name, config.thread_workers)
        encoder = loader.get_encoder()
        repo = MemoryRepository(conn, encoder)
        repo.save(content, session_id, source)
        logger.debug("保存完了: session_id=%s source=%s", session_id, source)
    except Exception:
        logger.exception("バックグラウンド保存中にエラーが発生しました")


class DaemonServer:
    """fuga-memory デーモン HTTP サーバー。

    Attributes:
        _pending: 処理中タスク数。
        _lock: _pending の排他制御。
        _last_request_time: 最後にリクエストを受け付けた時刻（monotonic）。
        _shutdown_event: 終了フラグ。
    """

    def __init__(self, config: Config, watchdog_interval: float = _WATCHDOG_INTERVAL) -> None:
        self._config = config
        self._pending = 0
        self._lock = threading.Lock()
        self._last_request_time = time.monotonic()
        self._shutdown_event = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._watchdog_interval = watchdog_interval

        # ログ設定
        log_path = config.db_path.parent / "daemon.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger("fuga_memory").addHandler(handler)

    def start(self) -> None:
        """サーバーを起動してブロックする。シグナルまたは shutdown() で終了する。"""
        server = ThreadingHTTPServer(("127.0.0.1", self._config.daemon_port), self._make_handler())
        self._httpd = server

        watchdog_thread = threading.Thread(
            target=self._watchdog, name="daemon-watchdog", daemon=True
        )
        watchdog_thread.start()

        logger.info("デーモン起動: port=%d", self._config.daemon_port)
        try:
            server.serve_forever()
        finally:
            logger.info("デーモン停止")

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        """DaemonHandler のクラスを生成する（server インスタンスをクロージャで渡す）。"""
        daemon = self

        class DaemonHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health":
                    daemon._handle_health(self)
                else:
                    self._send_error(404, "Not Found")

            def do_POST(self) -> None:
                if self.path == "/save":
                    daemon._handle_save(self)
                elif self.path == "/shutdown":
                    daemon._handle_shutdown(self)
                else:
                    self._send_error(404, "Not Found")

            def _send_error(self, code: int, message: str) -> None:
                body = json.dumps({"error": message}).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                logger.debug(format, *args)

        return DaemonHandler

    def _handle_health(self, handler: BaseHTTPRequestHandler) -> None:
        with self._lock:
            pending = self._pending
        body = json.dumps({"app": "fuga-memory", "pending": pending}).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _handle_save(self, handler: BaseHTTPRequestHandler) -> None:
        try:
            length = int(handler.headers.get("Content-Length", "0"))
            raw = handler.rfile.read(length)
            payload: dict[str, object] = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            body = json.dumps({"error": str(exc)}).encode()
            handler.send_response(400)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return

        content = payload.get("content")
        session_id = payload.get("session_id")
        source = payload.get("source")

        if (
            not isinstance(content, str)
            or not isinstance(session_id, str)
            or not isinstance(source, str)
        ):
            body = json.dumps({"error": "content, session_id, source は必須です"}).encode()
            handler.send_response(400)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return

        # 即座に 202 を返す
        handler.send_response(202)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

        # バックグラウンドでタスクを実行
        self._submit_save(content, session_id, source)

        # 最終リクエスト時刻を更新
        with self._lock:
            self._last_request_time = time.monotonic()

    def _handle_shutdown(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        self._shutdown_event.set()
        if self._httpd is not None:
            threading.Thread(target=self._httpd.shutdown, daemon=True).start()

    def _submit_save(self, content: str, session_id: str, source: str) -> None:
        """pending を +1 してスレッドを起動する。完了後に -1 する。"""
        with self._lock:
            self._pending += 1

        def run() -> None:
            try:
                _do_save_task(content, session_id, source, self._config)
            finally:
                with self._lock:
                    self._pending -= 1

        t = threading.Thread(target=run, daemon=True)
        t.start()

    def _watchdog(self) -> None:
        """定期的にアイドルタイムアウトを確認し、条件を満たせばシャットダウンする。"""
        while not self._shutdown_event.wait(timeout=self._watchdog_interval):
            with self._lock:
                pending = self._pending
                elapsed = time.monotonic() - self._last_request_time
            if pending == 0 and elapsed >= self._config.daemon_idle_timeout:
                logger.info(
                    "アイドルタイムアウト（%ds）に達しました。シャットダウンします。",
                    self._config.daemon_idle_timeout,
                )
                self._shutdown_event.set()
                if self._httpd is not None:
                    self._httpd.shutdown()
                break


def main() -> None:
    """コマンドライン: python -m fuga_memory.daemon.server --port <PORT>"""
    parser = argparse.ArgumentParser(description="fuga-memory デーモンサーバー")
    parser.add_argument("--port", type=int, required=True, help="リッスンポート番号")
    args = parser.parse_args()

    config = Config.load()
    # コマンドライン引数のポートで上書き
    from dataclasses import replace

    config = replace(config, daemon_port=args.port)

    server = DaemonServer(config)
    server.start()


if __name__ == "__main__":
    main()
