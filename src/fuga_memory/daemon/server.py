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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fuga_memory.config import Config
from fuga_memory.embedding.encoder import Encoder

logger = logging.getLogger(__name__)

# watchdog が確認する間隔（秒）。テストでは短くするために公開している。
_WATCHDOG_INTERVAL = 10.0

# /save エンドポイントが受け付けるコンテンツの最大バイト数（cli.py の _MAX_INPUT_BYTES と統一）
_MAX_CONTENT_BYTES = 1_048_576  # 1MB

# content フィールドの最大文字数（server.save_memory の _MAX_CONTENT_LENGTH と統一）
_MAX_CONTENT_LENGTH = 100_000  # 100,000文字

# デーモンプロセス全体でエンコーダを1回だけロードしてキャッシュする。
# (モデル名, スレッド数) をキーとし、初回リクエスト時に遅延ロードする。
_encoder_cache: dict[tuple[str, int], Encoder] = {}
_encoder_lock = threading.Lock()


def _get_or_load_encoder(model_name: str, thread_workers: int) -> Encoder:
    """エンコーダをキャッシュから返す。未ロードなら ModelLoader でロードしてキャッシュする。

    スレッドセーフ（double-checked locking）。
    キャッシュキーは (model_name, thread_workers) のペアで、
    thread_workers が異なる設定でも別エンコーダが生成される。
    """
    key = (model_name, thread_workers)
    if key not in _encoder_cache:
        with _encoder_lock:
            if key not in _encoder_cache:
                from fuga_memory.embedding.loader import ModelLoader

                loader = ModelLoader(model_name, thread_workers)
                _encoder_cache[key] = loader.get_encoder()
                logger.info("埋め込みモデルのロード完了: %s", model_name)
    return _encoder_cache[key]


def _do_save_task(content: str, session_id: str, source: str, config: Config) -> None:
    """バックグラウンドで save_memory を呼び出す。

    この関数はスレッドプールから実行される。例外はログに記録してサイレント処理する。
    エンコーダはプロセス内でキャッシュされるため、2回目以降はロード不要。
    """
    try:
        from fuga_memory.db.connection import get_connection
        from fuga_memory.db.repository import MemoryRepository
        from fuga_memory.db.schema import initialize_schema

        conn = get_connection(config.db_path)
        try:
            initialize_schema(conn, config.embedding_dim)
            encoder = _get_or_load_encoder(config.model_name, config.thread_workers)
            repo = MemoryRepository(conn, encoder, config.embedding_dim)
            repo.save(content, session_id, source)
            logger.debug("保存完了: session_id=%s source=%s", session_id, source)
        finally:
            conn.close()
    except Exception:
        logger.exception("バックグラウンド保存中にエラーが発生しました")


def _write_json_response(
    handler: BaseHTTPRequestHandler, status: int, body: dict[str, object]
) -> None:
    """JSON レスポンスを書き出すヘルパー。"""
    raw = json.dumps(body).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


class DaemonServer:
    """fuga-memory デーモン HTTP サーバー。

    Attributes:
        _pending: 処理中タスク数。
        _lock: _pending の排他制御。
        _last_request_time: 最後にリクエストを受け付けた時刻（monotonic）。
        _shutdown_event: 終了フラグ。
        _executor: バックグラウンド保存用スレッドプール（上限付き）。
    """

    # バックグラウンドワーカーの最大スレッド数
    _SAVE_POOL_MAX_WORKERS = 4

    def __init__(self, config: Config, watchdog_interval: float = _WATCHDOG_INTERVAL) -> None:
        self._config = config
        self._pending = 0
        self._lock = threading.Lock()
        self._last_request_time = time.monotonic()
        self._shutdown_event = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._watchdog_interval = watchdog_interval
        self._executor = ThreadPoolExecutor(max_workers=self._SAVE_POOL_MAX_WORKERS)

        # ログ設定
        # 同一プロセスで複数回 DaemonServer が生成されてもハンドラを重複追加しない
        log_path = config.db_path.parent / "daemon.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fuga_logger = logging.getLogger("fuga_memory")
        for existing in fuga_logger.handlers:
            if isinstance(existing, logging.FileHandler) and getattr(
                existing, "baseFilename", None
            ) == str(log_path):
                break
        else:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            fuga_logger.addHandler(file_handler)

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
            server.server_close()
            logger.info("デーモン停止")

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        """DaemonHandler のクラスを生成する（server インスタンスをクロージャで渡す）。"""
        # BaseHTTPRequestHandler のサブクラス内では self がハンドラインスタンスを指すため、
        # DaemonServer インスタンスを `daemon` という別名でクロージャに渡す。
        daemon = self

        class DaemonHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/health":
                    daemon._handle_health(self)
                else:
                    _write_json_response(self, 404, {"error": "Not Found"})

            def do_POST(self) -> None:
                if self.path == "/save":
                    daemon._handle_save(self)
                elif self.path == "/shutdown":
                    daemon._handle_shutdown(self)
                else:
                    _write_json_response(self, 404, {"error": "Not Found"})

            def log_message(self, format: str, *args: object) -> None:
                logger.debug(format, *args)

        return DaemonHandler

    def _handle_health(self, handler: BaseHTTPRequestHandler) -> None:
        with self._lock:
            pending = self._pending
        _write_json_response(handler, 200, {"app": "fuga-memory", "pending": pending})

    def _handle_save(self, handler: BaseHTTPRequestHandler) -> None:
        payload = self._parse_save_request(handler)
        if payload is None:
            return

        content = payload.get("content")
        session_id = payload.get("session_id")
        source = payload.get("source")

        if (
            not isinstance(content, str)
            or not isinstance(session_id, str)
            or not isinstance(source, str)
        ):
            _write_json_response(handler, 400, {"error": "content, session_id, source は必須です"})
            return

        # server.save_memory と同等の入力バリデーション
        if not content:
            _write_json_response(handler, 400, {"error": "content は空文字列にできません"})
            return
        if len(content) > _MAX_CONTENT_LENGTH:
            _write_json_response(
                handler,
                400,
                {"error": f"content が最大サイズを超えています: {len(content)} > {_MAX_CONTENT_LENGTH}"},
            )
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

    def _parse_save_request(self, handler: BaseHTTPRequestHandler) -> dict[str, object] | None:
        """リクエストボディを読み取って JSON をパースする。

        サイズ超過または不正 JSON の場合はエラーレスポンスを送り None を返す。
        """
        try:
            length = int(handler.headers.get("Content-Length", "0"))
        except ValueError as exc:
            _write_json_response(handler, 400, {"error": str(exc)})
            return None

        if length < 0:
            _write_json_response(handler, 400, {"error": "Content-Length が不正です"})
            return None

        if length > _MAX_CONTENT_BYTES:
            _write_json_response(
                handler,
                413,
                {"error": f"リクエストサイズが上限（{_MAX_CONTENT_BYTES:,} バイト）を超えています"},
            )
            return None

        try:
            raw = handler.rfile.read(length)
            result: dict[str, object] = json.loads(raw)
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            _write_json_response(handler, 400, {"error": str(exc)})
            return None

    def _handle_shutdown(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        self._shutdown_event.set()
        if self._httpd is not None:
            threading.Thread(target=self._httpd.shutdown, daemon=True).start()

    def _submit_save(self, content: str, session_id: str, source: str) -> None:
        """pending を +1 してスレッドプールにタスクを投入する。完了後に -1 する。"""
        with self._lock:
            self._pending += 1

        def run() -> None:
            try:
                _do_save_task(content, session_id, source, self._config)
            finally:
                with self._lock:
                    self._pending -= 1

        self._executor.submit(run)

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
    config = replace(config, daemon_port=args.port)

    server = DaemonServer(config)
    server.start()


if __name__ == "__main__":
    main()
