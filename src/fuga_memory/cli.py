"""Click を使った CLI エントリーポイント。

コマンド:
  serve  - MCP サーバーを起動する（stdio transport）
  search - 記憶を検索して結果を表示する
  save   - 記憶を保存する（引数・--stdin・--file の3方式）
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click

from fuga_memory import server as _server
from fuga_memory.config import Config
from fuga_memory.daemon.client import send_save_request
from fuga_memory.warnings import suppress_warnings


def _to_localtime(utc_str: str) -> str:
    """UTC ISO 文字列をシステムのローカルタイムに変換して返す。

    Args:
        utc_str: 末尾 'Z' の UTC ISO 8601 文字列（例: '2026-03-25T00:12:33Z'）

    Returns:
        ローカルタイムの文字列（例: '2026-03-25 09:12:33 JST'）
    """
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


_MAX_INPUT_BYTES = 1_048_576  # 1MB


def _read_stdin_limited(max_bytes: int = _MAX_INPUT_BYTES) -> str:
    """stdin をチャンク読み込みし、上限を超えた場合はエラーを発生させる。

    Args:
        max_bytes: 読み込み上限バイト数

    Returns:
        読み込んだテキスト

    Raises:
        click.UsageError: 入力が上限を超えた場合
    """
    raw_chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sys.stdin.buffer.read(8192)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise click.UsageError(f"入力が最大サイズ（{max_bytes:,} バイト）を超えています")
        raw_chunks.append(chunk)
    try:
        return b"".join(raw_chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise click.UsageError(
            f"入力のデコードに失敗しました（UTF-8ではありません）: {exc}"
        ) from exc


@click.group()
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="デバッグモード（ライブラリ警告を抑制しない）。",
)
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """fuga-memory: Claude Code 向け長期記憶 MCP サーバー。"""
    ctx.ensure_object(dict)
    # CLIフラグと設定ファイル/環境変数（Config.debug）の両方を参照
    config = Config.load()
    if not (debug or config.debug):
        suppress_warnings()


@main.command()
def serve() -> None:
    """MCP サーバーを起動する（stdio transport）。"""
    _server.mcp.run()


@main.command()
@click.argument("query")
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="返す最大件数（1 以上）。",
)
def search(query: str, top_k: int) -> None:
    """記憶を検索して結果を表示する。"""
    results = _server.search_memory(query, top_k=top_k)

    if not results:
        click.echo("記憶が見つかりませんでした。")
        return

    for i, item in enumerate(results, start=1):
        click.echo(f"[{i}] score={item['score']:.4f}  session={item['session_id']}")
        click.echo(f"    {item['content']}")
        click.echo(f"    source={item['source']}  created_at={_to_localtime(item['created_at'])}")
        click.echo()


@main.command()
@click.argument("content", required=False)
@click.option("--session-id", required=True, help="セッション識別子。")
@click.option(
    "--source",
    default="manual",
    show_default=True,
    help="記憶のソース。",
)
@click.option(
    "--stdin",
    "read_stdin",
    is_flag=True,
    help="標準入力からコンテンツを読む（Stop フック等で使用）。",
)
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="ファイルからコンテンツを読む。",
)
def save(
    content: str | None,
    session_id: str,
    source: str,
    read_stdin: bool,
    file_path: Path | None,
) -> None:
    """記憶を保存する。

    コンテンツの指定方法（いずれか1つ）:

    \b
      fuga-memory save "内容" --session-id <id>
      fuga-memory save --stdin --session-id <id>   # パイプ入力
      fuga-memory save --file memo.txt --session-id <id>
    """
    input_count = sum([content is not None, read_stdin, file_path is not None])
    if input_count > 1:
        raise click.UsageError("content 引数、--stdin、--file は同時に指定できません。")
    if input_count == 0:
        raise click.UsageError("content 引数、--stdin、--file のいずれかを指定してください。")

    if read_stdin:
        body = _read_stdin_limited()
    elif file_path is not None:
        # stat() は /dev/zero 等の特殊ファイルで不正確なため、実際に読んでバイト数を確認する
        with file_path.open("rb") as f:
            raw = f.read(_MAX_INPUT_BYTES + 1)
        if len(raw) > _MAX_INPUT_BYTES:
            raise click.UsageError(
                f"ファイルが最大サイズ（{_MAX_INPUT_BYTES:,} バイト）を超えています"
            )
        try:
            body = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise click.UsageError(
                f"ファイルのデコードに失敗しました（UTF-8ではありません）: {exc}"
            ) from exc
    else:
        body = str(content)

    config = Config.load()
    try:
        send_save_request(body, session_id, source, config=config)
        click.echo("保存キューに追加しました (バックグラウンド処理中)")
    except Exception as exc:
        logging.getLogger(__name__).warning("デーモン経由の保存に失敗、直接実行します: %s", exc)
        result = _server.save_memory(body, session_id=session_id, source=source)
        click.echo(f"保存しました: id={result['id']}  status={result['status']}")
