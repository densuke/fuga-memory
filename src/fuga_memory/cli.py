"""Click を使った CLI エントリーポイント。

コマンド:
  serve  - MCP サーバーを起動する（stdio transport）
  search - 記憶を検索して結果を表示する
  save   - 記憶を保存する（引数・--stdin・--file の3方式）
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from fuga_memory import server as _server


@click.group()
def main() -> None:
    """fuga-memory: Claude Code 向け長期記憶 MCP サーバー。"""


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
        click.echo(f"    source={item['source']}  created_at={item['created_at']}")
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
        body = sys.stdin.read()
    elif file_path is not None:
        body = file_path.read_text()
    else:
        body = str(content)

    result = _server.save_memory(body, session_id=session_id, source=source)
    click.echo(f"保存しました: id={result['id']}  status={result['status']}")
