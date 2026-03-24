"""Click を使った CLI エントリーポイント。

コマンド:
  serve  - MCP サーバーを起動する（stdio transport）
  search - 記憶を検索して結果を表示する
  save   - 記憶を保存する
"""

from __future__ import annotations

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
@click.argument("content")
@click.option("--session-id", required=True, help="セッション識別子。")
@click.option(
    "--source",
    default="manual",
    show_default=True,
    help="記憶のソース。",
)
def save(content: str, session_id: str, source: str) -> None:
    """記憶を保存する。"""
    result = _server.save_memory(content, session_id=session_id, source=source)
    click.echo(f"保存しました: id={result['id']}  status={result['status']}")
