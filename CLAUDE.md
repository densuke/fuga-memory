# fuga-memory: Claude Code Instructions

このドキュメントは、Claude Code に向けた開発指針および設定ガイドです。

## 開発指針

技術スタック、ディレクトリ構造、開発コマンド、および規約については、ルートディレクトリの **`AGENTS.md`** を参照してください。

## Claude Code 専用設定

### `~/.claude/settings.json` への追加例

> **注意**: `/path/to/fuga-memory` は fuga-memory をクローンしたディレクトリの絶対パスに置き換えてください。

```json
{
  "mcpServers": {
    "fuga-memory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/fuga-memory", "fuga-memory", "serve"]
    }
  },
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run --project /path/to/fuga-memory fuga-memory save --stdin --session-id \"${CLAUDE_SESSION_ID:-unknown}\" --source claude_code",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

### 記憶システムの使い分け（Claude Code 向けヒント）

fuga-memory（長期記憶）と Claude のコンテキスト（短期記憶）は目的が異なります。

| ユーザーの言葉 | 使うべき記憶 | 操作 |
|---|---|---|
| 「前回」「以前」「過去に」「別のセッションで」 | fuga-memory | search_memory で検索してから回答 |
| 「長期記憶に残して」「重要なので記録して」 | fuga-memory | save_memory で明示的に保存 |
| それ以外の現セッション内の話題 | コンテキスト | そのまま会話を続ける |

- セッション終了時は Stop フックが自動で fuga-memory に保存するため、通常「記憶して」と言う必要はありません。
- 「前回」「以前」などのキーワードが出たら、まず `search_memory` を呼んでから回答してください。
