# fuga-memory

Claude Code（および他のLLM）向け長期記憶MCPサーバー。

## 技術スタック

- **言語**: Python 3.13（uv管理）
- **MCP**: fastmcp
- **DB**: SQLite + FTS5（trigram）+ sqlite-vec
- **埋め込み**: sentence-transformers + cl-nagoya/ruri-v3-310m（ONNXバックエンド）
- **並行処理**: ThreadPoolExecutor（推論・DB操作）+ asyncio（MCPリクエスト受付）
- **検索統合**: RRF（k=60）+ 時間減衰（半減期30日）

## セットアップ

```bash
uv sync
```

初回サーバー起動時にruri-v3-310mモデルを自動ダウンロード（約600MB）。

## コマンド

```bash
# テスト実行（カバレッジ付き）
uv run pytest

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# 型チェック
uv run mypy src/

# MCPサーバー起動
uv run fuga-memory serve

# ファイルから記憶を保存
uv run fuga-memory save --file <path> --session-id <id>

# CLI検索
uv run fuga-memory search <query>
```

## ディレクトリ構造

```
src/fuga_memory/
├── cli.py          # CLIエントリーポイント（click）
├── server.py       # MCPサーバー定義（fastmcp）
├── config.py       # 設定（DBパス、モデル名、スレッド数等）
├── db/
│   ├── connection.py   # SQLite接続管理（WAL、sqlite-vec拡張）
│   ├── schema.py       # DDL（memoriesテーブル、FTS5、vec仮想テーブル）
│   └── repository.py   # CRUD操作
├── embedding/
│   ├── loader.py   # バックグラウンドモデルロード
│   └── encoder.py  # テキスト→768次元ベクトル
└── search/
    ├── fts.py      # FTS5検索
    ├── vector.py   # ベクトル検索
    ├── fusion.py   # RRF統合
    └── decay.py    # 時間減衰スコア計算

tests/
├── unit/           # ユニットテスト（in-memory DB、モックモデル）
├── integration/    # 統合テスト（パイプライン全体）
└── e2e/            # E2Eテスト（MCPプロトコルレベル）
```

## 開発ルール

- **TDD**: テストを先に書く（テストと実装は完全分離）
- **カバレッジ**: 80%以上必須
- **コミット**: Conventional Commits（`feat:`, `fix:`, `refactor:`, `docs:`, `test:`）
- **ブランチ**: GitHub Flow（feature/* → main PR）
- **ファイルサイズ**: 200-400行目安、最大800行
- **型ヒント**: 全関数に必須

## MCP ツール仕様

| ツール | 引数 | 説明 |
|--------|------|------|
| `save_memory` | `content: str, session_id: str, source: str = "manual"` | 記憶を保存 |
| `search_memory` | `query: str, top_k: int = 5` | ハイブリッド検索 |
| `list_sessions` | `limit: int = 20` | セッション一覧 |

## Claude Code 設定例

`~/.claude/settings.json` に追加：

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
            "command": "uv run --project /path/to/fuga-memory fuga-memory save --stdin --session-id \"${sessionId}\"",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

## 設計メモ

- **埋め込み次元**: 768（ruri-v3-310m）
- **プレフィックス規則**: 保存時=`検索文書: `、クエリ時=`検索クエリ: `
- **WALモード**: 並行読み取り対応、書き込みはシリアライズ
- **時間減衰**: `score * 0.5^(age_days/30)`
- **RRF**: `score = sum(1/(k+rank_i)) * decay`（k=60）
