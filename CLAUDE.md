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
- **PRマージ**: CIグリーン確認後、必ずユーザーの明示的な承認を得てからマージする（自動マージ禁止）
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

### デーモンの動作

`fuga-memory save` コマンドはバックグラウンドデーモン経由で動作する:

1. **初回呼び出し時**: デーモンプロセスを自動起動し `/health` が応答するまで待機（最大10秒）
2. **リクエスト送信**: `POST /save` を送り 202 を受けたら即座に返る（ブロックなし）
3. **バックグラウンド処理**: デーモンが埋め込み生成・DB 書き込みを非同期で実行
4. **自動終了**: 最後のリクエストから `daemon_idle_timeout`（デフォルト10分）経過後に自動終了
5. **セッション継続**: Claude 再起動後もデーモンが生きていれば透過的に再利用される
6. **フォールバック**: デーモン起動失敗時は従来の直接実行に自動切り替え

### デーモン設定

設定ファイル（`config.toml`）または環境変数で変更可能:

```toml
[fuga-memory]
daemon_port = 18520          # デーモンの待ち受けポート（デフォルト: 18520）
daemon_idle_timeout = 600    # アイドル自動終了までの秒数（デフォルト: 600秒=10分）
```

環境変数: `FUGA_MEMORY_DAEMON_PORT`、`FUGA_MEMORY_DAEMON_IDLE_TIMEOUT`

### デーモン管理

```bash
# デーモンの状態確認
curl http://127.0.0.1:18520/health

# 手動停止
curl -X POST http://127.0.0.1:18520/shutdown

# ログ確認
cat ~/.local/share/fuga-memory/daemon.log
```

## 設定ファイル

`config_file_paths()` が以下の順で探索し、最初に見つかったものを使用する:

| 優先度 | パス |
|--------|------|
| 1（macOS） | `~/Library/Application Support/fuga-memory/config.toml` |
| 2 | `$XDG_CONFIG_HOME/fuga-memory/config.toml`（未設定時は `~/.config/fuga-memory/config.toml`） |
| 3 | `~/.fuga-memory.toml` |

読み込み優先順位: デフォルト値 < 設定ファイル < 環境変数

テンプレートは `config.toml.example` および `.env.example` を参照。

## 設計メモ

- **埋め込み次元**: 768（ruri-v3-310m）
- **プレフィックス規則**: 保存時=`検索文書: `、クエリ時=`検索クエリ: `
- **WALモード**: 並行読み取り対応、書き込みはシリアライズ
- **時間減衰**: `score * 0.5^(age_days/30)`
- **RRF**: `score = sum(1/(k+rank_i)) * decay`（k=60）
