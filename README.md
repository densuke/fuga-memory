# fuga-memory

Claude Code（および他のLLM）向け長期記憶MCPサーバー。

会話内容をSQLiteに保存し、FTS5（全文検索）+ ベクトル検索（ruri-v3-310m）のハイブリッド検索で関連する記憶を呼び出します。

## 特徴

- **外部依存なし**: 全データをSQLite単一ファイルに格納
- **ハイブリッド検索**: FTS5キーワード検索 + ベクトル検索をRRFで統合
- **時間減衰**: 古い記憶のスコアを半減期30日で段階的に低下
- **軽量推論**: ONNXバックエンドでruri-v3-310mをCPUで動作
- **MCP対応**: Claude Code / Gemini / Copilot など複数LLMで共有可能
- **自動保存**: Claude CodeのStopフックでセッション終了時に自動保存

## インストール

```bash
git clone https://github.com/densuke/fuga-memory
cd fuga-memory
uv sync
```

初回サーバー起動時にruri-v3-310mモデルを自動ダウンロード（約600MB）。

## 使い方

### MCPサーバー起動

```bash
uv run fuga-memory serve
```

### CLI検索

```bash
uv run fuga-memory search "Rustのlifetimeについて"
```

### ファイルから保存

```bash
uv run fuga-memory save --file conversation.txt --session-id my-session
```

## 設定

### 設定ファイル（推奨）

`config.toml.example` をコピーして以下のいずれかに配置してください（先に見つかったものが使用されます）:

| OS | パス |
|----|------|
| macOS | `~/Library/Application Support/fuga-memory/config.toml` |
| Linux | `~/.config/fuga-memory/config.toml`（`$XDG_CONFIG_HOME` 対応） |
| 共通 | `~/.fuga-memory.toml` |

```bash
# macOS の場合
mkdir -p ~/Library/Application\ Support/fuga-memory
cp config.toml.example ~/Library/Application\ Support/fuga-memory/config.toml
```

設定ファイルの例:

```toml
[fuga-memory]
db_path = "~/.local/share/fuga-memory/memories.db"
decay_halflife_days = 14  # 記憶の半減期を2週間に短縮
default_top_k = 10
```

### 環境変数

設定ファイルより優先されます。Docker / CI などファイル配置が難しい環境向けです。
詳細は `.env.example` を参照してください。

| 変数 | デフォルト |
|------|-----------|
| `FUGA_MEMORY_DB_PATH` | `~/.local/share/fuga-memory/memories.db` |
| `FUGA_MEMORY_MODEL_NAME` | `cl-nagoya/ruri-v3-310m` |
| `FUGA_MEMORY_THREAD_WORKERS` | CPU数÷2 |
| `FUGA_MEMORY_RRF_K` | `60` |
| `FUGA_MEMORY_DECAY_HALFLIFE_DAYS` | `30` |
| `FUGA_MEMORY_DEFAULT_TOP_K` | `5` |

## Claude Code 設定

`~/.claude/settings.json` に追加：

```json
{
  "mcpServers": {
    "fuga-memory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/fuga-memory", "fuga-memory", "serve"]
    }
  }
}
```

Stopフックで自動保存する場合は `CLAUDE.md` を参照。

## 技術スタック

- Python 3.13, uv, fastmcp
- SQLite + FTS5（trigram）+ sqlite-vec
- sentence-transformers + cl-nagoya/ruri-v3-310m（ONNXバックエンド）
- ThreadPoolExecutor + asyncio

## ライセンス

MIT
